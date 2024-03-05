import beaker
from beaker import Beaker, ExperimentSpec, TaskSpec
from argparse import ArgumentParser
from pathlib import Path


def remove_leading_whitespace(cmd):
    for i, char in enumerate(cmd):
        if char != " ":
            break

    leading_whitespace = " " * i
    return cmd.replace(leading_whitespace, "")


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=str,
        help=("File on NFS with train instances."),
    )
    parser.add_argument(
        "--num_gpus", type=int, help="Number of GPUs to use.", default=4
    )
    parser.add_argument(
        "--beaker_model_path",
        type=str,
        help="Beaker dataset with model we should start training from.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="If given, the name of the starting model.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        help="Number of cpus for data processing",
        default=16,
    )
    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--total_batch_size", type=int, default=128)
    parser.add_argument(
        "--cluster",
        type=str,
        default=None,
        help="Cluster to use. It not given, try AllenNLP and S2.",
    )
    parser.add_argument(
        "--continued_finetune",
        action="store_true",
        help="If passed, use train settings for continued finetuning.",
    )

    return parser.parse_args()


def kickoff(args):
    """
    Kick off beaker training run.
    """
    # Candidate clusters to use.
    if args.cluster is None:
        cluster = [
            "ai2/s2-cirrascale",
            "ai2/s2-cirrascale-l40",
            "ai2/allennlp-cirrascale",
            "ai2/mosaic-cirrascale"
        ]
    else:
        cluster = [args.cluster]

    # Gradient accumulation steps.
    gradient_acc_steps = int(
        args.total_batch_size / args.num_gpus / args.batch_size_per_gpu
    )

    ####################

    # Settings for continued finetuning vs. single-stage.

    if args.continued_finetune:
        warmup_ratio = 0.06
    else:
        warmup_ratio = 0.03

    ####################

    # Build the command to execute.

    task_command = ["/bin/sh", "-c"]
    # argument = f"""\
    #     accelerate launch \
    #     --mixed_precision bf16 \
    #     --num_machines 1 \
    #     --num_processes {args.num_gpus} \
    #     --use_deepspeed \
    #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    #     open_instruct/finetune.py \
    #     --model_name_or_path /model \
    #     --use_flash_attn \
    #     --use_lora \
    #     --lora_rank 64 \
    #     --lora_alpha 16 \
    #     --lora_dropout 0.1 \
    #     --tokenizer_name /model \
    #     --use_slow_tokenizer \
    #     --train_file {args.train_file} \
    #     --max_seq_length 4096 \
    #     --preprocessing_num_workers {args.preprocessing_num_workers} \
    #     --per_device_train_batch_size {args.batch_size_per_gpu} \
    #     --gradient_accumulation_steps {gradient_acc_steps} \
    #     --learning_rate 1e-4 \
    #     --lr_scheduler_type linear \
    #     --warmup_ratio {warmup_ratio} \
    #     --weight_decay 0. \
    #     --num_train_epochs 1 \
    #     --output_dir /output \
    #     --logging_steps 1 &&
    #     python open_instruct/merge_lora.py \
    #     --base_model_name_or_path /model \
    #     --lora_model_name_or_path /output \
    #     --output_dir /output/llama7b_lora_merged/ \
    #     """

    argument = f"""\
        accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes {args.num_gpus} \
        --use_deepspeed \
        --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
        open_instruct/finetune.py \
        --model_name_or_path {args.model_name} \
        --use_flash_attn \
        --use_lora \
        --lora_rank 64 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --tokenizer_name {args.model_name} \
        --use_slow_tokenizer \
        --train_file {args.train_file} \
        --max_seq_length 4096 \
        --preprocessing_num_workers {args.preprocessing_num_workers} \
        --per_device_train_batch_size {args.batch_size_per_gpu} \
        --gradient_accumulation_steps {gradient_acc_steps} \
        --learning_rate 1e-4 \
        --lr_scheduler_type cosine \
        --warmup_ratio {warmup_ratio} \
        --weight_decay 0. \
        --num_train_epochs 1 \
        --output_dir /output \
        --logging_steps 1"""

#  &&
#         python open_instruct/merge_lora.py \
#         --base_model_name_or_path {args.model_name} \
#         --lora_model_name_or_path /output \
#         --output_dir /output/tulu7b_lora_merged/ \
    argument = remove_leading_whitespace(argument)
    
# output/tulu_v2_${MODEL_SIZE}_lora/
    ####################

    # Set Beaker datasets

    datasets = [  # Magic to get NFS mounted.
        beaker.DataMount(
            source=beaker.DataSource(host_path="/net/nfs.cirrascale"),
            mount_path="/net/nfs.cirrascale",
        ),
        # beaker.DataMount(
        #     source=beaker.DataSource(beaker=args.beaker_model_path),
        #     mount_path="/model",
        # ),
    ]

    ####################

    # Environment variables.

    env_vars = [
        beaker.EnvVar(name="CUDA_DEVICE_ORDER", value="PCI_BUS_ID"),
        beaker.EnvVar(name="WANDB_PROJECT", value="safety-adapt"),
    ]

    ####################

    # Create experiment spec and task.

    task_name = Path(args.train_file).stem
    model_name = (
        args.beaker_model_path.split("/")[-1]
        if args.model_name is None
        else args.model_name.split("/")[-1]
    )
    training_strategy = "continued" if args.continued_finetune else "lora" #"lora_scratch" # #

    tasks = [
        TaskSpec(
            name=f"{training_strategy}-{model_name}-{task_name}-epoch1",
            image=beaker.ImageSource(beaker="davidw/open-instruct"),
            command=task_command,
            arguments=[argument],
            result=beaker.ResultSpec(path="/output"),
            datasets=datasets,
            context=beaker.TaskContext(priority=beaker.Priority("high")),  # Priority. preemptible
            constraints=beaker.Constraints(cluster=cluster),
            env_vars=env_vars,
            resources=beaker.TaskResources(
                gpu_count=args.num_gpus, cpu_count=args.preprocessing_num_workers
            ),
        ),
    ]

    spec = ExperimentSpec(
        description=f"Safety adapt {training_strategy} finetuning of {model_name} for {task_name}",
        tasks=tasks,
        budget="ai2/oe-adapt",
    )
    workspace_name = "ai2/safety-adapt"

    # Make the experiment and run it.
    beaker_client = Beaker.from_env(default_workspace=workspace_name)
    with beaker_client.session():
        beaker_client.experiment.create(
            spec=spec,
            workspace=workspace_name,
        )


def main():
    args = get_args()
    kickoff(args)


if __name__ == "__main__":
    main()
