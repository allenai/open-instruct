import beaker
from beaker import Beaker, ExperimentSpec, TaskSpec
from argparse import ArgumentParser
from pathlib import Path


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=str,
        help=(
            "File on NFS with train instances. Specify relative to "
            "/net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/science-adapt/data/davidw"
        ),
    )
    parser.add_argument(
        "--num_gpus", type=int, help="Number of GPUs to use.", default=4
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        help="Number of cpus for data processing",
        default=16,
    )
    parser.add_argument("--model_size", type=str, default="7B")
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
        help="If passed, continue from final tulu checkpoint rather than starting from base lm.",
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
        ]
    else:
        cluster = [args.cluster]

    # File paths.
    conda_root = "/net/nfs.cirrascale/allennlp/davidw/miniconda3"
    open_instruct_dir = "/net/nfs.cirrascale/allennlp/davidw/proj/open-instruct"
    data_dir = "/net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/science-adapt/data/davidw"

    # Gradient accumulation steps.
    gradient_acc_steps = int(
        args.total_batch_size / args.num_gpus / args.batch_size_per_gpu
    )

    ####################

    # Settings for continued finetuning vs. single-stage.

    if args.continued_finetune:
        if args.model_size != "7B":
            raise Exception("Haven't trained a model other than 7B yet.")
        # Keep training from model trained on Tulu data; double the warmup.
        beaker_model_path = "01HKG46RNVAP3NSHNDH019R5KB"
        warmup_ratio = 0.06
    else:
        beaker_model_path = f"Yizhongw03/hf_llama2_model_{args.model_size}"
        warmup_ratio = 0.03

    ####################

    # Build the command to execute.

    conda_command = [
        f"{conda_root}/bin/conda",
        "run",
        "--cwd",
        open_instruct_dir,
        "--no-capture-output",
        "-n",
        "open-instruct",
    ]
    accelerate_command = [
        "accelerate",
        "launch",
        "--mixed_precision",
        "bf16",
        "--num_machines",
        "1",
        "--num_processes",
        args.num_gpus,
        "--use_deepspeed",
        "--deepspeed_config_file",
        "ds_configs/stage3_no_offloading_accelerate.conf",
    ]
    train_command = [
        "open_instruct/finetune.py",
        "--model_name_or_path",
        "/hf_llama_models",
        "--use_flash_attn",
        "--tokenizer_name",
        "/hf_llama_models",
        "--use_slow_tokenizer",
        "--train_file",
        f"{data_dir}/{args.train_file}",
        "--max_seq_length",
        "4096",
        "--preprocessing_num_workers",
        args.preprocessing_num_workers,
        "--per_device_train_batch_size",
        args.batch_size_per_gpu,
        "--gradient_accumulation_steps",
        gradient_acc_steps,
        "--learning_rate",
        "2e-5",
        "--lr_scheduler_type",
        "linear",
        "--warmup_ratio",
        str(warmup_ratio),
        "--weight_decay",
        "0.",
        "--num_train_epochs",
        "2",
        "--output_dir",
        "/output/",
        "--with_tracking",
        "--report_to",
        "wandb",
        "--logging_steps",
        1,
    ]
    # Full command.
    command = conda_command + accelerate_command + train_command

    ####################

    # Set Beaker datasets

    datasets = [  # Magic to get NFS mounted.
        beaker.DataMount(
            source=beaker.DataSource(host_path="/net/nfs.cirrascale"),
            mount_path="/net/nfs.cirrascale",
        ),
        beaker.DataMount(
            source=beaker.DataSource(beaker=beaker_model_path),
            mount_path="/hf_llama_models",
        ),
    ]

    ####################

    # Environment variables.

    env_vars = [
        beaker.EnvVar(name="NFS_HOME", value="/net/nfs.cirrascale/allennlp/davidw"),
        beaker.EnvVar(
            name="HF_HOME",
            value="/net/nfs.cirrascale/allennlp/davidw/cache/huggingface",
        ),
        beaker.EnvVar(
            name="CONDA_ENVS_DIRS",
            value=f"{conda_root}/envs",
        ),
        beaker.EnvVar(
            name="CONDA_PKGS_DIRS",
            value=f"{conda_root}/pkgs",
        ),
        beaker.EnvVar(name="CUDA_DEVICE_ORDER", value="PCI_BUS_ID"),
        beaker.EnvVar(name="WANDB_PROJECT", value="science-adapt"),
    ]

    ####################

    # Create experiment spec and task.

    task_name = Path(args.train_file).stem
    training_strategy = "continued" if args.continued_finetune else "1-stage"

    tasks = [
        TaskSpec(
            name=f"science_adapt_{training_strategy}_{task_name}",
            image=beaker.ImageSource(beaker="ai2/cuda11.8-cudnn8-dev-ubuntu20.04"),
            command=command,
            result=beaker.ResultSpec(path="/output"),
            datasets=datasets,
            context=beaker.TaskContext(priority=beaker.Priority("high")),  # Priority.
            constraints=beaker.Constraints(cluster=cluster),
            env_vars=env_vars,
            resources=beaker.TaskResources(
                gpu_count=args.num_gpus, cpu_count=args.preprocessing_num_workers
            ),
        ),
    ]

    spec = ExperimentSpec(
        description=f"Science adapt {training_strategy} finetuning for {task_name}",
        tasks=tasks,
    )
    workspace_name = "ai2/science-adapt"

    # Make the experiment and run it.
    beaker_client = Beaker.from_env(default_workspace=workspace_name)
    beaker_client.experiment.create(
        spec=spec,
        workspace=workspace_name,
    )


def main():
    args = get_args()
    kickoff(args)


if __name__ == "__main__":
    main()
