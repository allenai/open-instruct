import gc
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import load_dataset
from huggingface_hub import HfApi
from rich.pretty import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    get_scheduler,
)
from vllm import SamplingParams, LLM

from open_instruct.dataset_processor import (
    CHAT_TEMPLATES,
    INPUT_IDS_PROMPT_KEY,
    DatasetConfig,
    SFTDatasetProcessor,
    SimpleGenerateCollator,
    visualize_token,
)
from open_instruct.model_utils import (
    ModelConfig,
    batch_generation,
    batch_generation_vllm,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    print_rich_single_line_metrics,
    print_rich_table,
    save_with_accelerate,
    truncate_response,
    unwrap_model_for_generation,
)
# from open_instruct.online_eval import evaluate_vllm
from open_instruct.utils import (
    ArgumentParserPlus,
    get_wandb_tags,
    maybe_use_ai2_wandb_entity,
)
from open_instruct.vllm_utils import vllm_single_gpu_patch

api = HfApi()
INVALID_LOGPROB = 1.0


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: Optional[str] = None
    """A unique name of this run"""

    # wandb and HF tracking configs
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "open_instruct_internal"
    """The wandb's project name"""
    wandb_entity: Optional[str] = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = False
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: Optional[str] = None
    """Where to save the model"""

    # optimizer args
    eps: float = 1e-5
    """The epsilon value for the optimizer"""
    learning_rate: float = 2e-5
    """The initial learning rate for AdamW optimizer."""
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "linear"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""
    gradient_checkpointing: bool = True
    """Whether to use gradient checkpointing"""

    # various batch sizes
    num_train_epochs: int = 1
    """Number of epochs to train"""
    gradient_accumulation_steps: int = 8
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: Optional[int] = 1
    """The forward batch size per device (local_micro_batch_size)"""
    per_device_eval_batch_size: Optional[int] = 1
    """The forward batch size per device for evaluation (local_micro_batch_size)"""
    total_episodes: Optional[int] = 100000
    """The total number of episodes in the dataset"""
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    num_training_steps: Optional[int] = None
    """The number of training_steps to train"""
    num_evals: int = 4
    """The number of evaluations to run throughout training"""
    eval_freq: Optional[int] = None
    """The frequency of evaluation steps"""
    local_dataloader_batch_size: Optional[int] = None
    """The batch size per GPU for the dataloader"""

    # online settings
    num_epochs: int = 4
    """the number of epochs to train"""
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: Optional[int] = None
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""
    local_rollout_forward_batch_size: int = 64
    """per rank no grad forward pass in the rollout phase"""
    reward_model_path: str = "EleutherAI/pythia-160m"
    """the path to the reward model"""

    # generation config
    response_length: int = 53
    """the length of the response"""
    stop_token: Optional[Literal["eos", "period"]] = None
    """the stop token"""
    stop_token_id: Optional[int] = None
    """the truncation token id"""
    min_response_length: int = 0
    """stop only after this many tokens"""
    temperature: float = 0.7
    """the sampling temperature"""
    penalty_reward_value: float = -1.0
    """the reward value for responses that do not contain `stop_token_id`"""
    non_stop_penalty: bool = False
    """whether to penalize responses that do not contain `stop_token_id`"""

    # online DPO specific args
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    num_generation_per_prompt: int = 2
    """the number of generations per prompt (currently only support 2)"""
    loss_type: Literal["sigmoid", "ipo"] = "sigmoid"
    """the loss type for the DPO algorithm"""

    # vLLM settings. NOTE: currently we need to place the vLLM model on a separate GPU
    # for generation to work properly because vLLM would pre-alocate the memory.
    # To do so, we would need to do a moneky patch `vllm_single_gpu_patch` to make sure
    # the vLLM model is placed on the correct GPU.
    vllm_device: str = "cuda:1"
    """the device placement of the vllm model; typically we place the vllm model on a decicated GPU"""
    vllm_gpu_memory_utilization: float = 0.8
    """the GPU memory utilization of the vllm model; passed to `gpu_memory_utilization` to the `vLLM` instance"""


def calculate_runtime_args_and_accelerator(args: Args, model_config: ModelConfig) -> Accelerator:
    """calculate (in-place) runtime args such as the effective batch size, word size, etc."""
    # set up accelerator and a unique run name with timestamp
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    time_int = broadcast(time_tensor, 0).item()
    args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"

    # calculate runtime config
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
    args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    args.mini_batch_size = exact_div(
        args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
    )
    args.local_mini_batch_size = exact_div(
        args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
    )
    args.num_training_steps = args.total_episodes // args.batch_size
    args.eval_freq = max(1, args.num_training_steps // args.num_evals)
    # DPO logic: repeats the same prompt `num_generation_per_prompt` times
    args.local_dataloader_batch_size = exact_div(
        args.local_batch_size,
        args.num_generation_per_prompt,
        "`local_batch_size` must be a multiple of `num_generation_per_prompt`",
    )
    if args.push_to_hub:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = f"{args.exp_name}__{model_config.model_name_or_path.replace('/', '_')}"
        if args.hf_entity is None:
            args.hf_entity = api.whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"

    if args.with_tracking:
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()
    return accelerator


def main(args: Args, dataset_config: DatasetConfig, model_config: ModelConfig):
    accelerator = calculate_runtime_args_and_accelerator(args, model_config)
    local_seed = args.seed + accelerator.process_index

    # set up experiment tracking and seeds
    if accelerator.is_main_process:
        if args.with_tracking:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config={**asdict(args), **asdict(dataset_config), **asdict(model_config)},
                name=args.run_name,
                save_code=True,
                tags=[args.exp_name] + get_wandb_tags(),
            )
        writer = SummaryWriter(f"runs/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    device = torch.device(f"cuda:{accelerator.local_process_index}")
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True

    # create a tokenizer (pad from right)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, padding_side="right")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # NOTE: we do not resize the embedding
    tokenizer.chat_template = CHAT_TEMPLATES[dataset_config.chat_template]

    # create the dataset
    dataset = load_dataset(dataset_config.dataset_name)
    dataset_processor = SFTDatasetProcessor(tokenizer=tokenizer, config=dataset_config)
    dataset_processor.sanity_check_(dataset)
    with accelerator.main_process_first():
        dataset = dataset_processor.tokenize(dataset)
        dataset = dataset_processor.filter(dataset)
    train_dataset = dataset[dataset_config.dataset_train_split]
    eval_dataset = dataset[dataset_config.dataset_eval_split]

    # some more runtime logging
    if accelerator.is_main_process:
        pprint([args, dataset_config, model_config])
        visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)
        if args.with_tracking:
            # upload the visualized token length
            dataset_processor.get_token_length_visualization(
                dataset, save_path=f"runs/{args.run_name}/token_length.png"
            )
            wandb.log({"token_length": wandb.Image(f"runs/{args.run_name}/token_length.png")})

    # create the model and optimizer
    policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    ref_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    reward_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    model = policy
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    for module in [model, ref_model, reward_model]:
        disable_dropout_in_model(module)
    if args.stop_token:
        if args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        if args.stop_token == "period":
            args.stop_token_id = tokenizer.encode(".")[0]
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.eps)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_training_steps * args.num_train_epochs,
    )
    data_collator = SimpleGenerateCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.local_dataloader_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True,  # needed; otherwise the last batch will be of ragged shape
    )
    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    torch.manual_seed(local_seed)

    # deepspeed setup
    is_deepspeed_enabled = getattr(accelerator.state, "deepspeed_plugin", None) is not None
    mixed_precision = accelerator.state.mixed_precision
    if is_deepspeed_enabled:
        reward_model = prepare_deepspeed(reward_model, args.per_device_train_batch_size, mixed_precision)
        ref_model = prepare_deepspeed(ref_model, args.per_device_train_batch_size, mixed_precision)
    else:
        reward_model = reward_model.to(device)
        ref_model = ref_model.to(device)

    # online generation config
    def repeat_generator():
        while True:
            yield from dataloader

    iter_dataloader = iter(repeat_generator())
    generation_config = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
    )
    llm = None # a dummy value
    if accelerator.is_main_process:
        vllm_single_gpu_patch()
        llm = LLM(
            model=model_config.model_name_or_path,
            revision="main",
            tokenizer_revision="main",
            tensor_parallel_size=1,
            device=args.vllm_device,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )
        print("🔥🔥🔥 vllm loaded")
    # set device again because LLM calls `torch.cuda.set_device`
    # NOTE: do not remove this; otherwise accelerate hangs.
    torch.cuda.set_device(device)

    g_vllm_responses = torch.zeros((args.batch_size, args.response_length), device=device, dtype=torch.long)

    # set up the metrics and initial states
    stats_shape = (args.num_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
    loss_stats = torch.zeros(stats_shape, device=device)
    chosen_rewards_stats = torch.zeros(stats_shape, device=device)
    rejected_rewards_stats = torch.zeros(stats_shape, device=device)
    chosen_logprobs_stats = torch.zeros(stats_shape, device=device)
    rejected_logprobs_stats = torch.zeros(stats_shape, device=device)
    episode = 0
    model.train()

    # training loop
    start_time = time.time()
    for training_step in range(1, args.num_training_steps + 1):
        episode += 1 * args.batch_size
        scheduler.step()
        data = next(iter_dataloader)

        with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
            
            # (optionally) evaluate the model
            generation_model = unwrapped_model
            # if args.num_evals > 0 and (training_step - 1) % args.eval_freq == 0:
            #     table = evaluate_vllm(
            #         llm,
            #         generation_model,
            #         reward_model,
            #         accelerator,
            #         args.stop_token_id,
            #         eval_dataset,
            #         data_collator,
            #         tokenizer,
            #         args.response_length,
            #         device,
            #         args.per_device_eval_batch_size,
            #         args.world_size,
            #     )
            #     for key in table:
            #         table[key] = gather_object(table[key])
            #     df = pd.DataFrame(table)
            #     if accelerator.is_main_process:
            #         if args.with_tracking:
            #             wandb.log({"sample_completions": wandb.Table(dataframe=df)})
            #         else:
            #             print_rich_table(df)
            #     del table, df

            with torch.no_grad():
                try:
                    torch.cuda.synchronize()
                    queries = data[INPUT_IDS_PROMPT_KEY].to(device)
                    queries = queries.repeat(args.num_generation_per_prompt, 1)
                    context_length = queries.shape[1]
                    responses = []
                    postprocessed_responses = []
                    logprobs = []
                    ref_logprobs = []
                    scores = []
                    sequence_lengths = []
                    local_vllm_responses = batch_generation_vllm(
                        llm,
                        generation_config,
                        accelerator,
                        queries,
                        generation_model,
                        g_vllm_responses,
                        tokenizer.pad_token_id,
                        args.response_length,
                        device,
                    )
                    training_time_start = time.time()
                except RuntimeError as e:
                    print(f"Error in generation: {e}")
                query_responses = torch.cat((queries, local_vllm_responses), 1)
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    try:
                        torch.cuda.synchronize()
                        query = queries[i : i + args.local_rollout_forward_batch_size]
                        query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                        response = query_response[:, context_length:]
                        output = forward(unwrapped_model, query_response, tokenizer.pad_token_id)
                        logits = output.logits[:, context_length - 1 : -1]
                        logits /= args.temperature + 1e-7
                        all_logprob = F.log_softmax(logits, dim=-1)
                        logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                        del output, logits, all_logprob
                        torch.cuda.empty_cache()
                    except RuntimeError as e:
                        print(f"Error in forward: {e}")

                    try:
                        torch.cuda.synchronize()
                        ref_output = forward(ref_model, query_response, tokenizer.pad_token_id)
                        ref_logits = ref_output.logits[:, context_length - 1 : -1]
                        ref_logits /= args.temperature + 1e-7
                        ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                        ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                        del ref_output, ref_logits, ref_all_logprob
                        torch.cuda.empty_cache()
                    except RuntimeError as e:
                        print(f"Error in refforward: {e}")

                    try:
                        torch.cuda.synchronize()
                        # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                        postprocessed_response = response
                        if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                            postprocessed_response = truncate_response(args.stop_token_id, tokenizer.pad_token_id, response)

                        # Response Processing 2. run reward model on the truncated responses
                        postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                        sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                        _, score, _ = get_reward(
                            reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
                        )

                        responses.append(response)
                        postprocessed_responses.append(postprocessed_response)
                        logprobs.append(logprob)
                        ref_logprobs.append(ref_logprob)
                        sequence_lengths.append(sequence_length)
                        scores.append(score)
                    except RuntimeError as e:
                        print(f"Error in truncate_response: {e}")
                try:
                    torch.cuda.synchronize()
                    responses = torch.cat(responses, 0)
                    postprocessed_responses = torch.cat(postprocessed_responses, 0)
                    logprobs = torch.cat(logprobs, 0)
                    ref_logprobs = torch.cat(ref_logprobs, 0)
                    sequence_lengths = torch.cat(sequence_lengths, 0)
                    scores = torch.cat(scores, 0)
                    del (logprob, ref_logprob, score)
                    gc.collect()
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    print(f"Error in cat: {e}")

                try:
                    torch.cuda.synchronize()
                    # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                    # responses not passing that filter will receive a low (fixed) score
                    # only query humans on responses that pass that filter
                    contain_stop_token = torch.any(postprocessed_responses == args.stop_token_id, dim=-1)
                    # NOTE: only apply the stop token filter if the response is long enough
                    # otherwise the model could learn to generate the first token as the stop token
                    contain_stop_token = contain_stop_token & (sequence_lengths >= args.min_response_length)
                    if args.non_stop_penalty:
                        scores = torch.where(contain_stop_token, scores, torch.full_like(scores, args.penalty_reward_value))

                    # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                    response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                    padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                    logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                    ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                    # 4. compute rewards
                    kl = logprobs - ref_logprobs
                    non_score_reward = -args.beta * kl
                    non_score_reward_sum = non_score_reward.sum(1)
                    rlhf_reward = scores + non_score_reward_sum

                    # num_examples should be same as args.local_batch_size divided by 2
                    num_examples = scores.size(0) // 2
                    first_half = scores[:num_examples]
                    second_half = scores[num_examples:]

                    num_examples_range = torch.arange(num_examples).to(scores.device)
                    chosen_indices = torch.where(
                        first_half >= second_half, num_examples_range.clone(), num_examples_range.clone() + num_examples
                    )
                    rejected_indices = torch.where(
                        first_half < second_half, num_examples_range.clone(), num_examples_range.clone() + num_examples
                    )
                    scores_margin = scores[chosen_indices] - scores[rejected_indices]
                except RuntimeError as e:
                    print(f"Error in chosen_indices: {e}")
        try:
            torch.cuda.synchronize()
            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for epoch_idx in range(args.num_epochs):
                b_inds = np.random.permutation(args.local_batch_size // args.num_generation_per_prompt)
                minibatch_idx = 0
                for mini_batch_start in range(
                    0,
                    args.local_batch_size // args.num_generation_per_prompt,
                    args.local_mini_batch_size // args.num_generation_per_prompt,
                ):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size // args.num_generation_per_prompt
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(
                        0,
                        args.local_mini_batch_size // args.num_generation_per_prompt,
                        args.per_device_train_batch_size,
                    ):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            chosen_mb_inds = chosen_indices[micro_batch_inds]
                            chosen_responses = responses[chosen_mb_inds]
                            rejected_mb_inds = rejected_indices[micro_batch_inds]
                            rejected_responses = responses[rejected_mb_inds]

                            concat_mb_inds = torch.cat((chosen_mb_inds, rejected_mb_inds), dim=0)
                            concat_query_responses = query_responses[concat_mb_inds]
                            concat_output = forward(model, concat_query_responses, tokenizer.pad_token_id)
                            num_examples = chosen_mb_inds.shape[0]
                            chosen_logits = concat_output.logits[:num_examples]
                            rejected_logits = concat_output.logits[num_examples:]

                            # chosen
                            chosen_logits = chosen_logits[:, context_length - 1 : -1]
                            chosen_logits /= args.temperature + 1e-7
                            chosen_all_logprobs = F.log_softmax(chosen_logits, dim=-1)
                            chosen_logprobs = torch.gather(chosen_all_logprobs, 2, chosen_responses.unsqueeze(-1)).squeeze(
                                -1
                            )
                            chosen_logprobs = torch.masked_fill(
                                chosen_logprobs, padding_mask[chosen_mb_inds], INVALID_LOGPROB
                            )
                            chosen_ref_logprobs = ref_logprobs[chosen_mb_inds]
                            chosen_logprobs_sum = (chosen_logprobs * ~padding_mask[chosen_mb_inds]).sum(1)
                            chosen_ref_logprobs_sum = (chosen_ref_logprobs * ~padding_mask[chosen_mb_inds]).sum(1)

                            # rejected
                            rejected_logits = rejected_logits[:, context_length - 1 : -1]
                            rejected_logits /= args.temperature + 1e-7
                            rejected_all_logprobs = F.log_softmax(rejected_logits, dim=-1)
                            rejected_logprobs = torch.gather(
                                rejected_all_logprobs, 2, rejected_responses.unsqueeze(-1)
                            ).squeeze(-1)
                            rejected_logprobs = torch.masked_fill(
                                rejected_logprobs, padding_mask[rejected_mb_inds], INVALID_LOGPROB
                            )
                            rejected_ref_logprobs = ref_logprobs[rejected_mb_inds]
                            rejected_logprobs_sum = (rejected_logprobs * ~padding_mask[rejected_mb_inds]).sum(1)
                            rejected_ref_logprobs_sum = (rejected_ref_logprobs * ~padding_mask[rejected_mb_inds]).sum(1)

                            pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
                            ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

                            logits = pi_logratios - ref_logratios

                            if args.loss_type == "sigmoid":
                                losses = -F.logsigmoid(args.beta * logits)
                            elif args.loss_type == "ipo":
                                losses = (logits - 1 / (2 * args.beta)) ** 2
                            else:
                                raise NotImplementedError(f"invalid loss type {args.loss_type}")

                            loss = losses.mean()
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                chosen_rewards = args.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
                                rejected_rewards = args.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
                                loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = loss
                                chosen_rewards_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    chosen_rewards.mean()
                                )
                                rejected_rewards_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    rejected_rewards.mean()
                                )
                                chosen_logprobs_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    chosen_logprobs_sum.mean()
                                )
                                rejected_logprobs_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    rejected_logprobs_sum.mean()
                                )
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # fmt: off
                    del (
                        loss, logits, concat_output, concat_query_responses,
                        chosen_logits, rejected_logits, chosen_logprobs, rejected_logprobs,
                        chosen_responses, rejected_responses,
                    )
                    # fmt: on
                    # del everything and empty cache
                    torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"Error in training: {e}")
        try:
            torch.cuda.synchronize()
            with torch.no_grad():
                g_chosen_reward = accelerator.gather(chosen_rewards_stats)
                g_rejected_reward = accelerator.gather(rejected_rewards_stats)
                metrics = {
                    "episode": episode,
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": episode / len(train_dataset),
                    "time/from_scratch": time.time() - start_time,
                    "time/training": time.time() - training_time_start,
                    "val/num_stop_token_ids": (responses == args.stop_token_id).sum().item(),
                    "objective/kl": accelerator.gather(kl.sum(1).mean()).mean().item(),
                    "objective/entropy": accelerator.gather((-logprobs).sum(1).mean()).mean().item(),
                    "objective/non_score_reward": accelerator.gather(non_score_reward_sum).mean().item(),
                    "objective/rlhf_reward": accelerator.gather(rlhf_reward).mean().item(),
                    "objective/scores": accelerator.gather(scores.mean()).mean().item(),
                    "objective/scores_margin": accelerator.gather(scores_margin.mean()).mean().item(),
                    "objective/loss": accelerator.gather(loss_stats).mean().item(),
                    "rewards/chosen": g_chosen_reward.mean().item(),
                    "rewards/rejected": g_rejected_reward.mean().item(),
                    "rewards/accuracies": (g_chosen_reward > g_rejected_reward).float().mean().item(),
                    "rewards/margins": (g_chosen_reward - g_rejected_reward).mean().item(),
                    "logps/chosen": accelerator.gather(chosen_logprobs_stats).mean().item(),
                    "logps/rejected": accelerator.gather(rejected_logprobs_stats).mean().item(),
                }
                if accelerator.is_main_process:
                    print_rich_single_line_metrics(metrics)
                    for key, value in metrics.items():
                        writer.add_scalar(key, value, episode)
            del (queries, responses, postprocessed_responses, logprobs, ref_logprobs, sequence_lengths, scores)
            del (metrics, kl, non_score_reward, rlhf_reward)
            del (g_chosen_reward, g_rejected_reward)
        except RuntimeError as e:
            print(f"Error in metrics: {e}")
        gc.collect()
        torch.cuda.empty_cache()

    # save model
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    original_tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    save_with_accelerate(
        accelerator,
        model,
        original_tokenizer,
        args.output_dir,
        False,
        args.push_to_hub,
        args.hf_repo_id,
        args.hf_repo_revision,
    )


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, DatasetConfig, ModelConfig))
    main(*parser.parse())
