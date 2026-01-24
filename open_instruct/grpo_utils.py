from dataclasses import dataclass, field
from typing import Literal

from open_instruct.utils import calibrate_checkpoint_state_dir, download_latest_checkpoint_from_gs, get_beaker_whoami


@dataclass
class ExperimentConfig:
    # Experiment
    exp_name: str = "grpo"
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: str | None = None
    """RUNTIME VALUE: A unique name of this run"""

    # Optimizer
    learning_rate: float = 2e-5
    """The initial learning rate for AdamW optimizer."""
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "linear"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""
    warmup_ratio: float = 0.0
    """Ratio of warmup steps to total steps (takes precedence over `warm_up_steps`)"""
    weight_decay: float = 0.0
    """Weight decay for AdamW if we apply some."""
    max_grad_norm: float = 1.0
    """Maximum gradient norm for gradient clipping."""
    set_weight_decay_on_bias_and_norm: bool = True
    """Whether to set weight decay on bias and norm layers"""
    fused_optimizer: bool = False
    """Whether to use fused optimizer"""

    # Batch sizes
    per_device_train_batch_size: int = 1
    """The forward batch size per device (local_micro_batch_size)"""
    total_episodes: int = 100000
    """The total number of episodes in the dataset"""
    world_size: int | None = None
    """RUNTIME VALUE: The number of processes (GPUs) to use for training ONLY"""
    num_training_steps: int | None = None
    """RUNTIME VALUE: The number of training_steps to train"""
    local_eval_every: int = 100
    """Run evaluation after this many training steps. This controls in-loop evals, which reuse the generation/reward verifier setup. Set to -1 to disable."""
    save_freq: int = 200
    """How many train steps to save the model"""
    backend_timeout: int = 120
    """Timeout for inference/training backends in minutes. Default is 2 hours (120 min)."""

    # Algorithm
    num_epochs: int = 1
    """the number of epochs to train"""
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    clip_lower: float = 0.2
    """the lower clip range"""
    clip_higher: float = 0.2
    """the higher clip range. Sometimes we want this to be higher, see DAPO (https://arxiv.org/abs/2503.14476)"""
    truncated_importance_sampling_ratio_cap: float = 0.0
    """The maximum cap for truncated importance sampling ratio (0 means disabled)"""
    kl_estimator: Literal[0, 1, 2, 3] = 2
    """the KL estimator to use"""
    loss_denominator: str = "token"
    """Optional constant denominator for masked_mean; can be "token" or a float value.
    when "token", the loss is divided by the total number of tokens in the batch (standard LM training).
    when a float value, the loss is divided by this value (ideally, max tokens in batch, per Dr GRPO).
    """
    alpha: float = 0.6
    """The alpha value for doing polyak updates (ref_param = alpha * param + (1 - alpha) * ref_param)
    reference: [TR-DPO](https://huggingface.co/papers/2404.09656), but it's actually pretty commonly
    used. E.g., [TD3](https://arxiv.org/abs/1802.09477) uses https://github.com/vwxyzjn/cleanrl/blob/dcc289fc6f0bda492fa7360a155262cf826b12a5/cleanrl/td3_continuous_action.py#L269
    """
    ref_policy_update_freq: int | None = None
    """How many training steps to take before updating the reference policy."""
    load_ref_policy: bool = True
    """Whether to load and use a reference policy for KL penalty calculation."""
    loss_fn: Literal["dapo", "cispo"] = "dapo"
    """Whether to use DAPO or CISPO loss function."""
    record_entropy: bool = False
    """whether to record the entropy of the policy during training. Uses extra memory."""
    use_vllm_logprobs: bool = False
    """whether to use vLLM's logprobs for training instead of calculating them via forward pass"""
    temperature: float = field(default=1.0, init=False)
    """RUNTIME VALUE: Temperature for sampling, set from streaming_config."""

    # Ray
    single_gpu_mode: bool = False
    """whether to collocate vLLM and actor on the same node (mostly for debugging purposes)"""
    num_learners_per_node: list[int] = field(default_factory=lambda: [1])
    """number of GPU deepspeed learners per node (e.g., --num_learners_per_node 2 4 means 2 learner processes
    on the first node and 4 learner processes on the second node; each process will have 1 GPU)"""
    num_nodes: int = 1
    """Number of nodes for distributed training."""
    sequence_parallel_size: int = 1
    """sequence parallel size - how many GPUs we will parallelize sequences across during training.
    Useful for super-long context lengths."""
    deepspeed_stage: int = 0
    """the deepspeed stage"""
    deepspeed_zpg: int = 8
    """the deepspeed zpg value. Higher values are more memory efficient but slower. Set to 1 to disable zpg, which uses less memory but is significantly slower. Ideally is set to the number of GPUs per node (usually 8, default)."""
    deepspeed_offload_param: bool = False
    """whether to offload parameters to CPU (reduces GPU memory usage)"""
    deepspeed_offload_optimizer: bool = False
    """whether to offload optimizer states to CPU (reduces GPU memory usage)"""
    gather_whole_model: bool = True
    """whether to gather the whole model to boardcast (not doable for 70B but can be faster for 8B)"""
    enable_queue_dashboard: bool = True
    """whether to enable the ActorManager queue monitoring dashboard"""
    queue_dashboard_port: int | None = None
    """optional port for the dashboard server (if None, finds a free port automatically)"""

    # Experiment tracking
    verbose: bool = False
    """If toggled, debug output will be shown"""
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "open_instruct_internal"
    """The wandb's project name"""
    wandb_entity: str | None = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = True
    """Whether to upload the saved model to huggingface"""
    hf_entity: str | None = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: str | None = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: str | None = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: str | None = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: str = "output"
    """Where to save the model"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""
    keep_last_n_checkpoints: int = 3
    """How many checkpoints to keep in the output directory. -1 for all."""
    checkpoint_state_freq: int = -1
    """How often to save the model checkpoint, optimizer states, and lr scheduler states (in steps)"""
    checkpoint_state_dir: str | None = None
    """Where to save the model checkpoint (if applicable)"""
    gs_checkpoint_state_dir: str | None = None
    """The actual `checkpoint_state_dir` to use (handling the case where gs_bucket_path is provided)"""

    # Ai2 specific settings
    try_launch_beaker_eval_jobs_on_weka: bool = False
    """Whether to launch beaker evaluation jobs after training on weka"""
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: str | None = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: list[str] | None = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """the max generation length for evaluation for oe-eval"""
    oe_eval_beaker_image: str | None = None
    """the docker image for evaluation for oe-eval"""
    oe_eval_gpu_multiplier: int | None = None
    """multiply the gpus used for each oe-eval task"""
    eval_priority: Literal["low", "normal", "high", "urgent"] = "normal"
    """the priority of auto-launched evaluation jobs"""
    eval_workspace: str = "ai2/tulu-3-results"
    """the workspace to launch evaluation jobs on"""
    send_slack_alerts: bool = False
    """Whether to send Slack alerts on training failures"""

    # Evaluation behavior
    eval_on_step_0: bool = False
    """Whether to run local evaluation at training step 0. Defaults to False."""

    def __post_init__(self):
        if self.use_vllm_logprobs and self.truncated_importance_sampling_ratio_cap > 0.0:
            raise ValueError(
                "Cannot use both `use_vllm_logprobs` and `truncated_importance_sampling_ratio_cap`. "
                "use_vllm_logprobs sets old_logprobs to vLLM logprobs, making importance sampling pointless."
            )
        if self.loss_denominator != "token" and float(self.loss_denominator) <= 0:
            raise ValueError(
                f"loss_denominator must be a valid float greater than 0 if not 'token', got: {self.loss_denominator}"
            )
        if self.checkpoint_state_freq > 0 and self.checkpoint_state_dir is None:
            raise ValueError("`checkpoint_state_dir` must be provided if `checkpoint_state_freq` is greater than 0!")
        if self.checkpoint_state_dir is not None and self.checkpoint_state_freq == -1:
            raise ValueError("`checkpoint_state_freq` must be greater than 0 if `checkpoint_state_dir` is provided!")

        if self.gs_checkpoint_state_dir is not None and not self.gs_checkpoint_state_dir.startswith("gs://"):
            raise ValueError(f"`gs_checkpoint_state_dir` must start with 'gs://', got: {self.gs_checkpoint_state_dir}")
        if self.gs_bucket_path is not None and not self.gs_bucket_path.startswith("gs://"):
            raise ValueError(f"`gs_bucket_path` must start with 'gs://', got: {self.gs_bucket_path}")
        if self.sequence_parallel_size > 1 and self.deepspeed_stage != 3:
            raise ValueError("`sequence_parallel_size` > 1 requires `deepspeed_stage` to be 3!")

        if self.gs_bucket_path is not None and self.gs_checkpoint_state_dir is None:
            if self.checkpoint_state_dir is None:
                raise ValueError("`checkpoint_state_dir` must be provided when using `gs_bucket_path`!")
            checkpoint_dir_name = self.checkpoint_state_dir.rstrip("/")
            beaker_users = get_beaker_whoami()
            if beaker_users is not None:
                self.gs_checkpoint_state_dir = f"{self.gs_bucket_path}/{beaker_users}/{checkpoint_dir_name}"
            else:
                self.gs_checkpoint_state_dir = f"{self.gs_bucket_path}/{checkpoint_dir_name}"
            if not checkpoint_dir_name.startswith("/filestore"):
                self.checkpoint_state_dir = f"/filestore{self.checkpoint_state_dir}"

        if self.checkpoint_state_dir is not None:
            if self.gs_checkpoint_state_dir is not None:
                download_latest_checkpoint_from_gs(self.gs_checkpoint_state_dir, self.checkpoint_state_dir)
            calibrate_checkpoint_state_dir(self.checkpoint_state_dir)
        if not self.load_ref_policy and self.beta != 0.0:
            raise ValueError(
                "When load_ref_policy=False, beta must be 0.0. "
                f"Got beta={self.beta}. Set --beta 0.0 or --load_ref_policy to use KL penalty."
            )
