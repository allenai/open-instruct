import os
from dataclasses import dataclass, field
from typing import Literal

from open_instruct import utils
from open_instruct.utils import calibrate_checkpoint_state_dir, download_latest_checkpoint_from_gs, get_beaker_whoami


@dataclass
class ExperimentConfig:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    run_name: str | None = None

    learning_rate: float = 2e-5
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "linear"
    warm_up_steps: int = 0
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    set_weight_decay_on_bias_and_norm: bool = True
    fused_optimizer: bool = False

    per_device_train_batch_size: int = 1
    total_episodes: int = 100000
    world_size: int | None = None
    num_training_steps: int | None = None
    local_eval_every: int = 100
    save_freq: int = 200
    backend_timeout: int = 120

    num_epochs: int = 1
    num_mini_batches: int = 1
    beta: float = 0.05
    clip_lower: float = 0.2
    clip_higher: float = 0.2
    truncated_importance_sampling_ratio_cap: float = 0.0
    kl_estimator: Literal[0, 1, 2, 3] = 2
    loss_denominator: str = "token"
    alpha: float = 0.6
    ref_policy_update_freq: int | None = None
    load_ref_policy: bool = True
    loss_fn: Literal["dapo", "cispo"] = "dapo"
    record_entropy: bool = False
    use_vllm_logprobs: bool = False
    temperature: float = field(default=1.0, init=False)

    single_gpu_mode: bool = False
    num_learners_per_node: list[int] = field(default_factory=lambda: [1])
    num_nodes: int = 1
    sequence_parallel_size: int = 1
    deepspeed_stage: int = 0
    deepspeed_zpg: int = 8
    deepspeed_offload_param: bool = False
    deepspeed_offload_optimizer: bool = False
    gather_whole_model: bool = True
    enable_queue_dashboard: bool = True
    queue_dashboard_port: int | None = None

    verbose: bool = False
    with_tracking: bool = False
    wandb_project_name: str = "open_instruct_internal"
    wandb_entity: str | None = None
    push_to_hub: bool = True
    hf_entity: str | None = None
    hf_repo_id: str | None = None
    hf_repo_revision: str | None = None
    hf_repo_url: str | None = None
    output_dir: str = "output"
    save_traces: bool = False
    cache_dataset_only: bool = False
    keep_last_n_checkpoints: int = 3
    checkpoint_state_freq: int = -1
    checkpoint_state_dir: str | None = None
    gs_checkpoint_state_dir: str | None = None

    try_launch_beaker_eval_jobs_on_weka: bool = False
    try_auto_save_to_beaker: bool = True
    gs_bucket_path: str | None = None
    oe_eval_tasks: list[str] | None = None
    oe_eval_max_length: int = 4096
    oe_eval_beaker_image: str | None = None
    oe_eval_gpu_multiplier: int | None = None
    eval_priority: Literal["low", "normal", "high", "urgent"] = "normal"
    eval_workspace: str = "ai2/tulu-3-results"
    send_slack_alerts: bool = False

    eval_on_step_0: bool = False

    def __post_init__(self):
        if self.use_vllm_logprobs and self.truncated_importance_sampling_ratio_cap > 0.0:
            raise ValueError(
                "Cannot use both `use_vllm_logprobs` and `truncated_importance_sampling_ratio_cap`. "
                "use_vllm_logprobs sets old_logprobs to vLLM logprobs, making importance sampling pointless."
            )
        self.loss_denominator = utils.get_denominator(self.loss_denominator)
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
