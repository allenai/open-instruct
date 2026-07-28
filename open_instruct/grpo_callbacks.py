"""
GRPO-specific callbacks for OLMo-core Trainer.

These callbacks handle:
- vLLM weight synchronization after each training step
- Reference policy Polyak updates
"""

import contextlib
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import ray
import ray.exceptions
import ray.util.queue as ray_queue
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from olmo_core.distributed import utils as dist_utils
from olmo_core.train.callbacks import Callback
from olmo_core.train.train_module import TransformerTrainModule
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_utils, logger_utils, olmo_core_utils, utils, vllm_utils

logger = logger_utils.setup_logger(__name__)

_BLOCK_PATTERN = re.compile(r"blocks\.(\d+)\.(.*)")
_OLMO_CORE_TO_HF_LAYER_MAPPINGS = {
    "attention.w_q.weight": "self_attn.q_proj.weight",
    "attention.w_k.weight": "self_attn.k_proj.weight",
    "attention.w_v.weight": "self_attn.v_proj.weight",
    "attention.w_out.weight": "self_attn.o_proj.weight",
    # Qwen2/DeepSeek-R1-Distill-Qwen only (see olmo_core_utils.QWEN2_STYLE_HF_MODEL_TYPES):
    # Qwen3/LLaMA presets keep AttentionConfig.bias=False, so these never appear for them.
    "attention.w_q.bias": "self_attn.q_proj.bias",
    "attention.w_k.bias": "self_attn.k_proj.bias",
    "attention.w_v.bias": "self_attn.v_proj.bias",
    "attention.q_norm.weight": "self_attn.q_norm.weight",
    "attention.k_norm.weight": "self_attn.k_norm.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "feed_forward_norm.weight": "post_attention_layernorm.weight",
}

# Qwen2's o_proj has no bias; olmo-core's AttentionConfig.bias applies uniformly to all four
# attention projections, so the qwen2 preset's w_out.bias is a synthetic, permanently-zero,
# frozen param with no counterpart in vLLM's HF-format model (see
# olmo_core_utils.QWEN2_STYLE_HF_MODEL_TYPES / drop_frozen_zero_bias_for_hf_export). vLLM's
# weight loader errors on any name it doesn't recognize, so this must be dropped, not sent
# under its raw olmo-core name.
_OLMO_CORE_PARAMS_WITH_NO_HF_COUNTERPART = re.compile(r"blocks\.\d+\.attention\.w_out\.bias$")


def olmo_core_to_hf_name(name: str) -> str | None:
    """Convert OLMo-core parameter name to HuggingFace format for Qwen3/LLaMA/Qwen2 models.

    Returns None for params with no destination in vLLM's model (see
    _OLMO_CORE_PARAMS_WITH_NO_HF_COUNTERPART) -- callers must drop these rather than send them.
    """
    if _OLMO_CORE_PARAMS_WITH_NO_HF_COUNTERPART.match(name):
        return None
    if name == "embeddings.weight":
        return "model.embed_tokens.weight"
    if name == "lm_head.norm.weight":
        return "model.norm.weight"
    if name == "lm_head.w_out.weight":
        return "lm_head.weight"

    layer_match = _BLOCK_PATTERN.match(name)
    if layer_match:
        layer_idx = layer_match.group(1)
        rest = layer_match.group(2)
        if rest in _OLMO_CORE_TO_HF_LAYER_MAPPINGS:
            return f"model.layers.{layer_idx}.{_OLMO_CORE_TO_HF_LAYER_MAPPINGS[rest]}"

    return name


@dataclass
class StepTimingCallback(Callback):
    """Records outer-loop timing and utilization metrics per step.

    Priority is set very low so its post_step runs after every other callback
    (e.g. VLLMWeightSyncCallback), making time/total an end-to-end step duration.
    """

    priority: ClassVar[int] = -1000

    model_dims: utils.ModelDims
    vllm_num_engines: int = 1
    vllm_tensor_parallel_size: int = 1
    samples_per_prompt: int = 1
    num_training_gpus: int = 1

    _step_start: float = field(default=0.0, init=False, repr=False)
    _last_step_end: float = field(default=0.0, init=False, repr=False)
    _train_duration: float = field(default=0.0, init=False, repr=False)
    _training_start: float = field(default=0.0, init=False, repr=False)
    _num_total_tokens: int = field(default=0, init=False, repr=False)
    _prompt_lengths: list[int] = field(default_factory=list, init=False, repr=False)
    _response_lengths: list[int] = field(default_factory=list, init=False, repr=False)
    _prompt_sample_counts: list[int] = field(default_factory=list, init=False, repr=False)
    _prompt_attempt_counts: list[int] = field(default_factory=list, init=False, repr=False)
    _total_generation_time: float = field(default=0.0, init=False, repr=False)

    def pre_train(self) -> None:
        self._training_start = time.perf_counter()
        self._last_step_end = self._training_start

    def pre_step(self, batch: dict[str, Any]) -> None:
        self._step_start = time.perf_counter()
        metrics = batch["metrics"]
        self._prompt_lengths = list(metrics["batch/prompt_lengths"])
        self._response_lengths = list(metrics["batch/response_lengths"])
        self._prompt_sample_counts = list(metrics["batch/prompt_sample_counts"])
        self._prompt_attempt_counts = list(metrics["batch/prompt_attempt_counts"])
        self._total_generation_time = float(metrics["time/group_generation_max"])

    def post_train_batch(self) -> None:
        self._train_duration = time.perf_counter() - self._step_start

    def post_step(self) -> None:
        now = time.perf_counter()
        step_time = now - self._last_step_end
        total_training_time = now - self._training_start
        self._last_step_end = now

        train_module = cast(Any, self.trainer.train_module)
        num_step_tokens = int(train_module._last_num_step_tokens)
        self._num_total_tokens += num_step_tokens

        self.trainer.record_metric("time/total", step_time, reduce_type=None)
        self.trainer.record_metric("time/training", self._train_duration, reduce_type=None)
        self.trainer.record_metric("learner_tokens_per_second_step", num_step_tokens / step_time, reduce_type=None)
        self.trainer.record_metric(
            "learner_tokens_per_second_overall", self._num_total_tokens / total_training_time, reduce_type=None
        )

        utilization = utils.calculate_utilization_metrics(
            model_dims=self.model_dims,
            prompt_lengths=self._prompt_lengths,
            response_lengths=self._response_lengths,
            total_generation_time=self._total_generation_time,
            samples_per_prompt=self.samples_per_prompt,
            num_engines=self.vllm_num_engines,
            num_gpus_per_engine=self.vllm_tensor_parallel_size,
            training_time=self._train_duration,
            num_training_gpus=self.num_training_gpus,
            prompt_sample_counts=self._prompt_sample_counts,
            prompt_attempt_counts=self._prompt_attempt_counts,
        )
        for key, value in utilization.items():
            self.trainer.record_metric(key, float(value), reduce_type=None)


@dataclass
class VLLMWeightSyncCallback(Callback):
    """Callback to synchronize weights from training model to vLLM inference engines.

    After each training step, this callback:
    1. Pauses vLLM actors via actor_manager
    2. Gathers FSDP-sharded parameters using summon_full_params
    3. Broadcasts weights from rank 0 to vLLM engines
    4. Resumes vLLM actors
    """

    vllm_engines: list[ray.actor.ActorHandle]
    actor_manager: ray.actor.ActorHandle
    model_update_group: Any | None = None
    sync_interval: int = 1
    name_mapper: Callable[[str], str | None] | None = None

    @property
    def train_module(self) -> TransformerTrainModule:
        return cast(TransformerTrainModule, self.trainer.train_module)

    def post_step(self) -> None:
        if self.trainer.global_step % self.sync_interval != 0:
            return

        torch.cuda.empty_cache()

        broadcast_refs = vllm_utils.broadcast_weights_to_vllm(
            model=self.train_module.model,
            vllm_engines=self.vllm_engines,
            model_update_group=self.model_update_group,
            model_step=self.trainer.global_step,
            name_mapper=self.name_mapper,
        )
        sync_time_stats, _ = grpo_utils.perform_weight_sync(
            broadcast_refs, self.vllm_engines, self.actor_manager, inflight_updates=True
        )
        for name, value in sync_time_stats.items():
            self.trainer.record_metric(name, value, reduce_type=None)


@dataclass
class RefPolicyUpdateCallback(Callback):
    """Callback to update reference policy using Polyak averaging.

    Updates reference policy parameters as:
        ref_param = (1 - alpha) * ref_param + alpha * policy_param

    This is used for KL divergence computation in GRPO.
    """

    ref_policy: nn.Module
    alpha: float = 0.6
    update_interval: int = 1
    _fsdp2_submodules: list[FSDPModule] | None = field(default=None, init=False, repr=False)

    @property
    def train_module(self) -> TransformerTrainModule:
        return cast(TransformerTrainModule, self.trainer.train_module)

    def _get_fsdp2_submodules(self, model: nn.Module) -> list[FSDPModule]:
        if self._fsdp2_submodules is None:
            self._fsdp2_submodules = [m for _, m in vllm_utils._get_fsdp2_submodules(model)]
        return self._fsdp2_submodules

    def post_step(self) -> None:
        if self.trainer.global_step % self.update_interval != 0:
            return

        model = self.train_module.model

        if isinstance(model, FSDP):
            ctx = FSDP.summon_full_params(model, writeback=False, rank0_only=False)
        else:
            ctx = contextlib.nullcontext()

        fsdp2_submodules: list[FSDPModule] = []
        if isinstance(model, FSDPModule):
            fsdp2_submodules = self._get_fsdp2_submodules(model)
            for m in fsdp2_submodules:
                m.unshard()

        try:
            with ctx:
                for ref_param, param in zip(self.ref_policy.parameters(), model.parameters(), strict=True):
                    ref_param.data.mul_(1.0 - self.alpha).add_(param.data, alpha=self.alpha)
        finally:
            for m in fsdp2_submodules:
                m.reshard()


@dataclass
class HFCheckpointCallback(Callback):
    """Periodically saves a HuggingFace-format checkpoint during training.

    Complements olmo-core's native `CheckpointerCallback` (full, resumable training
    state: model + optimizer + LR scheduler). This instead saves fully materialized
    HF-format weights, matching the format `PolicyTrainerOLMoCoreProcess.save_model`
    writes at the end of training, so intermediate checkpoints can be evaluated or
    served without waiting for the run to finish.
    """

    output_dir: str
    model_name_or_path: str
    tokenizer: transformers.PreTrainedTokenizer
    save_freq: int

    @property
    def train_module(self) -> TransformerTrainModule:
        return cast(TransformerTrainModule, self.trainer.train_module)

    def post_step(self) -> None:
        if self.trainer.global_step % self.save_freq != 0:
            return

        # state_dict()/full_tensor() are collective FSDP operations; every rank must call them.
        state_dict = self.train_module.model.state_dict()
        state_dict = {
            k: v.full_tensor().cpu() if hasattr(v, "full_tensor") else v.cpu() for k, v in state_dict.items()
        }

        if dist_utils.get_rank() != 0:
            return

        step_dir = os.path.join(self.output_dir, f"step_{self.trainer.global_step}")
        olmo_core_utils.save_state_dict_as_hf(state_dict, step_dir, self.model_name_or_path, self.tokenizer)
        logger.info(f"Saved HF checkpoint at step {self.trainer.global_step} to {step_dir}")


@dataclass
class SyncGenerationGateCallback(Callback):
    """Notifies the DataPreparationActor once a training step has fully finished.

    Only meaningful when `async_steps == 0` (fully synchronous training): the actor's
    generation-ahead gate normally advances as soon as a batch is *pulled* off the queue
    (`DataPreparationActor.get_data`), which happens before that step's forward/backward/optim
    and weight sync run. With `async_steps == 0` that would let generation of step N+1 start
    while step N is still training. `post_step` runs after weight sync
    (`VLLMWeightSyncCallback`, default priority), so by the time this fires vLLM already has
    the updated weights.
    """

    priority: ClassVar[int] = -2000

    def post_step(self) -> None:
        if dist_utils.get_rank() != 0:
            return

        try:
            data_prep_actor = ray.get_actor(data_loader_lib.DATA_PREP_ACTOR_NAME)
            data_prep_actor.mark_trained.remote(self.trainer.global_step - 1)
        except (ray.exceptions.RayError, ValueError) as e:
            logger.warning(f"Failed to notify DataPreparationActor of trained step: {e}")


@dataclass
class DataPreparationActorCheckpointCallback(Callback):
    """Callback to save and restore DataPreparationActor state during checkpointing."""

    def state_dict(self) -> dict[str, Any]:
        """Save DataPreparationActor state in the global rank-0 trainer state."""
        if dist_utils.get_rank() != 0:
            return {}

        try:
            data_prep_actor = ray.get_actor(data_loader_lib.DATA_PREP_ACTOR_NAME)
            return {"data_prep_state": ray.get(data_prep_actor.get_state.remote())}
        except (ray.exceptions.RayError, ValueError) as e:
            logger.warning(f"Failed to get DataPreparationActor state: {e}")
            return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore DataPreparationActor state once, from global rank 0."""
        if dist_utils.get_rank() != 0 or "data_prep_state" not in state_dict:
            return

        try:
            data_prep_actor = ray.get_actor(data_loader_lib.DATA_PREP_ACTOR_NAME)
            ray.get(data_prep_actor.set_state.remote(state_dict["data_prep_state"]))
            logger.info("Restored DataPreparationActor state from checkpoint")
        except (ray.exceptions.RayError, ValueError) as e:
            logger.warning(f"Failed to restore DataPreparationActor state: {e}")


@dataclass
class EvalCallback(Callback):
    """Pushes eval prompts onto prompt_Q on cadence and drains eval results.

    Mirrors grpo_fast.py's main-loop eval coordination as an OLMo-core Callback,
    since grpo.py delegates the train loop to OLMo-core's Trainer. Only register
    on rank 0 when eval is enabled (eval_dataset is not None and local_eval_every > 0).
    """

    args: grpo_utils.GRPOExperimentConfig
    prompt_Q: ray_queue.Queue
    evaluation_inference_results_Q: ray_queue.Queue
    eval_dataset: Dataset
    eval_data_loader: data_loader_lib.HFDataLoader
    eval_generation_config: Any
    model_dims: utils.ModelDims
    base_env_config: Any
    tokenizer: Any
    max_possible_score: float
    actor_manager: ray.actor.ActorHandle | None = None

    _last_eval_collected: bool = field(default=True, init=False, repr=False)
    _eval_pending: bool = field(default=False, init=False, repr=False)
    _eval_step: int | None = field(default=None, init=False, repr=False)

    def pre_step(self, batch: dict[str, Any]) -> None:
        if not (
            (self.args.eval_on_step_0 and self.trainer.global_step == 1)
            or (self.trainer.global_step % self.args.local_eval_every == 0 and self.trainer.global_step > 1)
        ):
            return
        if not self._last_eval_collected:
            logger.warning(
                "[EvalCallback] previous eval round not fully collected; results may interleave. "
                "Consider increasing local_eval_every."
            )
        for eval_example in iter(self.eval_data_loader):
            data_loader_lib.add_prompt_to_generator(
                eval_example,
                self.trainer.global_step,
                self.prompt_Q,
                self.eval_generation_config,
                is_eval=True,
                base_env_config=self.base_env_config,
            )
        self.eval_data_loader.reset()
        self._eval_pending = True
        self._eval_step = self.trainer.global_step

    def post_step(self) -> None:
        assert self.args.num_training_steps is not None
        is_final_step = self.trainer.global_step >= self.args.num_training_steps
        if not self._eval_pending and not is_final_step:
            return

        eval_collected = grpo_utils.maybe_evaluate(
            args=self.args,
            training_step=self.trainer.global_step,
            evaluation_inference_results_Q=self.evaluation_inference_results_Q,
            tokenizer=self.tokenizer,
            episode=0,
            eval_dataset=self.eval_dataset,
            eval_generation_config=self.eval_generation_config,
            model_dims=self.model_dims,
            base_env_config=self.base_env_config,
            max_possible_score=self.max_possible_score,
            actor_manager=self.actor_manager,
            eval_step=self._eval_step,
        )
        self._last_eval_collected = eval_collected
        if eval_collected:
            self._eval_pending = False
