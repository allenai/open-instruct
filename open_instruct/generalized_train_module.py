"""
Generalized TransformerTrainModule with pluggable loss computation.

This module extracts the FSDP/microbatch infrastructure from OLMo-core's TransformerTrainModule
and makes the loss computation pluggable via a `compute_loss()` method that subclasses can override.

The default compute_loss() does CE + Z-loss (same as TransformerTrainModule).
Subclasses can override for different losses (DPO, GRPO, SFT, etc.).
"""

import contextlib
import logging
from abc import ABC
from collections.abc import Generator
from dataclasses import replace
from functools import cached_property
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.checkpoint import merge_state_dicts, prune_state_dict, swap_param_keys
from olmo_core.distributed.parallel import DataParallelType, build_world_mesh, get_dp_process_group
from olmo_core.distributed.utils import get_local_tensor, get_reduce_divide_factor, get_world_size, is_distributed
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.transformer import Transformer
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module import EvalBatchSpec, TrainModule
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.config import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerTensorParallelConfig,
)
from olmo_core.utils import gc_cuda, get_default_device, log_once, move_to_device
from torch.distributed import DeviceMesh
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

log = logging.getLogger(__name__)


class GeneralizedTransformerTrainModule(TrainModule, ABC):
    """
    A generalized TrainModule with FSDP/microbatch infrastructure and pluggable loss computation.

    This class extracts the common training infrastructure from TransformerTrainModule:
    - FSDP/DDP parallelization
    - Microbatch splitting and gradient accumulation
    - Learning rate scheduling
    - Gradient clipping
    - Distributed checkpointing

    The loss computation is factored into a `compute_loss()` method that subclasses can override.
    The default implementation does CE + Z-loss (same as TransformerTrainModule).
    """

    def __init__(
        self,
        model: Transformer,
        optim: OptimConfig,
        rank_microbatch_size: int,
        max_sequence_length: int,
        compile_model: bool = False,
        float8_config: Float8Config | None = None,
        dp_config: TransformerDataParallelConfig | None = None,
        tp_config: TransformerTensorParallelConfig | None = None,
        cp_config: TransformerContextParallelConfig | None = None,
        ep_config: TransformerExpertParallelConfig | None = None,
        ac_config: TransformerActivationCheckpointingConfig | None = None,
        z_loss_multiplier: float | None = None,
        autocast_precision: torch.dtype | None = None,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
        load_key_mapping: dict[str, str] | None = None,
        label_ignore_index: int = -100,
    ):
        super().__init__()

        if rank_microbatch_size % max_sequence_length != 0:
            raise OLMoConfigurationError(
                f"'rank_microbatch_size' ({rank_microbatch_size:,d} tokens) must be divisible by "
                f"'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )

        self.device = device or get_default_device()
        self.world_mesh: DeviceMesh | None = None
        if is_distributed():
            self.world_mesh = build_world_mesh(
                dp=dp_config, tp=tp_config, cp=cp_config, ep=ep_config, device_type=self.device.type
            )
            log.info(f"Data parallel world size = {get_world_size(self.dp_process_group):,d}")
        elif dp_config is not None or tp_config is not None or ep_config is not None or cp_config is not None:
            raise OLMoConfigurationError("Training parallelism configs are only valid for distributed training")

        if (
            ac_config is not None
            and ac_config.mode == TransformerActivationCheckpointingMode.budget
            and not compile_model
        ):
            raise OLMoConfigurationError(
                "Activation checkpointing with 'budget' mode requires compilation to be enabled"
            )

        self.model = parallelize_model(
            model,
            world_mesh=self.world_mesh,
            device=self.device,
            max_sequence_length=max_sequence_length,
            rank_microbatch_size=rank_microbatch_size,
            compile_model=compile_model,
            float8_config=float8_config,
            dp_config=dp_config,
            tp_config=tp_config,
            cp_config=cp_config,
            ep_config=ep_config,
            ac_config=ac_config,
        )
        self._model_mode: Literal["train", "eval"] | None = None

        self._dp_config = dp_config
        self._cp_config = cp_config
        self._tp_config = tp_config
        self._ep_config = ep_config
        self.label_ignore_index = label_ignore_index
        self.z_loss_multiplier = z_loss_multiplier
        self.rank_microbatch_size = rank_microbatch_size
        self.max_sequence_length = max_sequence_length
        self.autocast_precision = autocast_precision
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, strict=True
        )
        self.load_key_mapping = load_key_mapping

        log.info("Building optimizer...")
        self.optim: Optimizer = optim.build(self.model, strict=True)

    @property
    def dp_process_group(self) -> dist.ProcessGroup | None:
        return None if self.world_mesh is None else get_dp_process_group(self.world_mesh)

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(self.rank_microbatch_size, max_sequence_length=self.max_sequence_length)

    @property
    def dp_config(self) -> TransformerDataParallelConfig | None:
        return self._dp_config

    @property
    def tp_enabled(self) -> bool:
        return self._tp_config is not None

    @property
    def cp_enabled(self) -> bool:
        return self._cp_config is not None

    @property
    def ep_enabled(self) -> bool:
        return self._ep_config is not None

    @cached_property
    def world_size(self) -> int:
        return get_world_size()

    @cached_property
    def _reduce_divide_factor(self) -> float:
        return get_reduce_divide_factor(self.world_size)

    def pre_train(self):
        dp_ws = get_world_size(self.trainer.dp_process_group)
        if self.trainer.global_batch_size % (self.rank_microbatch_size * dp_ws) != 0:
            raise OLMoConfigurationError(
                f"global batch size ({self.trainer.global_batch_size:,d}) must be divisible by "
                f"micro-batch size ({self.rank_microbatch_size:,d}) x DP world size ({dp_ws})"
            )

    def state_dict(self, *, optim: bool | None = None) -> dict[str, Any]:
        if optim is None:
            optim = True
        return self._get_state_dict(self.state_dict_save_opts, optim=optim)

    def state_dict_to_load(self, metadata: Metadata, *, optim: bool | None = None) -> dict[str, Any]:
        has_optim_state: bool = False
        for key in metadata.state_dict_metadata:
            if key.startswith("optim."):
                has_optim_state = True
                break

        if optim is None:
            if not has_optim_state:
                log.warning("No optimizer state found in checkpoint")
                optim = False
            else:
                optim = True

        load_opts = self.state_dict_load_opts
        if optim:
            if not has_optim_state:
                raise RuntimeError("Checkpoint does not contain optimizer state, but 'optim=True' was requested")

            if "optim.param_groups.0.params" in metadata.state_dict_metadata:
                if load_opts.flatten_optimizer_state_dict:
                    log.warning(
                        "Loading checkpoint with an unflattened optimizer state even though "
                        "'flatten_optimizer_state_dict=True' in train module's 'state_dict_load_opts', "
                        "automatically switching to 'flatten_optimizer_state_dict=False'."
                    )
                    load_opts = replace(load_opts, flatten_optimizer_state_dict=False)
            else:
                if not load_opts.flatten_optimizer_state_dict:
                    log.warning(
                        "Loading checkpoint with a flattened optimizer state even though "
                        "'flatten_optimizer_state_dict=False' in train module's 'state_dict_load_opts', "
                        "automatically switching to 'flatten_optimizer_state_dict=True'."
                    )
                    load_opts = replace(load_opts, flatten_optimizer_state_dict=True)

        state_dict = self._get_state_dict(load_opts, optim=optim)
        if self.load_key_mapping is not None:
            swap_param_keys(state_dict, self.load_key_mapping, metadata=metadata)

        if not load_opts.strict:
            pruned_keys = prune_state_dict(state_dict, set(metadata.state_dict_metadata.keys()))
            if pruned_keys:
                log.warning(f"Checkpoint is missing the following keys: {pruned_keys}")

        return state_dict

    def state_dict_to_save(self, *, optim: bool | None = None) -> dict[str, Any]:
        if optim is None:
            optim = True
        return self._get_state_dict(self.state_dict_save_opts, optim=optim)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        load_optim = "optim" in state_dict

        if self.load_key_mapping is not None:
            swap_param_keys(state_dict, self.load_key_mapping, reverse=True, quiet=True)

        if not self.state_dict_load_opts.strict:
            flatten_optimizer_state_dict = False if not load_optim else ("state" not in state_dict["optim"])
            load_opts = replace(self.state_dict_load_opts, flatten_optimizer_state_dict=flatten_optimizer_state_dict)
            full_state_dict = self._get_state_dict(load_opts, optim=load_optim)
            merge_state_dicts(state_dict, full_state_dict)

        dist_cp_sd.set_model_state_dict(self.model, state_dict["model"], options=self.state_dict_load_opts)
        gc_cuda()
        if load_optim:
            dist_cp_sd.set_optimizer_state_dict(
                self.model, self.optim, state_dict["optim"], options=self.state_dict_load_opts
            )
            gc_cuda()

    def compute_loss(
        self, micro_batch: dict[str, Any], batch_num_tokens_for_loss: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Compute the loss for a micro-batch.

        Default implementation does CE + Z-loss (same as TransformerTrainModule).
        Subclasses can override for different loss functions (DPO, GRPO, etc.).

        Args:
            micro_batch: The micro-batch data.
            batch_num_tokens_for_loss: Number of tokens used in loss computation.

        Returns:
            Tuple of (loss, ce_loss, z_loss). z_loss may be None.
        """
        input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)
        _, loss, ce_loss, z_loss = self.model_forward(
            input_ids,
            labels=labels,
            ignore_index=self.label_ignore_index,
            loss_reduction="sum",
            z_loss_multiplier=self.z_loss_multiplier,
            loss_div_factor=batch_num_tokens_for_loss,
            return_logits=False,
            **model_kwargs,
        )
        return loss, ce_loss, z_loss

    def train_batch(self, batch: dict[str, Any], dry_run: bool = False):
        self._set_model_mode("train")

        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        batch_num_tokens = batch["labels"].numel()
        batch_num_tokens_per_instance = batch["labels"].shape[1]
        batch_num_tokens_for_loss = move_to_device((batch["labels"] != self.label_ignore_index).sum(), self.device)

        self.record_metric(
            "train/masked labels (%)",
            (batch_num_tokens - batch_num_tokens_for_loss) / batch_num_tokens,
            ReduceType.mean,
        )

        if (instance_mask := batch.get("instance_mask")) is not None:
            self.record_metric("train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean)
            batch_num_tokens_for_loss += (~instance_mask).sum() * batch_num_tokens_per_instance

        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: torch.Tensor | None = None
        if self.z_loss_multiplier is not None:
            z_batch_loss = move_to_device(torch.tensor(0.0), self.device)

        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                loss, ce_loss, z_loss = self.compute_loss(micro_batch, batch_num_tokens_for_loss)

                ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss

                loss.backward()

        del batch

        self.model.post_batch(dry_run=dry_run)

        if dry_run:
            self.model.reset_auxiliary_metrics()
            return

        if isinstance(self.optim, SkipStepOptimizer):
            if is_distributed():
                ce_batch_loss.div_(self._reduce_divide_factor)
                dist.all_reduce(ce_batch_loss)
                ce_batch_loss.div_(self.world_size)
                ce_batch_loss.mul_(self._reduce_divide_factor)
            self.record_ce_loss(ce_batch_loss)
            self.optim.latest_loss = ce_batch_loss
        else:
            self.record_ce_loss(ce_batch_loss, ReduceType.mean)

        if z_batch_loss is not None:
            assert self.z_loss_multiplier is not None
            self.record_metric("Z loss", z_batch_loss, ReduceType.mean, namespace="train")
            self.record_metric(
                "Z loss unscaled", z_batch_loss / self.z_loss_multiplier, ReduceType.mean, namespace="train"
            )

        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(reset=True).items():
            self.record_metric(metric_name, metric_val, reduction, namespace="train")

    def eval_batch(self, batch: dict[str, Any], labels: torch.Tensor | None = None) -> torch.Tensor | LMOutputWithLoss:
        if self.cp_enabled:
            raise RuntimeError(
                f"{self.__class__.__name__}.eval_batch() does not support context parallelism yet, "
                "please disable in-loop evals"
            )
        if self.tp_enabled:
            raise RuntimeError(
                f"{self.__class__.__name__}.eval_batch() does not support tensor parallelism yet, "
                "please disable in-loop evals"
            )

        input_ids, labels, model_kwargs = self._prepare_batch(batch, labels)

        self._set_model_mode("eval")

        with self._eval_batch_context():
            return self.model_forward(
                input_ids, labels=labels, ignore_index=self.label_ignore_index, loss_reduction="none", **model_kwargs
            )

    def optim_step(self):
        if self.max_grad_norm is not None:
            grad_norm = self._clip_grad_norm(self.max_grad_norm)
            self.trainer.record_metric("total grad norm", grad_norm, reduce_type=None, namespace="optim")
            if isinstance(self.optim, SkipStepOptimizer):
                self.optim.latest_grad_norm = grad_norm

        if self.scheduler is not None:
            for group_idx, group in enumerate(self.optim.param_groups):
                new_lr = self.scheduler.set_lr(group, self.trainer)
                self.trainer.record_metric(f"LR (group {group_idx})", new_lr, namespace="optim")

        self.optim.step()
        if isinstance(self.optim, SkipStepOptimizer):
            self.record_metric("step skipped", self.optim.step_skipped, namespace="optim")

        self.model.post_optim_step()

    def zero_grads(self):
        self.optim.zero_grad(set_to_none=True)

    def model_forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor | LMOutputWithLoss:
        with self._model_forward_context():
            return self.model(input_ids, labels=labels, **kwargs)

    def num_flops_per_token(self, seq_len: int) -> int:
        return self.model.num_flops_per_token(seq_len)

    @contextlib.contextmanager
    def _train_microbatch_context(self, micro_batch_idx: int, num_micro_batches: int) -> Generator[None, None, None]:
        is_last_mb = micro_batch_idx == num_micro_batches - 1
        with contextlib.ExitStack() as stack:
            if isinstance(self.model, FSDPModule):
                assert self.dp_config is not None
                self.model.set_is_last_backward(is_last_mb)
                if self.dp_config.name == DataParallelType.hsdp:
                    self.model.set_requires_all_reduce(is_last_mb)
            elif isinstance(self.model, DDP):
                if not is_last_mb:
                    stack.enter_context(self.model.no_sync())
            yield

    @contextlib.contextmanager
    def _eval_batch_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            yield

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(torch.autocast(self.device.type, dtype=self.autocast_precision))
            yield

    def _get_state_dict(self, sd_options: dist_cp_sd.StateDictOptions, optim: bool = True) -> dict[str, Any]:
        state_dict: dict[str, Any] = {"model": dist_cp_sd.get_model_state_dict(self.model, options=sd_options)}
        if optim:
            state_dict["optim"] = dist_cp_sd.get_optimizer_state_dict(self.model, self.optim, options=sd_options)
        return state_dict

    def _clip_grad_norm(
        self, max_grad_norm: float, norm_type: float = 2.0, foreach: bool | None = None
    ) -> torch.Tensor:
        if isinstance(self.model, FSDP):
            return self.model.clip_grad_norm_(max_grad_norm)

        parameters = [p for p in self.model.parameters()]
        grads = [p.grad for p in parameters if p.grad is not None]

        total_norm = nn.utils.get_total_norm(grads, norm_type=norm_type, error_if_nonfinite=False, foreach=foreach)

        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()

        torch.nn.utils.clip_grads_with_norm_(parameters, max_grad_norm, total_norm, foreach=foreach)
        return total_norm

    def _prepare_batch(
        self, batch: dict[str, Any], labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        input_ids = batch.pop("input_ids")
        labels = labels if labels is not None else batch.pop("labels", None)
        if "doc_lens" in batch and "max_doc_lens" in batch:
            log_once(log, "intra-document masking enabled")
        return input_ids, labels, batch

    def _set_model_mode(self, mode: Literal["train", "eval"]):
        if self._model_mode != mode:
            if mode == "train":
                self.model.train()
            elif mode == "eval":
                self.model.eval()
            else:
                raise ValueError(f"Invalid model mode: {mode}")
            self._model_mode = mode
