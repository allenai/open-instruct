"""DPO train module for olmo-core MoE checkpoints that train under OLMoDDPModel.

Kept out of ``olmo_core_train_modules`` so that the GRPO path, which shares that module,
does not acquire a dependency on ``OLMoDDPTrainModule`` -- a class that exists only on the
olmo-core revisions carrying the MoE/KDA work.
"""

from typing import Any

import torch
from olmo_core.io import file_exists, normalize_path
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.ddp.model import OLMoDDPModel
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizerConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.globals import set_global_arg
from olmo_core.train.train_module.transformer import OLMoDDPTrainModule
from olmo_core.train.train_module.transformer import config as transformer_config

from open_instruct import dpo_utils, logger_utils, olmo_core_ddp
from open_instruct.olmo_core_train_modules import DPOLMHead, DPOObjectiveMixin

logger = logger_utils.setup_logger(__name__)


class MoEDDPDPOTrainModule(DPOObjectiveMixin, OLMoDDPTrainModule):
    """DPO on an ``OLMoDDPModel`` MoE.

    The objective is ``DPOObjectiveMixin``, shared verbatim with the dense
    ``DPOTrainModule``; this class only supplies the stepping that
    ``OLMoDDPTrainModule`` needs and that ``TransformerTrainModule`` does not:

    - the model is ``model_parts`` (a list), not a single module;
    - gradients are reduced explicitly via ``finalize_grad_reduce()``, because
      ``_train_microbatch_context`` accumulates every non-final micro-batch under
      ``no_sync()``;
    - a dry run has to reset the MoE's auxiliary (load-balancing) metrics and signal
      completion, or the router statistics from the dry run leak into step 1.

    These models refuse FSDP2 by design -- ``prepare_experts_for_fsdp`` raises -- so this
    is the only train module they can use.
    """

    def __init__(
        self,
        model: OLMoDDPModel,
        optim: OLMoDDPOptimizerConfig,
        sample_microbatch_size: int,
        max_sequence_length: int,
        dpo_config: dpo_utils.DPOExperimentConfig,
        attn_implementation: AttentionBackendName,
        dp_config: transformer_config.TransformerDataParallelConfig,
        ep_config: transformer_config.TransformerExpertParallelConfig | None = None,
        ac_config: transformer_config.TransformerActivationCheckpointingConfig | None = None,
        compile_model: bool = False,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        z_loss_multiplier: float | None = None,
        reset_optimizer_states_on_load: bool = True,
        **kwargs: Any,
    ) -> None:
        self._assert_packing_backend(dpo_config, attn_implementation)
        # TODO(finbarrtimbers): Remove this hack once Transformer supports configuring the LM head.
        model.lm_head.__class__ = DPOLMHead
        # A DPO micro-batch carries a chosen and a rejected sequence per instance, hence
        # the factor of 2. olmo-core requires rank_microbatch_size % max_sequence_length == 0.
        rank_microbatch_size_tokens = sample_microbatch_size * max_sequence_length * 2
        super().__init__(
            model=model,
            optim=optim,
            rank_microbatch_size=rank_microbatch_size_tokens,
            max_sequence_length=max_sequence_length,
            dp_config=dp_config,
            ep_config=ep_config,
            ac_config=ac_config,
            compile_model=compile_model,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            device=device,
            z_loss_multiplier=z_loss_multiplier,
            reset_optimizer_states_on_load=reset_optimizer_states_on_load,
            **kwargs,
        )
        # self.device is resolved by the base class (get_default_device() when device is
        # None), so the metric accumulators must be built from it rather than the argument.
        self._init_dpo_objective(dpo_config, sample_microbatch_size, self.device)

    def train_batch(self, batch: dict[str, Any], dry_run: bool = False) -> None:
        self._require_optimizer()

        for model in self.model_parts:
            model.train()

        total_tokens, device = self._accumulate_dpo_microbatches(batch)

        # Two loops, matching OLMoDDPTrainModule.train_batch: every part's gradients are
        # reduced before any part runs post_batch.
        for model in self.model_parts:
            model.finalize_grad_reduce()
        for model in self.model_parts:
            model.post_batch(dry_run=dry_run)

        if dry_run:
            # The dry run exists to allocate peak memory, not to measure anything. Its
            # router counts would otherwise be folded into the first real step's
            # load-balancing metrics.
            for model in self.model_parts:
                model.reset_auxiliary_metrics()
            torch.cuda.empty_cache()
            set_global_arg("dry_run_done", True)
            return

        self._record_dpo_metrics(batch, total_tokens, device)


def build_moe_ddp_dpo_train_module(
    model: OLMoDDPModel,
    args: dpo_utils.DPOExperimentConfig,
    scheduler: Scheduler,
    ac_config: transformer_config.TransformerActivationCheckpointingConfig | None,
    max_grad_norm: float | None,
    device: torch.device,
) -> MoEDDPDPOTrainModule:
    """Build the MoE DPO train module from the checkpoint's own train_module config.

    The optimizer and dp_config come from the checkpoint rather than from open-instruct's
    defaults: an OLMoDDPModel's optimizer shards its state across the device mesh, so
    substituting a plain AdamW silently changes both the memory profile and the update.
    Only the DPO-specific knobs (lr, weight decay, schedule, batch geometry) are overridden.
    """
    assert args.config_name is not None and args.config_name.endswith(".json"), (
        "OLMoDDPModel requires --config_name pointing at the checkpoint's config json"
    )
    # Only .optim and .dp_config are read off the result -- the rest of the config
    # (batch geometry, schedule, activation checkpointing) is passed to the train module
    # directly below, which is also where rank_microbatch_size is derived.
    ddp_config = olmo_core_ddp.build_ddp_train_module_config(
        args.config_name,
        rank_microbatch_size=args.per_device_train_batch_size * args.max_seq_length * 2,
        max_sequence_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler=scheduler,
        max_grad_norm=max_grad_norm,
        compile_model=args.compile_model,
        ac_config=ac_config,
    )
    return MoEDDPDPOTrainModule(
        model=model,
        optim=ddp_config.optim,
        sample_microbatch_size=args.per_device_train_batch_size,
        max_sequence_length=args.max_seq_length,
        dpo_config=args,
        attn_implementation=args.attn_implementation,
        dp_config=ddp_config.dp_config,
        ac_config=ac_config,
        compile_model=args.compile_model,
        max_grad_norm=max_grad_norm,
        scheduler=scheduler,
        device=device,
        # Deliberately not the checkpoint's z_loss_multiplier: DPOLMHead returns per-token
        # log-probabilities rather than a cross-entropy, so there are no logits left for a
        # z-loss to regularize. Passing it through would be silently inert.
        z_loss_multiplier=None,
    )


def load_olmo_core_base_weights(train_module: OLMoDDPTrainModule, checkpoint_dir: str, work_dir: str) -> None:
    """Load SFT weights from an olmo-core checkpoint into an already-parallelized train module.

    DPO needs the weights in place before the reference log-probability cache is built, which
    happens before the Trainer exists, so this bypasses the Trainer's own load. The optimizer
    state is deliberately skipped: this is weight initialization, not a resume.

    Mirrors the Checkpointer's directory handling -- a full train checkpoint keeps its tensors
    under ``model_and_optim/``, while a weights-only export has them at the top level.
    """
    train_module_dir = f"{normalize_path(checkpoint_dir)}/model_and_optim"
    if not file_exists(f"{train_module_dir}/.metadata"):
        logger.info(f"No model_and_optim/ under {checkpoint_dir}; loading tensors from the directory itself")
        train_module_dir = normalize_path(checkpoint_dir)
    logger.info(f"Loading olmo-core weights from {train_module_dir}...")
    train_module.load_state_dict_direct(train_module_dir, work_dir=work_dir, load_optim_state=False)
