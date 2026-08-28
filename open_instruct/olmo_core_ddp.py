"""Config helpers for olmo-core models that train under ``OLMoDDPModel``.

Separate from ``olmo_core_utils`` because ``OLMoDDPOptimizerConfig`` and
``OLMoDDPTrainModuleConfig`` exist only on the olmo-core revisions carrying the MoE/KDA
work. Importing them from the shared utilities would make every open-instruct entrypoint
-- SFT, GRPO, dense DPO, even the dataset tooling -- fail to import on any other pin.
"""

import json
from typing import Any

from olmo_core.optim.moe_optimizer import OLMoDDPOptimizerConfig
from olmo_core.train.train_module.transformer.config import OLMoDDPTrainModuleConfig, TransformerDataParallelConfig

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def build_ddp_train_module_config(
    config_path: str,
    rank_microbatch_size: int,
    max_sequence_length: int,
    learning_rate: float,
    weight_decay: float,
    scheduler: Any,
    max_grad_norm: float | None,
    compile_model: bool,
    ac_config: Any,
) -> OLMoDDPTrainModuleConfig:
    """Build the nn.ddp train-module config for OLMoDDPModel (MoE v2) models.

    These models refuse FSDP -- prepare_experts_for_fsdp raises by design -- and
    train through the branch's own DDP train module instead. The optimizer and
    dp_config come verbatim from the checkpoint's train_module section (the
    combination that pretrained the model), with only the fine-tuning knobs
    overridden: lr, weight decay, scheduler, batch geometry and grad clipping.
    """
    with open(config_path) as config_file:
        payload = json.load(config_file)
    checkpoint_train_module = payload["train_module"]

    optim_config = OLMoDDPOptimizerConfig.from_dict(checkpoint_train_module["optim"])
    optim_config.lr = learning_rate
    optim_config.weight_decay = weight_decay

    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=max_sequence_length,
        optim=optim_config,
        dp_config=TransformerDataParallelConfig.from_dict(checkpoint_train_module["dp_config"]),
        scheduler=scheduler,
        compile_model=compile_model,
        ac_config=ac_config,
        max_grad_norm=max_grad_norm,
        z_loss_multiplier=checkpoint_train_module.get("z_loss_multiplier"),
    )
