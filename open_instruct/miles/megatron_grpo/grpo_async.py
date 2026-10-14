"""Fully async producer for teacher-free verifier GRPO."""

from open_instruct.miles.distillation import opd_async
from open_instruct.miles.megatron_grpo import grpo_config


class VerifierAsyncRollout(opd_async.OPDAsyncRollout):
    """The async OPD producer, restart ledger and optimizer-step age budget for verifier GRPO.

    Held-out prompts are scored by the training verifier without its group-centering
    post-process. Nothing here touches a teacher; the parent's opt-in OPD selection-audit and
    timing hooks stay off unless their ``OI_OPD_*`` switches are set.
    """

    eval_reward_path = grpo_config.REWARD_HOOK
