"""CPU preflight for the native score-centering objective."""

from open_instruct.miles.configuration import validation
from open_instruct.miles.errors import InputError


def is_enabled(options):
    return options.get("loss_type") == "score_centering"


def validate(options, *, fp32_lm_head):
    """Validate score centering before Core resolves serving precision."""
    if not is_enabled(options):
        return
    if not fp32_lm_head:
        raise InputError("Score centering requires core.fp32_lm_head=true")
    validation.integer(options.get("rollout_top_logprobs_num", 0), "miles.rollout_top_logprobs_num")
    validation.choice(options.get("score_centering_is", "none"), "miles.score_centering_is", ("none", "tis", "mis"))
    validation.number(
        options.get("score_centering_tis_clip", 2.0), "miles.score_centering_tis_clip", exclusive_min=True
    )
    low, high = options.get("score_centering_mis_low", 0.5), options.get("score_centering_mis_high", 5.0)
    validation.number(low, "miles.score_centering_mis_low", exclusive_min=True)
    validation.number(high, "miles.score_centering_mis_high", minimum=low)
    if options.get("advantage_estimator", "grpo") != "grpo":
        raise InputError("Score centering requires advantage_estimator=grpo")
    for name in (
        "use_tis",
        "custom_tis_function_path",
        "use_opsm",
        "true_on_policy_mode",
        "recompute_logprobs_via_prefill",
        "sglang_speculative_algorithm",
        "custom_pg_loss_reducer_function_path",
        "multi_lora",
        "use_opd",
        "custom_convert_samples_to_train_data_path",
    ):
        if options.get(name):
            raise InputError(f"Score centering is incompatible with miles.{name}; use score_centering_is for TIS/MIS")
    if options.get("sglang_enable_fp32_lm_head") is False:
        raise InputError("Score centering requires miles.sglang_enable_fp32_lm_head=true")


def uses_importance_sampling(options):
    return is_enabled(options) and options.get("score_centering_is", "none") in ("tis", "mis")
