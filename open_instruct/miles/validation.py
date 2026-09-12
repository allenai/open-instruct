"""Small, CPU-only input checks shared by the run file and Python API."""

import difflib
import json
import math
from pathlib import Path

import tomllib

from open_instruct.miles.errors import InputError


def boolean(value, name):
    if type(value) is not bool:
        raise InputError(f"{name} must be a boolean; use true or false without quotes.")
    return value


def integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        requirement = "positive integer" if minimum == 1 else "nonnegative integer"
        raise InputError(f"{name} must be a {requirement}; got {value!r}.")
    return value


def number(value, name, *, minimum=0, maximum=None, exclusive_min=False, exclusive_max=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    bounds = f"{'>' if exclusive_min else '>='} {minimum}"
    if maximum is not None:
        bounds += f" and {'<' if exclusive_max else '<='} {maximum}"
    if (
        not finite
        or value < minimum
        or (exclusive_min and value == minimum)
        or (maximum is not None and (value > maximum or (exclusive_max and value == maximum)))
    ):
        raise InputError(f"{name} must be a finite number {bounds}; got {value!r}.")
    return value


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise InputError(f'{name} must be a nonempty string; use a quoted value, for example "value".')
    return value


def choice(value, name, choices):
    if value not in choices:
        raise InputError(f"{name} must be one of {', '.join(map(repr, choices))}; got {value!r}.")
    return value


def mapping(value, name):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise InputError(f"{name} must be a table/object with named fields.")
    return value


def fields(value, name, allowed):
    mapping(value, name)
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        hints = []
        for key in unknown:
            matches = difflib.get_close_matches(key, sorted(allowed), n=1)
            if matches:
                hints.append(f"{key!r}: did you mean {matches[0]!r}?")
        raise InputError(f"Unknown {name} fields: {unknown}. " + (" ".join(hints) or "Remove these fields."))


def read_document(path):
    path = Path(path).expanduser()
    try:
        raw = path.read_text()
    except (OSError, UnicodeError) as error:
        raise InputError(f"Cannot read configuration {path}: {error}. Check the path and read permissions.") from error
    try:
        document = json.loads(raw) if path.suffix == ".json" else tomllib.loads(raw)
    except ValueError as error:
        raise InputError(f"Invalid configuration in {path}: {error}") from error
    return mapping(document, "Configuration")


def override_value(key, raw):
    try:
        return tomllib.loads("value=" + raw)["value"]
    except (tomllib.TOMLDecodeError, KeyError) as error:
        raise InputError(
            f"Invalid TOML value in override {key}; quote strings, for example --set '{key}=\"value\"'. "
            "Use unquoted true/false for booleans."
        ) from error


def runtime_values(values):
    """Check explicit common controls; preserve unrelated native sentinel values."""
    for key in (
        "actor_num_nodes",
        "actor_num_gpus_per_node",
        "num_gpus_per_node",
        "rollout_num_gpus",
        "rollout_num_gpus_per_engine",
        "global_batch_size",
        "rollout_batch_size",
        "n_samples_per_prompt",
        "num_rollout",
        "rollout_max_prompt_len",
        "rollout_max_response_len",
        "rollout_max_context_len",
        "eval_max_response_len",
        "eval_max_context_len",
        "n_samples_per_eval_prompt",
        "sglang_context_length",
        "sglang_server_concurrency",
        "sglang_max_running_requests",
        "sglang_max_total_tokens",
        "save_interval",
        "eval_interval",
        "lr_decay_iters",
        "update_weight_buffer_size",
    ):
        if key in values:
            integer(values[key], f"miles.{key}")
    for key in ("lr_warmup_iters", "seed", "rollout_seed"):
        if key in values:
            integer(values[key], f"miles.{key}", minimum=0)
    for key in (
        "lr",
        "min_lr",
        "lr_warmup_init",
        "weight_decay",
        "clip_grad",
        "eps_clip",
        "eps_clip_high",
        "rollout_temperature",
        "eval_temperature",
        "kl_loss_coef",
    ):
        if key in values:
            number(values[key], f"miles.{key}")
    for key in ("adam_eps", "async_data_buffer_capacity_factor"):
        if key in values:
            number(values[key], f"miles.{key}", exclusive_min=True)
    for key in ("adam_beta1", "adam_beta2", "lr_warmup_fraction"):
        if key in values:
            number(values[key], f"miles.{key}", maximum=1, exclusive_max=True)
    for key in ("sglang_mem_fraction_static", "rollout_top_p", "eval_top_p"):
        if key in values:
            number(values[key], f"miles.{key}", maximum=1, exclusive_min=True)
    if "lr" in values and values.get("min_lr", 0) > values["lr"]:
        raise InputError("miles.min_lr must be <= miles.lr; lower min_lr or raise the initial learning rate.")
    if "lr" in values and values.get("lr_warmup_init", 0) > values["lr"]:
        raise InputError("miles.lr_warmup_init must be <= miles.lr; warm up to the configured learning rate.")
    if (
        values.get("lr_warmup_fraction") is None
        and "lr_decay_iters" in values
        and values.get("lr_warmup_iters", 0) >= values["lr_decay_iters"]
    ):
        raise InputError("miles.lr_warmup_iters must be < miles.lr_decay_iters; leave at least one step after warmup.")


# Each running request can hold this many KDA recurrent-state slots under the
# branch-capable ``extra_buffer`` radix strategy (overlap allowance at chunk
# boundaries and branches); the pool must leave room for retained prefixes.
KDA_RADIX_STATE_SLOTS_PER_RUNNING_REQUEST = 5


def inference_capacity(values):
    """Port of olmo-miles' radix-cache and router capacity rules for the resolved options.

    Only explicit values are checked; the pinned runtime supplies its own defaults
    for anything absent.
    """
    running = values.get("sglang_max_running_requests")
    slots = values.get("sglang_max_mamba_cache_size")
    if values.get("sglang_disable_radix_cache", False):
        if running is not None and slots is not None and slots < running:
            raise InputError(
                "miles.sglang_max_mamba_cache_size must be at least sglang_max_running_requests "
                "when the radix cache is disabled"
            )
    elif "sglang_max_mamba_cache_size" in values or "sglang_mamba_radix_cache_strategy" in values:
        # These constraints belong to explicitly configured recurrent-state
        # caching. Dense attention and unresolved defaults have no KDA slots.
        strategy = values.get("sglang_mamba_radix_cache_strategy")
        if strategy != "extra_buffer":
            raise InputError(
                'the validated KDA radix path requires inference.mamba_radix_cache_strategy = "extra_buffer" '
                "when the radix cache is enabled"
            )
        if values.get("sglang_attention_backend", "triton") != "triton":
            raise InputError('the validated KDA radix path requires sglang_attention_backend = "triton"')
        if values.get("sglang_page_size", 1) != 1:
            raise InputError("the validated KDA radix path requires sglang_page_size = 1")
        if values.get("sglang_disable_overlap_schedule", False):
            raise InputError("the validated KDA radix path requires overlap scheduling enabled")
        if running is not None and slots is not None and slots <= KDA_RADIX_STATE_SLOTS_PER_RUNNING_REQUEST * running:
            raise InputError(
                "miles.sglang_max_mamba_cache_size must exceed "
                f"{KDA_RADIX_STATE_SLOTS_PER_RUNNING_REQUEST} * sglang_max_running_requests "
                "when the radix cache is enabled, leaving at least one slot for a retained prefix state"
            )
    if "router_cache_threshold" in values:
        number(values["router_cache_threshold"], "miles.router_cache_threshold", minimum=0.0, maximum=1.0)
    if "router_balance_abs_threshold" in values:
        integer(values["router_balance_abs_threshold"], "miles.router_balance_abs_threshold", minimum=0)
    if "router_balance_rel_threshold" in values:
        number(values["router_balance_rel_threshold"], "miles.router_balance_rel_threshold", minimum=1.0)
