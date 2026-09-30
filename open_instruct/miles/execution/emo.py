"""Resolve EMO checkpoint ancestry into an explicit causal RL execution mode."""

import copy

from open_instruct.miles.errors import InputError

FIELDS = (
    "emo_min_document_expert_pool",
    "emo_max_document_expert_pool",
    "emo_eval_document_expert_pool",
    "emo_eos_token_id",
)


def resolve_hf(config, requested_mode):
    """Return an owned config; only an explicit request may change source behavior."""
    result = copy.deepcopy(config)
    mode = result.get("emo_routing_mode")
    if requested_mode not in (None, "full_pool") or mode not in (None, "full_pool"):
        raise InputError("EMO supports only emo_routing_mode='full_pool'")
    active = any(result.get(key) is not None for key in FIELDS)
    if not active:
        if requested_mode is not None or mode is not None:
            raise InputError("Full-pool EMO selection requires an EMO-bearing checkpoint")
        return result
    if result.get("model_type") != "olmo3moe":
        raise InputError("EMO requires model_type='olmo3moe'")
    if result.get("gating_function", "softmax") != "softmax" or result.get("normalize_expert_weights", 1.0) != 1.0:
        raise InputError("EMO serving requires softmax gating with L1-normalized expert weights")
    if requested_mode is None and mode != "full_pool":
        raise InputError("EMO RL requires model.emo_routing_mode='full_pool'; document-pool execution is unsupported")
    experts, top_k = result.get("n_routed_experts"), result.get("num_experts_per_tok")
    if type(experts) is not int or type(top_k) is not int or not 0 < top_k <= experts:
        raise InputError("EMO requires 0 < num_experts_per_tok <= n_routed_experts")
    for key in FIELDS:
        value = result.get(key)
        # Native pretraining may leave eval pool unset. Explicit selection resolves it.
        if key == "emo_eval_document_expert_pool" and value is None and requested_mode:
            continue
        if type(value) is not int or value < (0 if key == "emo_eos_token_id" else 1):
            raise InputError(f"EMO requires a valid integer {key}")
    if not top_k <= result[FIELDS[0]] <= result[FIELDS[1]] <= experts:
        raise InputError("EMO requires top_k <= min_document_expert_pool <= max_pool <= num_experts")
    evaluation = result.get("emo_eval_document_expert_pool")
    if evaluation is not None and not top_k <= evaluation <= experts:
        raise InputError("EMO evaluation pool must be between top_k and num_experts")
    if requested_mode is None and evaluation != experts:
        raise InputError("Full-pool EMO checkpoint has a restricted evaluation pool")
    if requested_mode:
        if result.get("emo_source_config") is None:
            result["emo_source_config"] = {key: result.get(key) for key in FIELDS}
        result["emo_eval_document_expert_pool"] = experts
    result["emo_routing_mode"] = "full_pool"
    return result


def resolve_native(config, requested_mode):
    """Resolve routers in copied native block configs before HF conversion."""
    result = copy.deepcopy(config)
    found = False

    def visit(node):
        nonlocal found
        if not isinstance(node, dict):
            return
        router = node.get("routed_experts_router")
        if isinstance(router, dict) and router.get("emo") is not None:
            found = True
            source = router["emo"]
            metadata = {key: source.get(key.removeprefix("emo_")) for key in FIELDS}
            resolved = resolve_hf(
                {
                    **metadata,
                    "model_type": "olmo3moe",
                    "n_routed_experts": router.get("num_experts"),
                    "num_experts_per_tok": router.get("top_k"),
                    "emo_routing_mode": "full_pool" if source.get("full_pool") else None,
                },
                requested_mode,
            )
            source["full_pool"] = True
            source["eval_document_expert_pool"] = resolved["emo_eval_document_expert_pool"]
            if source.get("source_config") is None:
                source["source_config"] = resolved.get("emo_source_config")
        for value in node.values():
            if isinstance(value, dict):
                visit(value)

    visit(result)
    if requested_mode and not found:
        raise InputError("Full-pool EMO selection requires an EMO-bearing native checkpoint")
    return result
