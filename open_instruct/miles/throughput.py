"""Explain requested throughput budgets without promising GPU fit or optimality."""

from open_instruct.miles import graph_config


def report(options, core):
    issues = []

    def warn(code, message):
        issues.append({"code": code, "message": message})

    gpus = options.get("rollout_num_gpus")
    tp = options.get("rollout_num_gpus_per_engine", 1)
    engines = gpus // tp if gpus else None
    running = options.get("sglang_max_running_requests")
    pool = options.get("sglang_max_total_tokens")
    context = options.get("rollout_max_context_len", getattr(core, "max_sequence_length", None))
    chunk = options.get("sglang_chunked_prefill_size")
    collection = options.get("rollout_batch_size", 0) * options.get("n_samples_per_prompt", 1)
    http = options.get("sglang_server_concurrency")
    asynchronous = options.get("fully_async", False)
    if pool and context and pool < context:
        warn(
            "token_pool_below_context",
            f"Token pool {pool} is smaller than context limit {context}; long requests may not fit. Increase sglang_max_total_tokens or lower the context limit after checking GPU memory.",
        )
    if pool and context and running and pool < context * running:
        warn(
            "token_pool_limits_long_concurrency",
            f"At full context, the requested token pool holds only {pool // context} of {running} running requests per engine. Prefix sharing and actual lengths affect this bound; measure effective concurrency before increasing the pool.",
        )
    if chunk and chunk > 0 and pool and chunk > pool:
        warn(
            "prefill_chunk_above_pool",
            "sglang_chunked_prefill_size exceeds the token pool; reduce the chunk or increase the pool after a memory check.",
        )
    if engines and running and http and collection and not asynchronous and collection < engines * min(http, running):
        warn(
            "sync_collection_underfeeds_fleet",
            f"Synchronous collection has {collection} responses for {engines * min(http, running)} requested serving slots. More inference GPUs may sit idle; adjust fleet size or deliberately change the collection size.",
        )
    decode_graph = graph_config.explicit_settings(options)["decode"]
    graph_limit = max(decode_graph["bs"]) if decode_graph.get("bs") else decode_graph.get("max_bs")
    graphs = decode_graph.get("backend") not in (None, "disabled")
    if graphs and graph_limit and running and graph_limit < running:
        warn(
            "decode_graph_coverage",
            f"Decode graphs cover batches up to {graph_limit}, below {running} requested running requests. Larger batches may fall back; measure graph coverage and memory before increasing the capture limit.",
        )
    if options.get("colocate", False):
        warn(
            "resident_colocation_fit",
            "Core stays resident during colocated serving. The model, optimizer, activations and inference pools must fit together; a model fitting for inference alone is insufficient.",
        )
    if getattr(core, "diagnostic_interval", 0) or getattr(core, "replay_diagnostics", False):
        warn(
            "diagnostic_overhead",
            "Trainer/publication or replay diagnostics are enabled. Keep them for correctness qualification; measure their cost separately when selecting throughput defaults.",
        )
    if options.get("kl_loss_coef", 0) > 0 or options.get("use_kl_loss", False):
        warn(
            "reference_policy_cost",
            "KL adds a frozen reference model and scoring pass; include its GPU memory and forward time when sizing trainers.",
        )
    if asynchronous and options.get("rollout_submission_granularity") == "group":
        warn(
            "group_straggler_backfill",
            "Group submission keeps a slot occupied until its slowest sibling and rewards finish. If engines are idle despite available prompts, compare sample backfill without changing whole-group training.",
        )
    for kind in ("eval", "save"):
        interval = options.get(f"{kind}_interval")
        if interval is not None and interval <= 5:
            warn(
                f"frequent_{kind}",
                f"{kind}_interval={interval} frequently interrupts the training loop. Appropriate for mechanics checks; measure lifecycle time separately from steady throughput.",
            )
    return {
        "scope": "Static advisory checks; requested limits do not certify GPU memory or measured throughput",
        "engines": engines,
        "requested_full_context_slots_per_engine": pool // context if pool and context else None,
        "publication_mode": getattr(core, "publication_mode", "barrier"),
        "warnings": issues,
        "measure_before_scaling": [
            "effective engine running capacity and peak GPU memory",
            "warm generation and reward latency",
            "trainer consumer wait and completed-queue token discard fraction",
            "publication latency including fleet acknowledgments",
        ],
    }
