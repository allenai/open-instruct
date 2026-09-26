"""CPU-safe sizing and diagnostics for the existing FIFO async pipeline."""

import math


def producer_default(options):
    """One collection or two waves of requested serving slots, in whole groups."""
    samples = options["n_samples_per_prompt"]
    engines = options["rollout_num_gpus"] // options["rollout_num_gpus_per_engine"]
    slots = min(options["sglang_server_concurrency"], options["sglang_max_running_requests"])
    target = max(options["rollout_batch_size"] * samples, 2 * engines * slots)
    return ((target + samples - 1) // samples) * samples


def report(options, max_policy_lag):
    """Requested capacities, not a claim about model fit or achieved throughput.

    Missing low-level settings stay unresolved rather than assuming native defaults.
    The runtime calls this again after parsing supplies those defaults.
    """
    if not options.get("fully_async", False):
        return {"enabled": False, "warnings": []}
    samples = options.get("n_samples_per_prompt")
    groups = options.get("rollout_batch_size")
    gpus = options.get("rollout_num_gpus")
    tp = options.get("rollout_num_gpus_per_engine")
    engines = gpus // tp if gpus and tp else None
    http = options.get("sglang_server_concurrency")
    running = options.get("sglang_max_running_requests")
    http_slots = engines * http if engines is not None and http else None
    engine_slots = engines * running if engines is not None and running else None
    usable_slots = min(http_slots, engine_slots) if http_slots is not None and engine_slots is not None else None
    collection = groups * samples if groups and samples else None
    requested = options.get("async_max_concurrent_samples")
    effective = max(1, requested // samples) * samples if requested is not None and samples else collection
    factor = options.get("async_data_buffer_capacity_factor")
    buffer_groups = math.floor(factor * groups) if factor is not None and groups else None
    buffer_samples = buffer_groups * samples if buffer_groups is not None and samples else None
    batch = options.get("global_batch_size")
    warnings = []
    if requested is not None and effective is not None and requested != effective:
        warnings.append(
            f"async_max_concurrent_samples={requested} resolves to {effective} samples in whole groups of {samples}; "
            "use a multiple of n_samples_per_prompt."
        )
    if usable_slots is not None and effective is not None and effective < usable_slots:
        warnings.append(
            f"Producer budget {effective} is below {usable_slots} requested serving slots across {engines} engines; "
            "raise async_max_concurrent_samples or omit it in a structured run for automatic sizing."
        )
    if http is not None and running is not None and http < running:
        warnings.append(
            f"sglang_server_concurrency={http} is below sglang_max_running_requests={running}; "
            "the global HTTP semaphore cannot supply all requested engine slots. Raise server_concurrency if intentional capacity is unused."
        )
    if buffer_samples is not None and collection is not None and buffer_samples < collection:
        warnings.append(
            f"Completed buffer holds {buffer_samples} samples, less than one collection ({collection}); "
            "streaming assembly works, but bursts may block production. Consider async_data_buffer_capacity_factor >= 1."
        )
    if (
        buffer_samples is not None
        and batch
        and max_policy_lag is not None
        and buffer_samples > (max_policy_lag + 1) * batch
    ):
        warnings.append(
            f"Completed buffer spans {buffer_samples / batch:g} optimizer batches with max_policy_lag={max_policy_lag}; "
            "queued groups may expire before consumption. Reduce async_data_buffer_capacity_factor and measure stale-group discards."
        )
    if effective is not None and batch and max_policy_lag is not None and effective > (max_policy_lag + 1) * batch:
        warnings.append(
            f"Producer budget alone covers {effective / batch:g} optimizer batches with max_policy_lag={max_policy_lag}. "
            "This is headroom, not a predicted age: compare completed-queue drops and trainer wait before raising it further. "
            "If drops grow while training stays busy, reduce async_max_concurrent_samples."
        )
    combined = (
        effective + buffer_samples
        if options.get("rollout_submission_granularity") == "group"
        and effective is not None
        and buffer_samples is not None
        else None
    )
    if combined is not None and batch and max_policy_lag is not None and combined > max_policy_lag * batch:
        warnings.append(
            f"Producer plus completed buffer can retain {combined / batch:g} future optimizer batches "
            f"with max_policy_lag={max_policy_lag}. A fast generator can fill both under one policy while training runs; "
            "older groups may then expire. This is a capacity warning, not a predicted discard rate. "
            "Compare drops and trainer wait when reducing async_max_concurrent_samples or the completed-buffer factor."
        )
    if options.get("rollout_submission_granularity") == "sample":
        warnings.append(
            "Sample backfill limits unfinished samples, not all retained siblings. Partly completed groups can hold "
            "more responses than producer_sample_budget; monitor retained work, long-response age and memory."
        )
    return {
        "enabled": True,
        "scope": "requested capacities; engine memory/state pools and routing can reduce utilization",
        "engines": engines,
        "global_http_slots": http_slots,
        "requested_engine_slots": engine_slots,
        "producer_sample_budget": effective,
        "producer_group_budget": effective // samples if effective is not None and samples else None,
        "submission_granularity": options.get("rollout_submission_granularity") or "sample",
        "completed_buffer_groups": buffer_groups,
        "completed_buffer_samples": buffer_samples,
        "owned_plus_buffer_samples": combined,
        "samples_per_collection": collection,
        "samples_per_optimizer_step": batch,
        "warnings": warnings,
    }
