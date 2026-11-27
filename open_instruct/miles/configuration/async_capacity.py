"""Describe the work that an asynchronous rollout pipeline is configured to hold.
The report relates serving slots, prompt groups, completed-buffer capacity and
policy lag so users can spot settings that leave engines idle or retain stale
responses. A separate helper sizes the default producer budget in whole groups;
these calculations use requested limits, not model speed or GPU-memory estimates.
"""

import math


def producer_default(options):
    """One collection or two waves of requested serving slots, in whole groups."""
    samples = options["n_samples_per_prompt"]
    engines = options["rollout_num_gpus"] // options["rollout_num_gpus_per_engine"]
    slots = min(options["sglang_server_concurrency"], options["sglang_max_running_requests"])
    target = max(options["rollout_batch_size"] * samples, 2 * engines * slots)
    return ((target + samples - 1) // samples) * samples


def jit_default_bounds(
    *,
    rollout_batch_size,
    producer_group_budget,
    buffer_capacity_factor,
    max_retained_batches=None,
    prepared_batch_overlap=False,
):
    """Mirror miles.rollout.jit_pacer.default_bounds without importing the GPU runtime."""
    raw_min = 2.0 if prepared_batch_overlap else 1.5
    if max_retained_batches is not None:
        max_batches = max_retained_batches
    else:
        max_batches = max(raw_min, producer_group_budget / rollout_batch_size + buffer_capacity_factor)
    min_batches = max(1.0, min(raw_min, max_batches))
    initial_batches = min(max_batches, min_batches + 1.0)
    return min_batches, max_batches, initial_batches


def report(options, max_policy_lag, *, prepared_batch_overlap=False):
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
    auto_jit_pacing = bool(options.get("async_auto_jit_pacing", False))
    max_retained_batches = options.get("async_max_retained_batches")
    jit_min_batches = jit_max_batches = jit_initial_batches = None
    if auto_jit_pacing and groups and effective is not None and samples and factor is not None:
        jit_min_batches, jit_max_batches, jit_initial_batches = jit_default_bounds(
            rollout_batch_size=groups,
            producer_group_budget=effective // samples,
            buffer_capacity_factor=factor,
            max_retained_batches=(
                max_retained_batches
                if max_retained_batches is not None
                and math.isfinite(max_retained_batches)
                and max_retained_batches >= 1.0
                else None
            ),
            prepared_batch_overlap=prepared_batch_overlap,
        )
    max_retained_groups = (
        math.floor(max_retained_batches * groups)
        if max_retained_batches is not None and math.isfinite(max_retained_batches) and groups
        else None
    )
    max_retained_samples = max_retained_groups * samples if max_retained_groups is not None and samples else None
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
            "some reserved serving capacity cannot be filled. This can be intentional when inference supplies "
            "the trainer fast enough. Increase async_max_concurrent_samples only if generation limits useful "
            "throughput; compare trainer wait, queue age and stale-work discards."
        )
    if http is not None and running is not None and http < running:
        warnings.append(
            f"sglang_server_concurrency={http} is below sglang_max_running_requests={running}; "
            "the global HTTP semaphore cannot supply all requested engine slots. Raise server_concurrency if intentional capacity is unused."
        )
    if engine_slots is not None and effective == engine_slots:
        warnings.append(
            "Producer budget equals one wave of GPU running slots; unfinished samples in transit or awaiting "
            "response processing can leave engines idle. Increase async_max_concurrent_samples for backfill "
            "(start with two waves), then measure occupancy and policy age. HTTP capacity below running admission is reported separately."
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
    if max_retained_batches is not None and (not math.isfinite(max_retained_batches) or max_retained_batches < 1.0):
        warnings.append(
            f"async_max_retained_batches={max_retained_batches} is below 1.0; "
            "set async_max_retained_batches >= 1.0 so a full rollout batch can be assembled."
        )
    if (
        max_retained_samples is not None
        and batch
        and max_policy_lag is not None
        and max_retained_samples > max_policy_lag * batch
    ):
        warnings.append(
            f"async_max_retained_batches={max_retained_batches:g} ({max_retained_groups} groups, "
            f"{max_retained_samples} samples) spans {max_retained_samples / batch:g} optimizer batches with "
            f"max_policy_lag={max_policy_lag}. Under trainer backpressure, older retained groups may expire before "
            "consumption; reduce async_max_retained_batches and measure stale-group discards."
        )
    if (
        max_retained_groups is not None
        and effective is not None
        and samples
        and max_retained_groups < effective // samples
    ):
        warnings.append(
            f"async_max_retained_batches={max_retained_batches:g} ({max_retained_groups} groups) is below "
            f"producer_group_budget={effective // samples}; retained ownership will cap active groups before "
            "async_max_concurrent_samples is reached."
        )
    if (
        options.get("rollout_submission_granularity") == "sample"
        and max_retained_batches is None
        and not auto_jit_pacing
    ):
        warnings.append(
            "Sample backfill limits unfinished samples, not all retained siblings. Partly completed groups and "
            "completed groups awaiting a full FIFO can exceed producer_sample_budget; set "
            "--async-max-retained-batches or --async-auto-jit-pacing to bound total retained groups or monitor "
            "retained work, long-response age and memory."
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
        "auto_jit_pacing": auto_jit_pacing,
        "jit_min_retained_batches": jit_min_batches,
        "jit_max_retained_batches": jit_max_batches,
        "jit_initial_retained_batches": jit_initial_batches,
        "max_retained_batches": max_retained_batches,
        "max_retained_groups": max_retained_groups,
        "max_retained_samples": max_retained_samples,
        "samples_per_collection": collection,
        "samples_per_optimizer_step": batch,
        "warnings": warnings,
    }
