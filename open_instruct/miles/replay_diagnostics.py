"""Opt-in route assertions for live replay qualification, including recomputation.

These hooks synchronize small counters after every microbatch. They are intended
for correctness trials, not production throughput measurements.
"""

from contextlib import contextmanager

import torch

from open_instruct.miles import contract, data


@contextmanager
def checked_context(actor, module, batch, context):
    # Rollout IDs arrive on CPU; the router copies them to its compute device.
    # Keep diagnostic references/counters together without changing replay inputs.
    device = batch["tokens"].device
    expected = {name: ids.to(device) for name, ids in data.router_routes(module.model, batch).items()}
    routers = dict(module.model.named_modules())
    calls = {name: {"entered": 0, "returned": 0, "grad_enabled": 0} for name in expected}
    mismatches = torch.zeros((), dtype=torch.int64, device=batch["tokens"].device)
    handles = []

    def before(name, router, inputs):
        nonlocal mismatches
        counts = calls[name]
        counts["entered"] += 1
        counts["grad_enabled"] += int(torch.is_grad_enabled())
        actual = getattr(router, "replay_expert_indices", None)
        if actual is None or actual.shape != expected[name].shape:
            raise ValueError(f"Missing or malformed replay override: {name}")
        mismatches += (actual.to(device) != expected[name]).sum()

    def after(name, router, inputs, output):
        nonlocal mismatches
        actual = output[1]
        if actual is None or actual.shape != expected[name].shape:
            raise ValueError(f"Missing or malformed returned replay routes: {name}")
        calls[name]["returned"] += 1
        mismatches += (actual.to(device) != expected[name]).sum()

    try:
        for name in expected:
            handles.append(routers[name].register_forward_pre_hook(lambda m, x, name=name: before(name, m, x)))
            handles.append(routers[name].register_forward_hook(lambda m, x, y, name=name: after(name, m, x, y)))
        with context:
            yield
        # Validation-only contexts deliberately perform no forward.
        if any(row["entered"] for row in calls.values()):
            mismatch_count = int(mismatches)
            contract.record(
                actor.args,
                dict(
                    event="replay_routes",
                    rollout_id=actor.clock.next_rollout_id,
                    phase="training" if module.model.training else "scoring",
                    tokens=int(batch["tokens"].numel()),
                    captured_tokens=int(batch["tokens"].numel()) - len(batch.get("total_lengths", [None])),
                    synthetic_tail_tokens=len(batch.get("total_lengths", [None])),
                    samples=len(batch.get("total_lengths", [None])),
                    mismatches=mismatch_count,
                    layers=calls,
                ),
            )
            if mismatch_count:
                raise ValueError(f"Replay routes diverged: {mismatch_count} expert IDs")
    finally:
        for handle in handles:
            handle.remove()


def audit_contracts(contracts, updates, local_samples):
    """Require every rank/sample/layer and observed recomputation, not just flags."""
    if not contracts or updates < 1 or local_samples < 1:
        raise ValueError("Replay audit requires nonempty ranks, updates and samples")
    summary = {}
    inventory = None
    for rank, rows in contracts.items():
        replay = [row for row in rows if row["event"] == "replay_routes"]
        for update in range(updates):
            for phase in ("scoring", "training"):
                selected = [r for r in replay if r["rollout_id"] == update and r["phase"] == phase]
                if sum(r.get("samples", 1) for r in selected) != local_samples:
                    raise ValueError(f"Incomplete replay coverage: rank {rank}, update {update}, {phase}")
                for row in selected:
                    names = set(row["layers"])
                    if not names or (inventory is not None and names != inventory):
                        raise ValueError("Replay routed-layer inventory differs")
                    inventory = names
                    if row["mismatches"] or row["captured_tokens"] != row["tokens"] - row.get(
                        "synthetic_tail_tokens", 1
                    ):
                        raise ValueError("Replayed expert IDs or token alignment differ")
                    for counts in row["layers"].values():
                        minimum = 2 if phase == "training" else 1
                        if counts["entered"] < minimum or counts["returned"] < minimum:
                            raise ValueError("Missing returned routes during scoring or backward recomputation")
                        if phase == "training" and counts["grad_enabled"] < 1:
                            raise ValueError("Replay training never ran with gradients enabled")
        if inventory is None:
            raise ValueError("No routed layers observed")
        summary[rank] = {"checked_microbatches": len(replay), "routed_layers": len(inventory)}
    return {
        "passed": True,
        "ranks": summary,
        "synthetic_tail": "One unscored final token per sample; auxiliary semantics unchanged",
    }
