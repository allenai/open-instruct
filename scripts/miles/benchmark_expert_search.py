"""CPU replay of expert scheduling on captured per-document routing histograms.

These panel histograms describe the recorded tokens, not live RL collections or
synthetic replay-tail metadata. Timing excludes building histograms from token IDs.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from open_instruct.miles.training import expert_schedule


def benchmark(panel, configurations, budgets):
    for population in sorted({row["population"] for row in panel}):
        rows = [r for r in panel if r["checkpoint"] == "initial" and r["population"] == population]
        layers, docs = sorted({r["layer"] for r in rows}), sorted({r["id"] for r in rows})
        counts = {(r["id"], r["layer"]): np.asarray(r["counts"], dtype=np.int64) for r in rows}
        tokens = {r["id"]: r["tokens"] for r in rows}
        for world, ep, token_budget in configurations:
            usable = len(docs) // world * world
            if not usable:
                continue
            lengths = [tokens[d] for d in docs[:usable]]
            if max(lengths) > token_budget:
                continue
            hist = np.stack(
                [np.stack([counts[d, layer].reshape(ep, -1).sum(axis=1) for layer in layers]) for d in docs[:usable]]
            )
            settings = dict(world=world, ep_degree=ep, max_tokens=token_budget)
            for seconds, proposals in budgets:
                statistics = {}
                started = time.perf_counter()
                order, before, after = expert_schedule.plan_order(
                    lengths,
                    hist,
                    **settings,
                    seed=17,
                    max_proposals=proposals,
                    search_seconds=seconds,
                    statistics=statistics,
                )
                elapsed = time.perf_counter() - started
                # Rebuild from scratch independently of the incremental cache.
                expected = expert_schedule.measure(order, lengths, hist, **settings)
                assert expected == after
                assert all(after[k] <= before[k] for k in before)
                yield dict(
                    population=population,
                    world=world,
                    ep=ep,
                    token_budget=token_budget,
                    seconds=seconds,
                    proposal_budget=proposals,
                    samples=usable,
                    before=before,
                    after=after,
                    elapsed_seconds=elapsed,
                    statistics=statistics,
                    reordered=order != list(range(usable)),
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.panel.read_bytes()
    configurations = [
        (4, 2, 4096),
        (8, 2, 4096),
        (16, 8, 4096),
        (16, 4, 4096),
        (16, 8, 34816),
        (16, 4, 34816),
        (32, 8, 34816),
    ]
    budgets = [(0, 0), (0.025, 1024), (0.1, 1024), (0.25, 1024), (10, 4000)]
    result = dict(panel=str(args.panel), sha256=hashlib.sha256(raw).hexdigest(), results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in benchmark(json.loads(raw), configurations, budgets):
        result["results"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        gain = 100 * (1 - row["after"]["critical_work_proxy"] / row["before"]["critical_work_proxy"])
        print(
            f"{row['population']:7} EP{row['ep']} DP{row['world'] // row['ep']} "
            f"budget={row['token_budget']} search={row['seconds']}s "
            f"work reduction={gain:.2f}% elapsed={row['elapsed_seconds'] * 1000:.1f}ms "
            f"{row['statistics']['stop']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
