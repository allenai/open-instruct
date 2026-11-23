"""Score format_probe.py raw outputs with the training verifier and print a comparison table.

Reward follows training: MathVerifier score, or 0 when the response was truncated (finish type "length").
Needs open_instruct importable and ANTLR 4.11 (e.g. a venv with antlr4-python3-runtime==4.11).

Usage: PYTHONPATH=/path/to/open-instruct python score_format_probe.py PROBE_DIR
"""

import json
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

from open_instruct.ground_truth_utils import MathVerifier

VERIFIER = MathVerifier()


def score(item):
    text, label, truncated = item
    return 0.0 if truncated else VERIFIER([], text, label).score


def summarize(rows):
    rewards = [x for r in rows for x in r["rewards"]]
    return {
        "prompts": len(rows),
        "mean_reward": round(statistics.mean(rewards), 4),
        "pass_at_k": round(statistics.mean(max(r["rewards"]) for r in rows), 4),
        "mixed_groups": round(statistics.mean(0 < sum(r["rewards"]) < len(r["rewards"]) for r in rows), 4),
        "truncated": round(statistics.mean(x for r in rows for x in r["truncated"]), 4),
        "boxed": round(statistics.mean(x for r in rows for x in r["boxed"]), 4),
        "mean_tokens": round(statistics.mean(x for r in rows for x in r["tokens"]), 1),
        "median_tokens": statistics.median(x for r in rows for x in r["tokens"]),
    }


def main():
    root = Path(sys.argv[1])
    table = []
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 32) as pool:
        for path in sorted(root.glob("*.raw.json")):
            data = json.loads(path.read_text())
            items, shape = [], []
            for prompt in data["prompts"]:
                for response in prompt["responses"]:
                    truncated = (response["finish"] or {}).get("type") == "length"
                    items.append((response["text"], prompt["label"], truncated))
                shape.append(len(prompt["responses"]))
            scores = pool.map(score, items, chunksize=16)
            rows, k = [], 0
            for prompt, n in zip(data["prompts"], shape):
                responses = prompt["responses"]
                rows.append({
                    "source": prompt["source"],
                    "id": prompt.get("id"),
                    "rewards": scores[k : k + n],
                    "truncated": [(r["finish"] or {}).get("type") == "length" for r in responses],
                    "boxed": ["\\boxed" in r["text"] for r in responses],
                    "tokens": [r["tokens"] for r in responses],
                })
                k += n
            name = path.name.removesuffix(".raw.json")
            summary = {"template": name, "seconds": data["meta"]["seconds"]}
            for source in ("train", "math500", "all"):
                subset = [r for r in rows if source in ("all", r["source"])]
                if subset:
                    summary[source] = summarize(subset)
            (root / f"{name}.scored.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
            table.append(summary)
    print(f"{'template':18s} {'src':8s} {'reward':>7s} {'pass@8':>7s} {'mixed':>6s} {'trunc':>6s} {'boxed':>6s} {'mean_tok':>8s} {'med_tok':>7s}")
    for summary in table:
        for source in [k for k in ("train", "math500", "all") if k in summary]:
            m = summary[source]
            print(f"{summary['template']:18s} {source:8s} {m['mean_reward']:7.3f} {m['pass_at_k']:7.3f} {m['mixed_groups']:6.3f} "
                  f"{m['truncated']:6.3f} {m['boxed']:6.3f} {m['mean_tokens']:8.0f} {m['median_tokens']:7.0f}")
    (root / "summary.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
