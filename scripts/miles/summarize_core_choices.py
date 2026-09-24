"""Summarize retained hero Core/SGLang probe artifacts with correct greedy ties."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root, *, experiment, result_dataset):
    first = json.loads((root / "emo" / "auto.json").read_text())
    summary = {
        "experiment": experiment,
        "result_dataset": result_dataset,
        "provenance": json.loads((root / "provenance.json").read_text()),
        "analysis_note": "Cached choices use actual output_ids. The original probe logged a top-k representative that could break ties differently; those raw CHOICES counts are not used. Forced choice counts concern the reported representative, not a separately generated token.",
        "runtime_lock": json.loads((root / "runtime.lock.json").read_text()),
        "workload": {
            "domains": [r["domain"] for r in first["rollouts"]],
            "prompts": len(first["rollouts"]),
            "tokens_per_prompt": [len(r["output_ids"]) for r in first["rollouts"]],
            "concurrency": 1,
            "sampling": "greedy, ignore_eos=true",
            "warmup": "one equal-length request for each prompt per engine",
        },
        "arms": {},
    }
    for arm in ["emo", "non-emo"]:
        core = json.loads((root / arm / "core.json").read_text())
        armout = {"model": core["model"], "modes": {}}
        for mode, records in core["comparisons"].items():
            serving = json.loads((root / arm / f"{mode}.json").read_text())
            modeout = {
                "engine_args": serving["engine_args"],
                "load_seconds": serving["load_seconds"],
                "cached": {},
                "forced": {},
            }
            duration = sum(r["seconds"] for r in serving["rollouts"])
            count = sum(len(r["output_ids"]) for r in serving["rollouts"])
            modeout["timing"] = {
                "seconds": duration,
                "tokens": count,
                "tokens_per_second": count / duration,
                "per_prompt_seconds": [r["seconds"] for r in serving["rollouts"]],
            }
            for kind, label in [("rollouts", "cached"), ("forced", "forced")]:
                for scorer in ["full_sequence", "prefix_at_a_time"]:
                    delta = []
                    mismatch = []
                    choices = 0
                    sampled_ties = 0
                    shape_flips = 0
                    for r, s in zip(records[kind], serving[kind], strict=True):
                        c = r[scorer]["core"]
                        delta.extend(a - b for a, b in zip(s["logprobs"], c["logprobs"], strict=True))
                        shape_flips += sum(
                            a != b
                            for a, b in zip(
                                r["full_sequence"]["core"]["argmax_ids"],
                                r["prefix_at_a_time"]["core"]["argmax_ids"],
                                strict=True,
                            )
                        )
                        for i, (a, b) in enumerate(
                            zip(
                                (s["output_ids"] if kind == "rollouts" else s["argmax_ids"]),
                                c["argmax_ids"],
                                strict=True,
                            )
                        ):
                            choices += 1
                            if kind == "rollouts" and s["output_ids"][i] != max(s["top2"][i], key=lambda v: v[0])[1]:
                                sampled_ties += 1
                            if a != b:
                                mismatch.append(
                                    {
                                        "row": r["row"],
                                        "domain": r["domain"],
                                        "offset": i,
                                        "sg_token": a,
                                        "core_token": b,
                                        "core_margin": c["top2_margin"][i],
                                        "core_trajectory_token_rank": c["chosen_rank"][i],
                                        "core_trajectory_token_margin": c["chosen_margin"][i],
                                        "sg_top2": s["top2"][i],
                                        "sg_top2_tied": s["top2"][i][0][0] == s["top2"][i][1][0],
                                    }
                                )
                    delta = np.array(delta)
                    absolute = np.abs(delta)
                    ratios = np.exp(-delta)
                    modeout[label][scorer] = {
                        "positions": choices,
                        "argmax_matches": choices - len(mismatch),
                        "argmax_agreement": 1 - len(mismatch) / choices,
                        "mean_abs_logprob": float(absolute.mean()),
                        "max_abs_logprob": float(absolute.max()),
                        "p95_abs_logprob": float(np.quantile(absolute, 0.95)),
                        "ratio_outside_20pct_fraction": float(((ratios < 0.8) | (ratios > 1.2)).mean()),
                        "core_shape_argmax_changes": shape_flips,
                        "greedy_token_vs_reported_top1_changes": sampled_ties,
                        "disagreements": mismatch,
                        "choice_definition": "actual greedy token versus Core argmax"
                        if kind == "rollouts"
                        else "reported top-1 representative versus Core argmax; ties are ambiguous",
                        "strict_core_preference_disagreements": sum(
                            m["core_trajectory_token_rank"] > 1 for m in mismatch
                        )
                        if kind == "rollouts"
                        else None,
                    }
            armout["modes"][mode] = modeout
        a = json.loads((root / arm / "auto.json").read_text())["rollouts"]
        b = json.loads((root / arm / "full.json").read_text())["rollouts"]
        armout["cross_mode_greedy"] = [
            {
                "domain": x["domain"],
                "first_different_offset": next(
                    (i for i, (u, v) in enumerate(zip(x["output_ids"], y["output_ids"], strict=True)) if u != v), None
                ),
            }
            for x, y in zip(a, b, strict=True)
        ]
        summary["arms"][arm] = armout
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--result-dataset", required=True)
    args = parser.parse_args()
    report = summarize(args.results, experiment=args.experiment, result_dataset=args.result_dataset)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
