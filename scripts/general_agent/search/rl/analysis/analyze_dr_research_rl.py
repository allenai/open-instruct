#!/usr/bin/env python3
"""One-shot analyzer for a Deep-Research (DR-Tulu) RL run: wandb metrics + trajectory analysis.

Give it just the wandb run URL. It:
  1. Pulls and sanity-checks the wandb metrics (analyze_wandb.py), printing the
     learning / evolving-rubric / stability / tools / infra tables and automated FLAGS
     tuned for a continuous evolving-rubric reward with search/browse tools.
  2. Reads `exp_name`, `rollouts_save_path`, `response_length`, and `max_possible_score`
     from the run's config, then analyzes the saved trajectories (analyze_trajectories.py):
     the reward distribution, the truncated-vs-completed reward gap (budget), per-tool
     usage, and within-group reward spread — reported as a trend over training.

This is the entry point the `analyze-dr-research-rl` skill drives.

Usage:
    uv run python analyze_dr_research_rl.py https://wandb.ai/ai2-llm/oe-general-agents/runs/m8xd8yr4
    uv run python analyze_dr_research_rl.py <url> --per-step-cap 96            # faster trajectory pass
    uv run python analyze_dr_research_rl.py <url> --decode-examples 1          # also show example tails
    uv run python analyze_dr_research_rl.py <url> --skip-trajectories          # wandb only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_trajectories  # noqa: E402
import analyze_wandb  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="wandb run URL or 'entity/project/run_id'")
    ap.add_argument("--rollouts-dir", default=None, help="override; default reads rollouts_save_path from config")
    ap.add_argument("--response-length", type=int, default=None, help="override; default reads from config")
    ap.add_argument("--max-score", type=float, default=None, help="override; default reads max_possible_score")
    ap.add_argument("--per-step-cap", type=int, default=0, help="sample N rollouts/step for a faster pass (0=all)")
    ap.add_argument("--decode-examples", type=int, default=0, help="decode K example tails per reward bucket")
    ap.add_argument("--step", type=int, default=None, help="step for decode examples (default: last)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--skip-trajectories", action="store_true", help="wandb metrics only")
    args = ap.parse_args()

    config = analyze_wandb.run(args.run)

    if args.skip_trajectories:
        return
    exp_name = config.get("exp_name")
    if not exp_name:
        print("\n[trajectories] no exp_name in config; pass --exp-name to analyze_trajectories.py directly.")
        return
    if not config.get("save_traces"):
        print("\n[trajectories] run did not set --save_traces; no rollouts to analyze.")
        return

    rollouts_dir = args.rollouts_dir or config.get("rollouts_save_path") or analyze_trajectories.DEFAULT_ROLLOUTS_DIR
    response_length = (
        args.response_length or config.get("response_length") or analyze_trajectories.DEFAULT_RESPONSE_LENGTH
    )
    max_score = args.max_score or config.get("max_possible_score") or analyze_trajectories.DEFAULT_MAX_SCORE

    print("\n" + "#" * 92)
    print("# TRAJECTORY / REWARD / TOOL ANALYSIS")
    print("#" * 92)
    analyze_trajectories.run(
        exp_name=exp_name,
        rollouts_dir=rollouts_dir,
        response_length=response_length,
        max_score=max_score,
        per_step_cap=args.per_step_cap,
        decode=args.decode_examples,
        step=args.step,
        tokenizer=args.tokenizer,
    )


if __name__ == "__main__":
    main()
