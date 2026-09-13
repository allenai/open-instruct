#!/usr/bin/env python
# Copyright 2024 AllenAI. All rights reserved.
"""CLI: filter synthetic / RLAIF preference JSONL by simple quality signals.

Example:

```bash
python scripts/filter_preference_quality.py \\
  --input output/synthetic_preferences.jsonl \\
  --output output/synthetic_preferences_filtered.jsonl \\
  --stats-output output/preference_filter_stats.json \\
  --drop-identical \\
  --min-score-margin 0.1 \\
  --require-judge-agreement \\
  --judge-label-key judge_labels
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/filter_preference_quality.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_instruct.preference_quality import filter_preference_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input preference JSONL path")
    parser.add_argument("--output", required=True, help="Output filtered JSONL path")
    parser.add_argument("--stats-output", default=None, help="Optional JSON stats path")
    parser.add_argument("--chosen-key", default="chosen")
    parser.add_argument("--rejected-key", default="rejected")
    parser.add_argument("--drop-identical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-score-margin", type=float, default=None)
    parser.add_argument("--chosen-score-key", default="chosen_score")
    parser.add_argument("--rejected-score-key", default="rejected_score")
    parser.add_argument("--judge-label-key", default=None)
    parser.add_argument(
        "--require-judge-agreement",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args(argv)

    stats = filter_preference_jsonl(
        args.input,
        args.output,
        stats_path=args.stats_output,
        chosen_key=args.chosen_key,
        rejected_key=args.rejected_key,
        drop_identical=args.drop_identical,
        min_score_margin=args.min_score_margin,
        chosen_score_key=args.chosen_score_key,
        rejected_score_key=args.rejected_score_key,
        judge_label_key=args.judge_label_key,
        require_judge_agreement=args.require_judge_agreement,
    )
    print(json.dumps(stats.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
