# Preference quality filtering (RLAIF)

Synthetic preference pipelines (LLM-as-judge / RLAIF) often emit noisy pairs:
identical completions, tiny score margins, or multi-judge disagreements. Feeding
those pairs straight into reward modeling or DPO can amplify positional bias and
reward hacking.

Open Instruct now ships a small, dependency-light filter for this step:

- Python API: `open_instruct.preference_quality`
- CLI: `scripts/filter_preference_quality.py`

## Where it fits

```text
generation.py  ->  synthetic_preference_dataset.py (LLM judge)
               ->  filter_preference_quality.py      <-- here
               ->  reward_modeling.py / dpo.py / GRPO prefs
```

See also [Synthetic preference dataset](synthetic_preference_dataset.md) and
[Grouped Relative Policy Optimization (GRPO)](grpo.md).

## What it does

| Filter | Flag / kwarg | Effect |
|--------|--------------|--------|
| Identical pairs | `--drop-identical` / `drop_identical` | Drop when chosen and rejected assistant texts match |
| Score margin | `--min-score-margin` / `min_score_margin` | Drop when \|chosen_score − rejected_score\| is below the threshold (rows without scores are kept) |
| Judge agreement | `--require-judge-agreement` + `--judge-label-key` | Drop when a list of judge labels is not unanimous |

The filter also reports retention rate and a simple length-bias ratio
(`mean(chosen_len) / mean(rejected_len)`), which is useful when auditing whether
the judge systematically prefers longer answers.

## CLI example

```bash
python scripts/filter_preference_quality.py \
  --input output/synthetic_preferences.jsonl \
  --output output/synthetic_preferences_filtered.jsonl \
  --stats-output output/preference_filter_stats.json \
  --drop-identical \
  --min-score-margin 0.1
```

## Python example

```python
from open_instruct.preference_quality import filter_preference_rows

kept, stats = filter_preference_rows(
    rows,
    drop_identical=True,
    min_score_margin=0.1,
    judge_label_key="judge_labels",
    require_judge_agreement=True,
)
print(stats.to_dict())
```

## Shuffle un-mapping helper

`should_swap_after_shuffle(preferred, shuffled_index)` encodes the correct
post-judge unshuffle for pairwise prompts that randomize response order to
reduce positional bias. This is used by
`open_instruct/rejection_sampling/synthetic_preference_dataset.py`.

## Related research direction

Disagreement-aware or curriculum-style preference filtering is a natural
extension: keep easy/high-margin pairs first, then introduce harder /
disagreement-weighted examples. The helpers here are intentionally small so they
can be composed into such curricula without changing the training loops.
