# Difficulty Sampling

This directory contains tooling for building per-instance difficulty metadata
for RLVR curricula.

## Create A Difficulty Map

Use `create_difficulty_map.py` to build a difficulty map from a Hugging Face
dataset that already contains per-row pass-rate aggregates.

The script expands pass-count summaries into binary attempt outcomes, fits a
Beta prior across binary outcomes, estimates per-item difficulty, and writes
JSONL difficulty files plus schema and metadata sidecars.

### Examples

Write local difficulty files:

```bash
uv run scripts/data/difficulty_sampling/create_difficulty_map.py \
  --hf-dataset mnoukhov/dapo-math-17k-processed-filtered-qwen3-4b-base-32samples \
  --hf-split train \
  --output /tmp/dapo_math_qwen3_difficulty
```

Write local files and push the single output group to the Hub:

```bash
uv run scripts/data/difficulty_sampling/create_difficulty_map.py \
  --hf-dataset mnoukhov/dapo-math-17k-processed-filtered-qwen3-4b-base-32samples \
  --hf-split train \
  --task math \
  --output /tmp/dapo_math_qwen3_difficulty \
  --push-to-hub your-org/dapo-math-qwen3-difficulty \
  --split train
```

Hub uploads require exactly one task/model output group, so use `--task` or a
single-group input dataset when pushing.

## Difficulty Metadata Format

`grpo_fast.py` can optionally replace uniform prompt reshuffling with
`DifficultyCurriculumSampler`, a bucket-aware RLVR curriculum driven by
per-instance difficulty metadata. The current recommended metadata format comes
from the beta-binomial estimator in `create_difficulty_map.py`:

```json
{
  "difficulty": {
    "value": 0.9999999997624719,
    "posterior_mean": 0.003437858035078528,
    "posterior_lower_bound": 2.3752813430506325e-10,
    "expected_quantile": 0.10139684528348392,
    "bucket_index": 4,
    "bucket_count": 5
  }
}
```

- `posterior_mean` is the estimated solve probability for that prompt. Lower
  means harder.
- `bucket_index = 0` is the easiest bucket and
  `bucket_index = bucket_count - 1` is the hardest.
- The sampler uses a smooth distribution with a configurable easy-heavy
  bootstrap phase, then gradually shifts mass toward harder buckets instead of
  hard-switching between discrete phases.
- Within each bucket, examples are weighted by a blend of uncertainty
  (`4 * p * (1 - p)`) and hardness (`1 - p`), so borderline prompts stay
  attractive while already-solved prompts are naturally down-weighted.
- If `--curriculum_adaptive true` is set, bucket probabilities are additionally
  blended with live reward / advantage statistics so buckets with useful
  learning signal can get more mass during training.

## Recommended Starting Point

For `bucket_count=5`:

- Bootstrap (first ~100 steps by default): buckets 0 and 1 dominate so the
  model sees easier prompts while it settles into the chat template and task
  format.
- Early after bootstrap: bucket 2 highest, buckets 1 and 3 nonzero, bucket 4
  low.
- Mid: buckets 2 and 3 dominate, with bucket 4 increasing.
- Late: buckets 3 and 4 dominate, while buckets 0-2 remain nonzero.

Useful flags:

```bash
--curriculum difficulty \
--curriculum_metadata_field difficulty \
--curriculum_bootstrap_steps 100 \
--curriculum_bootstrap_target 0.125 \
--curriculum_warmup_target 0.5 \
--curriculum_final_target 1.0 \
--curriculum_warmup_steps 500 \
--curriculum_total_steps 10000 \
--curriculum_min_hard_frac 0.05 \
--curriculum_max_hard_frac 0.50 \
--curriculum_bucket_sigma 0.0 \
--curriculum_bootstrap_sigma 0.0 \
--curriculum_uncertainty_weight 0.5 \
--curriculum_adaptive true
```

Tuning tips:

- Increase `curriculum_bootstrap_steps` to keep the easy bootstrap around
  longer.
- Lower `curriculum_bootstrap_target` to bias more strongly toward the easiest
  buckets early.
- Lower `curriculum_bucket_sigma` or `curriculum_bootstrap_sigma` to
  concentrate probability on fewer neighboring buckets.
- Lower `curriculum_warmup_target` if you want the post-bootstrap warmup to
  stay easier for longer.

## Metrics

The most useful curriculum metrics are:

- `curriculum/progress`
- `curriculum/static_bucket_prob_*`
- `curriculum/adaptive_bucket_prob_*`
- `curriculum/bucket_prob_*`
- `curriculum/sampled_bucket_count_*`
- `curriculum/bucket_reward_mean_*`
- `curriculum/bucket_abs_advantage_mean_*`

See `scripts/train/qwen/qwen3_4b_dapo_math_difficulty_curriculum.sh` for a
concrete launch example.
