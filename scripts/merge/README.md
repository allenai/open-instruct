# Model Merging Guide

This guide covers how to merge HuggingFace models using [mergekit](https://github.com/arcee-ai/mergekit) and evaluate the merged models.

## Setup

### 1. Clone mergekit

```bash
cd /path/to/open-instruct
git clone https://github.com/arcee-ai/mergekit.git
```

The `mergekit/` directory is gitignored, so it won't be committed.

### 2. Install mergekit

Follow the [mergekit installation instructions](https://github.com/arcee-ai/mergekit#installation).

**Note:** If you're merging models that use custom transformers (e.g., OLMo), you may need to install the custom transformers library first:

```bash
# Example for OLMo models (adjust as needed)
pip install ai2-olmo
```

## Creating a Merge Config

Create a YAML config file that specifies which models to merge and how:

```yaml
# Example: configs/merge_config.yaml
models:
  - model: /weka/path/to/model/1
    parameters:
      weight: 1.0
  - model: /weka/path/to/model/2
    parameters:
      weight: 1.0
merge_method: linear
dtype: float16
```

### Config Options

- **models**: List of models to merge with their weights
- **weight**: Relative weight for each model (normalized by default, so 1.0 + 1.0 = 50/50 merge)
- **merge_method**: Algorithm to use (`linear`, `slerp`, `ties`, `dare_ties`, etc.)
- **dtype**: Output precision (`float16`, `bfloat16`, `float32`)

### Example: Merging 3 Checkpoints (Equal Weights)

```yaml
models:
  - model: /weka/oe-adapt-default/nathanl/checkpoints/MODEL_A/step46412-hf
    parameters:
      weight: 1.0
  - model: /weka/oe-adapt-default/nathanl/checkpoints/MODEL_B/step46412-hf
    parameters:
      weight: 1.0
  - model: /weka/oe-adapt-default/nathanl/checkpoints/MODEL_C/step46412-hf
    parameters:
      weight: 1.0
merge_method: linear
dtype: float16
```

## Running the Merge

```bash
mergekit-yaml /path/to/config.yaml ./output-dir
```

### Important: Tokenizer Handling

The tokenizer may not be copied automatically. After merging, verify the tokenizer exists in the output directory:

```bash
ls ./output-dir/tokenizer*
```

If missing, copy it from one of the source models:

```bash
cp /weka/path/to/source/model/tokenizer* ./output-dir/
cp /weka/path/to/source/model/special_tokens_map.json ./output-dir/
cp /weka/path/to/source/model/tokenizer_config.json ./output-dir/
```

## Evaluating the Merged Model

After merging, evaluate using the standard eval pipeline:

```bash
uv run scripts/submit_eval_jobs.py \
    --model_name "merged-model-name" \
    --location "/path/to/merged/model" \
    --cluster ai2/jupiter ai2/ceres \
    --is_tuned \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --use_hf_tokenizer_template \
    --beaker_image yanhongl/oe_eval_olmo3_devel_v5 \
    --oe_eval_tasks "gpqa:0shot_cot::qwen3-instruct,minerva_math_500::hamish_zs_reasoning_deepseek" \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --run_id placeholder \
    --oe_eval_max_length 32768 \
    --process_output r1_style \
    --skip_oi_evals
```

## Full Workflow Example

See `scripts/merge/merge_and_eval.sh` for an automated script that:
1. Creates a merge config
2. Runs the merge
3. Copies the tokenizer if needed
4. Submits evaluation jobs

## Notes

- **Linear merging** has not been extensively tested with linear (non-transformer) models
- Weights are normalized by default, so `1.0 + 1.0 + 1.0` gives equal (33.3%) weight to each model
- For different ratios, use different weights (e.g., `2.0 + 1.0` = 66.7% / 33.3%)
