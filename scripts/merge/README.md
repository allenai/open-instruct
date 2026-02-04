# Model Merging Guide

Merge HuggingFace models on Beaker using either [mergekit](https://github.com/arcee-ai/mergekit) or direct safetensors averaging.

## Files

| File | Description |
|------|-------------|
| `mergekit_merge.sh` | Beaker launcher using mergekit (for supported architectures) |
| `direct_merge.sh` | Beaker launcher using direct safetensors averaging (for all architectures) |
| `direct_merge.py` | Python script for direct safetensors averaging (used by `direct_merge.sh`) |
| `launch_merges.sh` | Example batch script that launches multiple merge jobs |

## Which Script to Use

- **`mergekit_merge.sh`**: Works with standard HuggingFace architectures (e.g., Llama, OLMo3). Supports multiple merge methods (`linear`, `slerp`, `ties`, `dare_ties`).
- **`direct_merge.sh`**: Works with any architecture, including hybrid models (e.g., `Olmo3_5HybridForCausalLM`) that mergekit doesn't support. Only does linear (weighted average) merging.

## Usage

Both scripts take the same arguments:

```bash
./scripts/merge/<script>.sh <output_dir> <model1> <model2> [model3] ...
```

Models can be local paths or HuggingFace model IDs (mergekit only).

### Mergekit

```bash
./scripts/merge/mergekit_merge.sh \
    /weka/oe-adapt-default/nathanl/merged/my-merge \
    /weka/path/to/model1 \
    /weka/path/to/model2
```

### Direct Merge

```bash
./scripts/merge/direct_merge.sh \
    /weka/oe-adapt-default/nathanl/merged/my-merge \
    /weka/path/to/model1 \
    /weka/path/to/model2
```

## How It Works

Both scripts use base64-encoding to safely pass configs/scripts through the shell layers (bash -> mason.py -> Beaker), avoiding YAML escaping issues. On Beaker, the encoded string is decoded and executed.

- `mergekit_merge.sh`: Base64-encodes a YAML config, installs mergekit on Beaker, and runs `mergekit-yaml`
- `direct_merge.sh`: Base64-encodes `direct_merge.py`, sends it to Beaker, and runs it directly

Both scripts copy tokenizer files and `chat_template.jinja` from the first model.

### Custom Transformers

For models that require a custom transformers build (e.g., new OLMo architectures not yet in released transformers), `mergekit_merge.sh` has commented-out variables for installing from a PR:

```bash
# In mergekit_merge.sh - uncomment and set PR number as needed
# CUSTOM_INSTALL="uv pip install git+https://github.com/huggingface/transformers.git@refs/pull/XXXXX/head"
# INSTALL_CMD="${CUSTOM_INSTALL} && uv pip install mergekit"
```

## Examples

### Mergekit with HuggingFace models

[Test run](https://beaker.org/ex/01KGK2AWYGFY5E6R0M3F94NN49)

```bash
./scripts/merge/mergekit_merge.sh \
    /weka/oe-adapt-default/nathanl/merged/test-mergekit \
    allenai/Olmo-3-7B-Think-DPO \
    allenai/Olmo-3-7B-Think-SFT
```

### Mergekit with local weka checkpoints

[Test run](https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01KGK1QXQ7XQHMC0NPKK5HG52A)

```bash
./scripts/merge/mergekit_merge.sh \
    /weka/oe-adapt-default/nathanl/merged/test-mergekit-olmo3 \
    /weka/oe-adapt-default/saumyam/checkpoints/olmo2-7B-sft/rl-sft/olmo3-32b-SFT-1e-4/step10500-hf \
    /weka/oe-adapt-default/saumyam/checkpoints/olmo2-7B-sft/rl-sft/olmo3-32b-SFT-1e-4/step9000-hf
```

### Direct merge for hybrid models

[Test run](https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01KGK1BBTG0Q4MFHMK6TDJNH9X)

```bash
# 2-model merge
./scripts/merge/direct_merge.sh \
    /weka/oe-adapt-default/nathanl/merged/sft-2model-linear \
    /weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR2.5e-5/step46412-hf \
    /weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR4.5e-5_seed42/step46412-hf

# 3-model merge
./scripts/merge/direct_merge.sh \
    /weka/oe-adapt-default/nathanl/merged/sft-3model-linear \
    /weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR2.5e-5/step46412-hf \
    /weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR4.5e-5_seed42/step46412-hf \
    /weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR1e-5/step46412-hf
```

## Notes

- **MergeKit** doesn't support hybrid models (e.g., `Olmo3_5HybridForCausalLM`) because it expects uniform layer structure. Use `direct_merge.sh` for those.
- Weights are normalized by default, so `1.0 + 1.0 + 1.0` gives equal (33.3%) weight to each model
- For different ratios, use different weights (e.g., `2.0 + 1.0` = 66.7% / 33.3%)
