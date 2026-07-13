# general_agent utils

Shared helper scripts for the general-agent SFT/RL workflows.

## `convert_sft_epoch_checkpoints.sh` — load & serve SFT epoch checkpoints

When an SFT run (`open_instruct/finetune.py`) is launched with `--checkpointing_steps epoch`
(as all our terminal/search SFT scripts now are), each epoch is saved via
`accelerator.save_state()` into

```
<CKPT_ROOT>/epoch_N/
  pytorch_model/          # DeepSpeed ZeRO-3 sharded model + optimizer state
  zero_to_fp32.py         # bundled consolidation helper
  ...                     # NO config.json / tokenizer / *.safetensors
```

This is a **training-state** checkpoint, not a model dir — you cannot
`from_pretrained()` it directly. This script consolidates each `epoch_N/` into an
HF-loadable model dir.

> **SFT only.** Terminal/GRPO RL (`grpo_fast.py`) does not need this. Its `--save_freq`
> checkpoints are written with `save_pretrained` (config + safetensors + tokenizer) and
> load/serve directly — they can go straight into CG-conversion. This script keys off the
> SFT `epoch_*/pytorch_model/` layout, which the RL path never produces.

### Convert

```bash
scripts/general_agent/utils/convert_sft_epoch_checkpoints.sh CKPT_ROOT [CONFIG_SRC] [OUT_ROOT]
```

- `CKPT_ROOT` — the run dir containing `epoch_*/` (e.g. a `deletable_checkpoint/<user>/<exp_name>`).
- `CONFIG_SRC` — where to copy `config.json` + `generation_config.json` + tokenizer from.
  Defaults to the base model `Qwen/Qwen3.5-9B`. **Prefer the run's final `output_dir`** once
  the run completes — it carries the exact config + chat template used.
- `OUT_ROOT` — output dir (defaults to `<CKPT_ROOT>/hf`).

For each epoch it (1) runs the bundled `zero_to_fp32.py` to consolidate the ZeRO shards to
`model.safetensors`, then (2) attaches config + tokenizer from `CONFIG_SRC`. Idempotent
(skips already-converted epochs). CPU-only; ~40–70 GB RAM and ~36 GB disk per epoch (fp32 9B).

### Load (HF transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
path = "<OUT_ROOT>/epoch_2"
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
```

### Serve (vLLM)

- **Qwen3-8B and other `*ForCausalLM` runs:** the converted dir serves in vLLM as-is.
- **Qwen3.5-9B (and Qwen3.5 family):** the SFT config is
  `architectures=[Qwen3_5ForCausalLM]` / `model_type=qwen3_5_text`, which **no vLLM version
  registers** (only `Qwen3_5ForConditionalGeneration`). vLLM errors with
  `Model architectures ['Qwen3_5ForCausalLM'] are not supported`. Run the (lossless)
  CG-conversion first, then serve the `_cg` output:

  ```bash
  python agent/scripts/convert_qwen35_causallm_to_cg.py <epoch_dir> <epoch_dir>_cg
  ```

  (`--language-model-only` does **not** help here.) See memory
  `reference_qwen35_causallm_to_cg_conversion` / `project_drtulu_qwen35_eval_toolformat`.
