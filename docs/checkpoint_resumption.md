# Checkpoint Resumption Reference

This document covers how checkpoint saving and resumption works across the three training paradigms: SFT (`finetune.py`), DPO (`dpo.py`), and RL/GRPO (`grpo_fast.py`). The mechanisms differ significantly between them.

---

## Quick Comparison

| | SFT | DPO | GRPO |
|---|---|---|---|
| **Auto-resume from latest** | Yes — from `output_dir` | Yes — OLMo-core Trainer handles it | No — must provide `--checkpoint_state_dir` explicitly |
| **Explicit checkpoint arg** | `--resume_from_checkpoint` | `--resume_from_checkpoint` (overrides auto) | `--checkpoint_state_dir` (doubles as save and load path) |
| **Where training state is saved** | `output_dir/step_N/` | `output_dir/` (OLMo-core format) | `checkpoint_state_dir/global_stepN/` (DeepSpeed format) |
| **Where final model is saved** | `output_dir/` | `output_dir/` | `output_dir/` (separate from training state) |
| **Checkpoint format** | Accelerate (`step_N/` or `epoch_N/`) | OLMo-core native | DeepSpeed ZeRO (`global_stepN/`) |
| **Model weights saved** | ✓ | ✓ | ✓ |
| **Optimizer state saved** | ✓ | ✓ | ✓ |
| **LR scheduler saved** | ✓ | ✓ | ✓ |
| **RNG state saved** | ✓ (implicit via Accelerate) | ✓ | ✓ (explicit: CPU, CUDA, NumPy, Python) |
| **Dataloader position saved** | ✓ (via `skip_first_batches`) | ✓ (OLMo-core data loader) | ✓ (data prep actor state) |
| **Reference policy saved** | N/A | N/A | ✓ (separate `ref_policy/` subdir) |

---

## SFT (`finetune.py`)

### How Auto-Resume Works

On startup, `get_last_checkpoint_path()` checks `output_dir`. If it contains any checkpoint subdirectories (named `step_N` or `epoch_N`) that have a `COMPLETED` marker file, the script automatically resumes from the one with the highest step/epoch number. No flag is needed — just point to the same `output_dir`.

A `COMPLETED` file is written after the checkpoint is fully flushed to disk, so partial/incomplete checkpoints are never loaded.

```
output_dir/
  step_500/
    COMPLETED       ← only checkpoints with this file are considered
    optimizer.bin
    scheduler.bin
    rng_state.pth
    model/
  step_1000/
    COMPLETED
    ...
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--output_dir` | `output/` | Where checkpoints are written and where auto-resume looks |
| `--resume_from_checkpoint` | `None` | Explicit path to a checkpoint directory; overrides auto-resume |
| `--checkpointing_steps` | `None` | Save every N steps, or `"epoch"`. No saving if unset. |
| `--keep_last_n_checkpoints` | `3` | Delete older checkpoints; `-1` keeps all |

### Dataloader Resume

The step number is extracted from the checkpoint directory name (e.g. `step_500` → 500). The script computes how many batches to skip within the current epoch using `accelerator.skip_first_batches()`. If you resume from `step_500` with a batch size and dataset that implies 500 steps = 2.5 epochs, the dataloader fast-forwards to the correct position in epoch 3.

### Resuming From a Specific Old Checkpoint

```bash
python scripts/update_command_args.py scripts/train/... \
    --resume_from_checkpoint /path/to/output_dir/step_500 | uv run bash
```

---

## DPO (`dpo.py`)

### How Auto-Resume Works

DPO uses OLMo-core's `Trainer`, which takes `save_folder=args.output_dir`. OLMo-core's `build()` automatically detects any existing checkpoints in `save_folder` and resumes from the latest one. There is no explicit load call in `dpo.py` — the Trainer manages discovery and loading internally.

```python
trainer = train.TrainerConfig(
    save_folder=args.output_dir,
    ...
    save_overwrite=True,    # keeps only recent checkpoints
).build(train_module, data_loader)
```

`save_overwrite=True` means OLMo-core manages retention automatically — it does not keep unbounded history.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--output_dir` | `output/` | Where OLMo-core saves and auto-resumes from |
| `--resume_from_checkpoint` | `None` | Override to resume from a specific path |
| `--checkpointing_steps` | `500` | Save every N steps |
| `--ephemeral_save_interval` | `None` | Save a temporary "ephemeral" checkpoint more frequently (e.g. every 50 steps); gets overwritten by the next ephemeral save. Useful for crash recovery without keeping many full checkpoints. Must be less than `checkpointing_steps`. |
| `--keep_last_n_checkpoints` | `3` | How many non-ephemeral checkpoints to retain |

### Resuming From a Specific Old Checkpoint

```bash
python scripts/update_command_args.py scripts/train/... \
    --resume_from_checkpoint /path/to/output_dir/step50-of-100 | uv run bash
```

---

## GRPO (`grpo_fast.py`)

### Key Distinction: Two Separate Directories

GRPO separates training state from the final model output:

- **`--checkpoint_state_dir`** — training state checkpoint, in DeepSpeed ZeRO format. Contains model weights, optimizer states, scheduler, RNG states, reference policy, and data pipeline position. This is what you point to when resuming.
- **`--output_dir`** — the final HuggingFace-format model export. Written at the end of training (or at `--save_freq` intervals). Not used for resumption.

### How Resume Works

There is **no automatic discovery**. If `--checkpoint_state_dir` is provided and the directory exists, the Learner calls DeepSpeed's `load_checkpoint()`, which automatically finds the latest valid checkpoint within that directory (using DeepSpeed's `latest` tag file). The `training_step` value in the saved state is used to fast-forward the data pipeline and optimizer step counter.

If `--checkpoint_state_dir` is provided but the directory does not exist, a warning is logged and training starts from scratch.

```
checkpoint_state_dir/
  global_step200/             ← DeepSpeed checkpoint
    zero_pp_rank_0_mp_rank_00_model_states.pt
    zero_pp_rank_0_mp_rank_00_optim_states.pt
    ...
  global_step400/
    ...
  ref_policy/
    pytorch_model.bin         ← reference policy weights (separate from DeepSpeed checkpoint)
  latest                      ← DeepSpeed tag: points to most recent global_stepN
```

### What Gets Saved

- Model weights (DeepSpeed ZeRO sharded)
- Optimizer states (DeepSpeed ZeRO sharded)
- LR scheduler state
- RNG states: CPU, NumPy, Python, and per-GPU CUDA states
- `training_step`: the step counter, used to resume the data pipeline
- Reference policy weights (in `ref_policy/pytorch_model.bin`, outside the DeepSpeed checkpoint)
- Data prep actor state (position in the streaming data pipeline)

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint_state_dir` | `None` | Path to save/load training state. If it exists on startup, training resumes from it. **Auto-set by mason** to a unique timestamped path unless you provide a `/weka/` path. |
| `--checkpoint_state_freq` | `200` | Save a checkpoint every N training steps |
| `--keep_last_n_checkpoints` | `3` | Number of DeepSpeed checkpoints to retain in `checkpoint_state_dir` |
| `--gs_checkpoint_state_dir` | `None` | GCS path to sync checkpoints to after each save. Set automatically when `--gs_bucket_path` is provided. |
| `--gs_bucket_path` | `None` | GCS bucket for backup. Auto-derives `gs_checkpoint_state_dir` and prepends `/filestore` to local path. |
| `--deepspeed_checkpoint_load_universal` | `False` | Load checkpoint across different parallelism configs (e.g. different DeepSpeed stage or tensor parallel size). Required when resuming from a checkpoint saved with a different setup. |
| `--save_freq` | `40` | Save an HF-format model to `output_dir` every N training steps |
| `--output_dir` | `None` | Where HF-format model exports go. Not used for resumption. |

### Mason's Auto-Override of `--checkpoint_state_dir`

When you submit via mason without specifying `--checkpoint_state_dir`, mason auto-generates:

```
/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/<whoami>/<timestamp>_<rand>/
```

This means every fresh `mason.py` invocation creates a new checkpoint directory — there is no automatic continuity between runs unless you explicitly pass the same `--checkpoint_state_dir` from the previous run.

**To resume a previous run:**

1. Find the checkpoint directory from the previous run (logged by mason or visible in W&B config).
2. Pass it explicitly to `--checkpoint_state_dir`:

```bash
python scripts/update_command_args.py scripts/train/dr-tulu/rl_qwen35_4b_drtulu.sh \
    --checkpoint_state_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/shashankg/1234567890_42/ \
    | uv run bash
```

Mason will then see that the path already starts with `/weka/` and will **not** override it, leaving your explicit path intact.

### W&B Resumption

Mason sets `WANDB_RUN_ID` and `WANDB_RESUME=allow` for GRPO jobs. The run ID is a random 8-character string generated once when mason starts. On a Beaker retry, the same run ID is reused, so W&B seamlessly continues the same run.

However, if you manually re-submit with a new mason invocation, a new run ID is generated — the W&B run will appear as a new run. To continue the same W&B run across manual re-submissions, you would need to set `WANDB_RUN_ID` manually to match the previous run.

---

## Common Gotchas

### Changing Dataset Args Between Runs
All three scripts reconstruct the dataloader position from the saved step count. If you change the dataset composition (different datasets, sample counts, or transform functions) between a save and a resume, the step-to-data mapping will be off — the model will see different data than expected at that step. This is not detected automatically.

### Changing Parallelism Config (GRPO only)
If you change `--deepspeed_stage`, `--vllm_tensor_parallel_size`, or sequence parallelism between a save and a resume, the checkpoint will fail to load due to shape mismatches in the sharded optimizer states. Set `--deepspeed_checkpoint_load_universal True` to attempt cross-config loading, but this is not guaranteed to work in all cases.

### Changing Optimizer Type (GRPO / DPO)
If you switch optimizer type between runs (e.g. from AdamW to Muon), the saved optimizer states will not match the new optimizer and loading will fail.

### SFT: `output_dir` Already Exists
If you run SFT with an `output_dir` that already contains checkpoints from a prior unrelated run, the script will automatically resume from those checkpoints. Delete or rename the directory if you want a clean start.
