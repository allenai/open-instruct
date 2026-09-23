# Ascend NPU SFT

Ascend NPU supervised fine-tuning is supported through `open_instruct/finetune.py` with Hugging Face Accelerate. The NPU path has been exercised with single-card training, two-card HCCL training, DeepSpeed ZeRO-3, LoRA, gradient checkpointing, checkpoint resume, model reload, and padding-free packing.

## Environment

Install a CANN, PyTorch, and `torch_npu` combination listed in the [Ascend PyTorch compatibility documentation](https://github.com/Ascend/pytorch#release-notes). Source the matching CANN environment before importing `torch_npu`:

```bash
source /path/to/Ascend/cann/set_env.sh
python -c 'import torch, torch_npu; assert torch.npu.is_available()'
```

The validated smoke environment used Python 3.12, CANN 9.1.0-beta.3, PyTorch 2.10.0, `torch_npu` 2.10.0.post2, and DeepSpeed 0.19.6 on Ascend 910B3. These versions describe the observed environment; use the official compatibility matrix rather than treating them as universal pins.

The repository's default `uv sync` sources CUDA PyTorch and CUDA attention packages. For NPU, start from a compatible NPU environment, install the remaining `pyproject.toml` dependencies normally, then overlay the NPU pins and this checkout without dependency resolution (after sourcing CANN) — the default set would otherwise resolve CUDA-flavored vLLM wheels and overwrite the NPU stack:

```bash
python -m pip install --no-deps -r requirements_npu.txt
python -m pip install -e . --no-deps
```

`requirements_npu.txt` pins only the packages that must match the vLLM-Ascend stack or provably break on it (numpy: CANN op execution fails under numpy>=2 on this stack); everything else resolves from `pyproject.toml`. The validated environment is described by the versions above rather than a lock file.

Do not install CUDA FlashAttention, bitsandbytes, or CUDA vLLM wheels into the NPU environment.

## Single-Card SFT

Set local model and parquet paths, then run a bounded smoke before increasing sequence length or training duration:

```bash
export MODEL_PATH=/path/to/model
export DATA_PATH=/path/to/train.parquet
export OUTPUT_DIR=/path/to/output
export ASCEND_RT_VISIBLE_DEVICES=0

python open_instruct/finetune.py \
  --model_name_or_path "$MODEL_PATH" \
  --tokenizer_name "$MODEL_PATH" \
  --max_seq_length 512 \
  --per_device_train_batch_size 1 \
  --max_train_steps 1 \
  --logging_steps 1 \
  --dataset_mixer_list "$DATA_PATH" 16 \
  --dataset_mixer_list_splits train \
  --chat_template_name tulu \
  --output_dir "$OUTPUT_DIR" \
  --do_not_randomize_output_dir true \
  --push_to_hub false \
  --try_launch_beaker_eval_jobs false \
  --try_auto_save_to_beaker false
```

Local model directories are used directly and are not passed to Hugging Face Hub validation or download APIs.

## Multi-Card And ZeRO-3

Accelerate selects `MULTI_NPU` and HCCL when `torch_npu` is active:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1

accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
  open_instruct/finetune.py \
  ...
```

For DeepSpeed ZeRO-3, use the existing Accelerate configuration:

```bash
accelerate launch --num_processes 2 --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
  open_instruct/finetune.py \
  ...
```

## Sequence Parallelism

Ulysses sequence parallelism must run through Accelerate's DeepSpeed path. A plain `--multi_gpu` launch only creates data-parallel workers and does not install the Ulysses dataloader adapter:

```bash
accelerate launch --num_processes 2 --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
  open_instruct/finetune.py \
  --sequence_parallel_size 2 \
  ...
```

The weighted loss aggregation uses DeepSpeed's Ulysses process group on NPU (Accelerate's `torch_device_mesh` SP group is not wired up the same way there).

Known limitation: sequence parallelism cannot currently run end to end on this stack. This is not an NPU-specific gap — it matches upstream GPU behavior under the same DeepSpeed version:

- Without `--packing`, DeepSpeed >= 0.19.0 raises `ValueError` because `UlyssesSPDataLoaderAdapter.refill` requires `position_ids` in every dataloader batch, and upstream `finetune.py`'s non-packing collator does not emit them. The upstream SP run in PR #1539 predates this check (DeepSpeed 0.18.x tolerated a missing `position_ids`); with `pyproject.toml`'s open `deepspeed>=0.18.3` resolving to 0.19.x today, the same launch fails identically on CUDA.
- With `--packing`, the padding-free collator emits 1-D flattened tensors while the adapter assumes 2-D `[batch, seq]` batches (`IndexError`), on any device and any DeepSpeed version.

Fixing the non-packing path (inject per-sample `arange` `position_ids` in the SP collator) is planned as a separate upstream PR, so this branch stays behavior-aligned with `main`.

## Packing

`--packing` remains padding-free with FlashAttention. When the selected attention implementation is SDPA, `finetune.py` constructs an explicit block-diagonal causal mask so tokens cannot attend across packed sample boundaries. This fallback is correct but uses O(L²) mask memory; validate memory at the intended sequence length before a long run.

## Current Boundaries

- `open_instruct/olmo_core_finetune.py` is not NPU-enabled. The current OLMo-core dependency selects CPU when CUDA is unavailable and contains CUDA/NCCL-specific training, checkpoint, RNG, and memory paths.
- QLoRA/bitsandbytes, Liger kernels, and CUDA FlashAttention are not covered by the NPU SFT path.
- The current evidence covers bounded single-node smokes. It does not establish long-run convergence, multi-node stability, or performance parity with CUDA.
