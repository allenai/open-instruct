# H083 fp32 identity test

Ledger-manager's ruling (2026-10-23), fixed before the result.
- **Context:** the bf16 gradient comparison (`../gradcmp`) measured 5.81e-3. This test asks whether that is bf16 rounding of
  a reordered backward pass, or a change of math by the big-slice layer's KDA sharding pins (maxtext `c5f1251d`).
- **Setup:** same digests, placement (`olmo35-ep-as-cp` fsdp16 x ep8), init, 1-step SGD read-out and `compare.py` as
  `../gradcmp`.
- **Rule:** rel ≤ 1e-5 in fp32 means class B: the pins are value identities. rel > 1e-5 means a math change: stop and report.

## fp32 settings
- `dtype=float32` for activations; weights and gradients were already fp32.
- `matmul_precision=highest`. It exists in both images (base.yml, untouched by either layer), and MoE and KDA read it
  (`moe.py` weight sum and dense/einsum paths, `olmoe3.py`). `kda_matmul_precision=highest` was already set.
- `JAX_DEFAULT_MATMUL_PRECISION=highest`, for ops called with no precision argument. That includes `jax.lax.ragged_dot` in
  `moe.py` `jax_ragged_dot_gmm`.
- `megablox=false`, so the MoE runs through `jax.lax.ragged_dot`.

## bf16 casts and precision limits found

| where | what | handled |
|---|---|---|
| megablox Pallas gmm (`pallas_mosaic_tpu_v2_gmm_kernel.py:1376`) | `del precision`: the precision argument is ignored; `ops.py` passes `Precision.DEFAULT` | **cannot be overridden**, so megablox is turned off |
| `moe.py` `jax_ragged_dot_gmm` | `jax.lax.ragged_dot` without `precision=` | covered by `JAX_DEFAULT_MATMUL_PRECISION=highest` |
| `olmoe3.py:353` KDA chunk `compute_dtype` | bf16 only if `gdn_state_dtype=bfloat16` | default `float32`, kept |
| `olmoe3.py` `astype(x.dtype)` (conv weights, o_norm) | follows the activation dtype | fp32 under `dtype=float32` |
| `embeddings.py:190` `attend_on_embedding` | hard `astype(bfloat16)` of the table | not used: `olmo35-tiny` has an untied head (`logits_via_embedding: false`) |
| megablox gmm `acc_dtype` bf16 | only with input quantization | not used (no quantization; megablox off) |

**Not verified:** whether XLA's TPU ragged-dot lowering honours HIGHEST for fp32 operands. If the result sits above 1e-5,
check this first.

## VMEM
Attempt 1, at the bf16 runs' `--xla_tpu_scoped_vmem_limit_kib=16384`, failed to compile at sequence 65536 and 32768 alike:
`Used 17.09M of 16.00M vmem`. The `-v24` overlays raise the limit to 24576 KiB in both images. It is a compiler budget,
the same for both, and the read is still image vs image. Approved by ledger-manager.

Attempt 2 (`-v24`) failed the same way, still against 16.00M. That is the v4 core's physical VMEM, which the flag cannot
raise. The overflowing op is the TPU ragged-dot kernel's backward pass (`transpose(jvp(...))/ragged_dot_general`) with
fp32 windows at the 512 x 1024 forward tiles. Attempt 3 (`-v24t`) halves those tiles
(`w{i,o}_tile_fwd_{batch_seq,embed_dim,mlp_dim}=256/512/512`) in both images. This is an execution setting: it changes how
the kernel blocks its accumulation, the same way in both images, and not the math being compared.

## Shape
Tried in order (the `-v24` overlays): 16 x 65536, then 16 x 32768, then 16 x 16384 (`SEQ_LEN`, both images alike) if HBM runs out. A VMEM overflow stops the ladder instead. The one used
is in `RECEIPT.md` and the event line.

## Run and receipt
- **Run:** `run.sh` on Hammond, from `/weka/oe-eval-default/abhishekr/watch/tpu/gc32/`. It is idempotent across restarts.
- **Receipt:** `RECEIPT.md`, holding:
  - the bf16 and fp32 results side by side, whole model and worst tensors;
  - `result.json`, `checks.txt` (batch-1 hashes, step-1 loss), `run.log` and `logs/`.
- **Event line:** `GRAD-CMP-FP32 ...` in the x16prod event log.
