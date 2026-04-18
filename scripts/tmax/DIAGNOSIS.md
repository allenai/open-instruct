# Qwen3.5 hybrid: vLLM ↔ HF logprob drift — findings

Working notes from investigating why Qwen3.5 RL training shows larger logprob
divergence (and TIS < 1) than dense models.

## Quick facts on the architecture

`Qwen/Qwen3.5-4B` is hybrid:
- 3 of every 4 decoder layers are **linear attention** (`Qwen3_5GatedDeltaNet`) —
  a gated-DeltaNet with a recurrent state of shape `[H_v, D_k, D_v]`.
- The 4th is standard full attention with gated output.
- `linear_num_value_heads=32`, `linear_num_key_heads=16`, `linear_key_head_dim=128`,
  `linear_value_head_dim=128`, `linear_conv_kernel_dim=4`.

Weights in vLLM are prefixed `language_model.` (it is registered as a VLM).

## What the two backends are actually doing

| Phase | vLLM path | HF (training) path |
|---|---|---|
| prefill (prompt) | `chunk_gated_delta_rule` with `cu_seqlens` | n/a (training doesn't prefill separately) |
| decode (each new token) | `fused_recurrent_gated_delta_rule` with `ssm_state` kept in-place | n/a |
| scoring rollouts | n/a | `chunk_gated_delta_rule` with `cu_seqlens` (via the packing patch) |

So the logprob vLLM emits for a sampled token comes almost entirely from the
**recurrent** kernel, while the logprob HF recomputes comes from the **chunk**
kernel. They are mathematically the same operation, but bf16 numerics differ
and the state-accumulation order is different.

## Measured baseline (single 80GB A100, bf16, seed=42, no prefix cache)

### Single-GPU, single-turn

| model | tokens | mean \|Δlogp\| | max \|Δlogp\| |
|---|---:|---:|---:|
| Qwen3.5-4B hybrid, 512 | 512 | 0.0123 | 0.21 |
| Qwen3-4B dense, 512 | 512 | 0.0149 | 0.21 |
| Qwen3.5-4B hybrid, 2048 | 2048 | 0.0148 | 0.21 |
| Qwen3.5-4B hybrid, 4060 | 4060 | 0.0181 | 0.21 |
| Qwen3.5-4B hybrid, **16000** | 16000 | 0.0124 | 0.24 |
| Qwen3.5-9B hybrid, 4096 | 4096 | 0.0142 | 0.37 |
| Qwen3.5-9B hybrid, **16000** | 16000 | 0.0124 | 0.24 |
| Qwen3.5-4B, pack left 256 **with patch** | 512 | 0.0136 | 0.14 |
| Qwen3.5-4B, pack left 256 **no patch** | 512 | **0.129** | **3.53** |

**Key finding:** Drift does NOT grow linearly with sequence length. It plateaus at
~0.012-0.014 beyond ~512 tokens. Testing 9B at 16k confirms the same floor as 4B.

### Multi-turn (simulated tool-use)

**4B, 2-turn, 256+256 tokens:**

| scenario | turn-1 | tool | turn-2 | overall |
|---|---:|---:|---:|---:|
| 2-turn, 256+256 tokens | 0.0060 | masked | 0.0134 | 0.0097 |

**9B, 10-turn, 2000 assistant + 1000 tool tokens per turn (~30k response, context up to 27k):**

| turn | ctx start | n tokens | mean\|Δlogp\| | max\|Δlogp\| |
|---:|---:|---:|---:|---:|
| 1 | 57 | 2000 | 0.00799 | 0.172 |
| 2 | 3057 | 2000 | 0.00009 | 0.010 |
| 3 | 6057 | 2000 | 0.00005 | 0.001 |
| 4–6 | 9k–15k | 2000 | ≤0.00004 | ≤0.001 |
| 7 | 18057 | 2000 | 0.00015 | 0.125 |
| 8 | 21057 | 2000 | 0.00263 | 0.224 |
| 9 | 24057 | 2000 | 0.00001 | 0.001 |
| 10 | 27057 | 2000 | 0.00193 | 0.352 |
| **overall** | — | **20000** | **0.00129** | **0.352** |

Key findings: drift does NOT grow monotonically with turn depth or context length.
The per-turn means are dominated by how "confident" the model is about the specific
sampled tokens (high-confidence tokens → near-zero diff regardless of depth).
Overall mean across 30k tokens of 10-turn tool-use is 0.00129 — **10× lower** than
the 0.014 single-turn floor.

### SP=2 (9B, 4k tokens, per-batch FLACPContext fix)

| test | mean \|Δlogp\| | max \|Δlogp\| |
|---|---:|---:|
| Single GPU (no SP) | 0.0142 | 0.37 |
| SP=2, packing (pack_pad_left=256), per-batch fix | 0.0140 | 0.23 |
| SP=2, no packing (single sequence), per-batch fix | 0.0140 | 0.25 |
| SP=2, packing, **static context (bug)** | 0.0161 | **2.51** |

The per-batch FLACPContext fix keeps SP=2 identical to single-GPU. The static context
bug is benign at 4k (mean only slightly elevated) but causes large max|diff| and the
window mean grows toward 0.03 near the SP boundary — much worse at longer sequences.

### Conclusion on offline drift

After applying all fixes (packing patch, per-batch CP context, prefix cache disabled),
the offline kernel mismatch floor is at most **~0.012–0.014** (single-turn, any length
up to 16k, SP=2) and as low as **~0.001** in realistic multi-turn tool-use at 30k
tokens. Drift does NOT grow with sequence length or turn depth.

**The 0.07 in training definitively cannot be explained by kernel mismatch alone** —
even at 30k multi-turn context the offline floor is 50× lower. The most likely
additional factor is **async model staleness**: with `--async_steps 4`, by the time
training computes `local_logprobs` (with W_N), the stored `vllm_logprobs` may be from
W_{N-k} for k ≥ 1. Even one gradient step of divergence on a Qwen3.5 hybrid model
appears sufficient to explain the gap. Check `model_step_mean` in WandB to confirm —
if it lags the training step by more than 1, async divergence is the cause.

If the metric is 0.07 **at step 0** (before any gradient update), then staleness is
ruled out too. In that case check the `GIT_COMMIT` env var in WandB — the running
Docker image may predate the per-batch FLACPContext fix (commit `6a8c0390`) or the
prefix-cache disable (`c7acb6ee`).

Alternatively, if the Docker image used was built from code before the per-batch
FLACPContext fix (`6a8c0390`) or prefix-cache disable (`c7acb6ee`), those bugs would
still apply and inflate the metric. Verify by checking the `GIT_COMMIT` tag in WandB.

## Suspects for the RL-training-specific drift

Ordered from most likely to have meaningful effect down to nice-to-haves.

### (1) Ulysses SP / FLACPContext is built once, globally, from rank number

`grpo_fast.py` constructs `FLACPContext` once at init:

```python
self.cp_context = FLACPContext(
    group=sp_group,
    is_first_rank=(sp_rank == 0),
    is_last_rank=(sp_rank == sp_world_size - 1),
    pre_num_ranks=sp_rank,
    post_num_ranks=sp_world_size - sp_rank - 1,
    conv1d_kernel_size=conv_kernel_size,
)
```

Then per-batch, `_compute_packing_kwargs` only updates `cu_seqlens` /
`cu_seqlens_cpu` on that same object — the rank metadata is frozen.

This is the proper construction **only** if there is a single sub-sequence
spanning all SP ranks. In open-instruct we pack multiple rollouts into one
`query_response` row (up to `--pack_length 35840`), and `UlyssesSPSplitter`
slices that row into `chunk_len = max_seqlen / sp_world_size`. A sub-sequence
boundary can land anywhere inside a rank's chunk.

Consequences for linear-attention state flow:
- For a sub-sequence that is fully inside rank R, FLA will still expect it to
  receive prior state from R-1 (because `is_first_rank=False, pre_num_ranks=R`).
  That's wrong — it should start from zero state.
- For a sub-sequence that ends inside rank R, FLA will still try to push its
  state to R+1. Also wrong.

As the model trains, `A_log`, `dt_bias`, and the conv1d weights evolve; the
spurious state that is passed becomes progressively less compatible with
what a "clean" forward would produce. That matches the user's observation
that TIS drops over training.

The fix is to build the CP context per-batch from the **global** cu_seqlens
using the supported API (see `fla/ops/cp/README.md`):

```python
from fla.ops.cp import build_cp_context

cp_context = build_cp_context(
    cu_seqlens=global_cu_seqlens,   # before Ulysses sharding
    group=sp_group,
    conv1d_kernel_size=conv_kernel_size,
    cu_seqlens_cpu=global_cu_seqlens.cpu(),
)
```

`build_cp_context → get_cp_cu_seqlens` derives per-rank `pre_num_ranks`,
`is_first_rank`, `local cu_seqlens`, `pre_num_conv_tokens`, etc. correctly,
including for sub-sequences that do not cross rank boundaries. The
README makes the key guarantee explicit: *"only the first sequence in
the local batch can be a continuation from a previous rank — all other
sequences start fresh"*.

### What this branch changes

- `UlyssesSPSplitter.split_collated_batch` now also stashes the *unsharded*
  `position_ids` per sample onto the batch as `global_position_ids`.
- `CollatedBatchData` has a new optional field `global_position_ids:
  list[torch.Tensor] | None`.
- `grpo_utils.build_fla_cp_context_for_sample` constructs the correct
  `FLACPContext` per sample via `fla.ops.cp.build_cp_context`, synthesising
  a trailing sub-sequence for right-padding so total tokens match
  `local_seq_len * sp_world_size` (FLA's part_len).
- `grpo_utils.compute_logprobs` accepts a new `cp_contexts: list | None`
  argument — one per-sample context rather than a single shared one.
- `grpo_fast.PolicyTrainerRayProcess` no longer pre-builds a static
  `FLACPContext`; it stores `_sp_rank/_sp_group/_sp_world_size/_conv_kernel_size`
  and calls `_build_cp_contexts_for_batch` each step, one context per sample.

Fresh-built contexts already carry rank-local `cu_seqlens`, so
`_compute_packing_kwargs` no longer mutates the context.

For `sequence_parallel_size == 1`, non-hybrid models, or when
`global_position_ids` is absent, every entry in `cp_contexts` is `None`
and the code path is unchanged. Sanity check: single-sequence hybrid
forward produced the same mean-|Δ| (0.01364) before and after the patch.

### (2) vLLM prefix caching keeps an SSM state across turns

`--vllm_enable_prefix_caching` is on in all `scripts/tmax/*.sh`. For standard
KV cache this is fine. For linear-attention layers it is subtler: vLLM stores
one SSM state per cached prefix slot, produced by `fused_recurrent_gated_delta_rule`
(decode) or `chunk_gated_delta_rule` (prefill). If the rollout has multiple
tool turns, the turn-2 prefill loads the SSM state from the end of turn-1's
decode — which was not computed with the same kernel HF will use to re-score.

Recommendation: warn/disable prefix caching when the config has any
`linear_attention` layer types, or document that you need to re-score with
`fused_recurrent_gated_delta_rule` applied token-at-a-time to truly match.
I'd lean toward disabling it for Qwen3.5 until (1) is fixed and the resulting
drift is re-measured.

### (3) Baseline bf16 kernel drift (unavoidable)

`chunk_gated_delta_rule` pypi-fla vs vllm-fla differ by ~0.001 relative, and
both match a torch fp32 reference to ~bf16. That's not the problem by itself,
but it sets a floor of ~1% mean logprob diff per token in bf16 — the same
floor dense models have.

The cheapest way to lower this floor is to reduce the chunk kernel's internal
accumulation — e.g. `chunk_size=32` instead of the hardcoded 64 inside
`fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule_fwd`. Worth trying as
a second pass after (1) is in.

## Script

`scripts/tmax/diagnose_logprobs.py` runs as two processes so each phase has
full GPU memory:

```bash
# Generate with vLLM and capture per-token logprobs
python scripts/tmax/diagnose_logprobs.py --phase vllm \
    --model Qwen/Qwen3.5-4B --max_new_tokens 2048 \
    --npz /tmp/q35.npz

# Score with HF (patched) and report drift
python scripts/tmax/diagnose_logprobs.py --phase hf \
    --model Qwen/Qwen3.5-4B --apply_patch 1 \
    --npz /tmp/q35.npz --label q35_patched
```

`--pack_pad_left N` packs N dummy tokens before the target to exercise the
packing-patch path. Comparing `--apply_patch 1` vs `--apply_patch 0` in that
mode is the quickest sanity check that the patch is still doing what it
should.
