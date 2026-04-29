# GRPO trainer↔vLLM divergence on the OLMo-core path

Status: **open**. Last updated: 2026-04-29.

## TL;DR

On Qwen3-4B-Base DAPO-Math, `val/tis_clipfrac` is ~570× higher when training
with `grpo.py` (OLMo-core trainer) than with `grpo_fast.py` (HF / DeepSpeed
trainer), even though the OLMo-core forward pass is bit-identical to HF's
forward pass under matched FA3 kernels. The divergence is not in the forward
math — it is in **what we feed to the forward**: `forward_for_logprobs` does
not pass packed-sequence document boundaries (`doc_lens` / `max_doc_lens`) to
the OLMo-core `Transformer`. OLMo-core therefore computes cross-document
attention on packed sequences, while vLLM (and HF FA3) computes intra-doc
attention. That mismatch is exactly the kind of small per-token logprob skew
that blows up `tis_clipfrac`.

## Numbers

| Run | Trainer | OLMo-core rev | `val/tis_clipfrac` mean | max |
|---|---|---|---|---|
| `parozgke` | `grpo_fast.py` (HF) | n/a | 5.7e-6 | 3.7e-5 |
| `paoxtmxx` | `grpo.py` (OLMo-core) | `0690fabf` | ~3.5e-3 | — |
| `il33h8fl` | `grpo.py` (OLMo-core) | `61091dba` | 3.2e-3 (279 steps) | 7.8e-3 |

The OLMo-core packed-attention fix at `61091dba` made the *probe* bit-identical
but barely moved end-to-end `tis_clipfrac` (3.5e-3 → 3.2e-3). That is the clue:
the problem is at the call site, not in the kernel.

## What the parity probe proved

`scripts/diagnostics/olmo_core_hf_parity.py` loads Qwen3-4B-Base into both HF
and OLMo-core and compares per-layer hidden states + per-token logprobs on a
single doc and on a packed 2-doc sequence.

With matched FA3 on both sides (`--hf_attn flash_attention_3`, OLMo
`attn_backend=flash_3`):

- **single-doc**: `max|Δlogits| = 0` across all 36 blocks.
- **packed 2-doc, FIXED path** (OLMo gets explicit `doc_lens` + `max_doc_lens`):
  bit-identical to HF, which derives the same intra-doc structure from
  `position_ids` resets via FA3's `cu_seqlens` plumbing.
- **packed 2-doc, BROKEN path** (OLMo gets only `attention_mask=None` +
  `position_ids`): OLMo silently drops both kwargs and runs full cross-doc
  attention. Diverges from HF.

Conclusion: OLMo-core's `Transformer.forward` requires **explicit**
`doc_lens` / `max_doc_lens` to do intra-doc attention on a packed sequence.
Passing only `position_ids` is a no-op for OLMo-core, even though it is
sufficient for HF FA3.

## Where the trainer goes wrong

The shared logprob-recompute helper is `forward_for_logprobs` in
`open_instruct/grpo_utils.py`:

```python
# open_instruct/grpo_utils.py:379
def forward_for_logprobs(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    pad_token_id: int,
    temperature: float,
    return_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = model(
        input_ids=query_responses,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    ...
```

`doc_lens` / `max_doc_lens` are not in the signature and not in the call. The
OLMo-core trainer hits this from two places in
`open_instruct/olmo_core_train_modules.py`:

- **L391** — `compute_logprobs(self.model, data_BT, ...)` for `old_logprobs`
  when `num_mini_batches > 1`.
- **L433** — `forward_for_logprobs(self.model, data_BT.query_responses[i],
  data_BT.attention_masks[i], data_BT.position_ids[i], ...)` for the
  `new_logprobs` computed every step.

`CollatedBatchData` itself does not even carry doc boundaries:

```python
# open_instruct/data_types.py:104
class CollatedBatchData:
    query_responses: list[torch.Tensor]
    attention_masks: list[torch.Tensor]
    position_ids: list[torch.Tensor]
    advantages: list[torch.Tensor]
    response_masks: list[torch.Tensor]
    vllm_logprobs: list[torch.Tensor]
```

So the doc-boundary information is lost upstream of the trainer entirely. We
build it for the data path (`pack_sequences` in `rl_utils.py`,
`reset_position_ids` for the per-doc `position_ids`), but we never thread it
into the train batch.

This is exactly the BROKEN path from the parity probe — and the trainer is
running it in production with `pack_length=10240`, where most packed
sequences hold multiple docs.

## Why HF doesn't have this problem

`grpo_fast.py` calls the same `forward_for_logprobs`. It works because, in
`transformers >= 5.0` with `attn_implementation="flash_attention_3"`, HF's
attention layer detects position-id resets and constructs intra-doc
`cu_seqlens` for FA3 *automatically*. No explicit `doc_lens` needed.

OLMo-core's `Transformer` does not do that detection. It only honors intra-doc
attention if you pass `doc_lens` / `max_doc_lens` directly.

So the same call path produces:

- **HF FA3**: intra-doc attention (matches vLLM) → tis_clipfrac ≈ 6e-6
- **OLMo-core flash_3**: cross-doc attention (does NOT match vLLM) → tis_clipfrac ≈ 3e-3

## Why the kernel-level fix barely moved the needle

OLMo-core `61091dba` fixed the *kernel* path so that, when given doc_lens, it
correctly handles the varlen flash path. But our trainer still does not give
it doc_lens, so the fix is invisible at the call site we actually use.

The probe was run with explicit `doc_lens` and confirmed equality. That was
necessary but not sufficient — necessary so we know the upstream kernel
isn't broken, but not sufficient because the consumer (`grpo.py`) still
calls the model the wrong way.

## Other suspects, ranked

1. **Missing doc_lens in forward_for_logprobs** — primary suspect, see above.
   Predicted impact: most of the 570× gap. Falsifiable by the fix below.
2. **FSDP-sharded forward ≠ unsharded forward.** The probe ran unsharded on
   a single GPU; production runs `fsdp_shard_degree=4` in bf16. All-gather
   ordering can introduce O(1e-3) drift. Plausible residual after fix #1
   lands; not plausible as the dominant term given the magnitude.
3. **Weight-sync OLMo-core → vLLM.** HF→vLLM is well-trodden; OLMo-core →
   HF-shape-buffer → vLLM is a different transform path. parozgke does not
   exercise this. Worth checking with a one-step parity (sync, then dump
   trainer + vLLM logprobs on the same packed batch).
4. **bf16 master vs fp32 cast.** DeepSpeed keeps fp32 master; OLMo-core's
   optim wrapping may not. If recompute pulls bf16 params after a step that
   updated bf16 directly, drift accumulates differently. Small effect.

## Proposed fix

Two-part change, smallest scope first:

1. **Plumb doc boundaries through `CollatedBatchData`.** Add
   `doc_lens: list[torch.Tensor] | None` and `max_doc_lens: list[int] | None`
   (or compute them from `attention_masks` on the fly in the collator —
   `attention_masks` is already a per-doc-index integer mask, which is what
   `reset_position_ids` consumes).
2. **Pass them through `forward_for_logprobs` to OLMo-core** when the model
   is an OLMo-core `Transformer`. HF doesn't accept those kwargs, so gate on
   model type or pass via `**extra_kwargs` and let HF's wrapper drop them.

Concretely, in `open_instruct/grpo_utils.py:389`:

```python
output = model(
    input_ids=query_responses,
    attention_mask=attention_mask,
    position_ids=position_ids,
    **extra_kwargs,  # {"doc_lens": ..., "max_doc_lens": ...} for OLMo-core
)
```

Update both call sites in `olmo_core_train_modules.py` (L391, L433) to pass
the new fields. `grpo_fast.py` keeps calling without them — HF derives
intra-doc structure from `position_ids`.

## Verification plan

Before merging:

1. Re-run `scripts/diagnostics/olmo_core_hf_parity.py` against the patched
   `forward_for_logprobs` (i.e. test the actual function path, not a probe
   that reaches into the model directly). Confirm packed BROKEN path now
   matches FIXED path.
2. Launch a short DAPO smoke (≤100 steps) with the fix on jupiter; expect
   `val/tis_clipfrac` mean to drop by ≥100× toward the parozgke baseline.
3. If a residual remains > ~1e-4, drill into FSDP-sharded vs unsharded
   forward and the OLMo-core→vLLM weight sync (suspects #2 and #3).

## References

- match-grpo.md "Bug 5" — original silent-kwarg-drop write-up.
- `scripts/diagnostics/olmo_core_hf_parity.py` — parity probe.
- `scripts/diagnostics/olmo_core_packed_parity.py` — packed-only probe.
- Reference run (HF, clean): https://wandb.ai/ai2-llm/open_instruct_internal/runs/parozgke
- Current run (OLMo-core, broken): https://wandb.ai/ai2-llm/open_instruct_internal/runs/il33h8fl
- Beaker: https://beaker.org/ex/01KQDB3FJ2STHP6GCEENY8K7CJ
