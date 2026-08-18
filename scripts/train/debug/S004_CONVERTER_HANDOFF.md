# Handoff: HF converter + vLLM plugin for OLMoE3 s004 (MoE + GDN hybrid)

Goal: make `OLMoE3-dev-260614-s004` **evaluable**. Training already works
(see below); the result cannot be scored because there is no HF export path and
no vLLM architecture for it.

## Read this first: the lesson that matters most

On the previous model in this series (Olmo-Hybrid-7B), **three CPU-side checks
passed green on a badly broken converter**: 523/523 key-and-shape match, config
regeneration, and a state-dict round trip. The bug — HF hardcodes `eps=1e-5` on
the gated-DeltaNet output norm while the config says `1e-6` — produced a **38%
logit error** with 70% argmax agreement. Training on it would have looked
completely healthy.

**So: build the numerical comparison harness BEFORE writing conversion code.**
`scripts/train/debug/check_olmo_core_matches_hf.py` (116 lines) is the working
example: it runs a forward pass through both implementations on the same input,
compares logits, and bisects per-layer to localise a mismatch. Port it first,
then write the converter against it. Key/shape checks are necessary and nowhere
near sufficient.

Related trap from the same effort: `save_pretrained` wrote *legacy* weight
names, which passed every state-dict test and then failed inside vLLM with
"weights were not initialized". Verify the **written files**, not just the
in-memory dict.

## The template to copy

`~/repos/scaling-ladders/ladders/mainline/` (Yashas's; he evaluates base models
with it regularly) solves exactly this problem for a *dense* peri-norm GDN
hybrid. Reuse aggressively:

| file | lines | relevance |
|---|---|---|
| `transformers_plugin/src/transformers_plugin/modeling_mainline_ladder.py` | 1109 | peri-norm blocks, NoPE, GDN, gated attention |
| `transformers_plugin/src/transformers_plugin/convert_mainline_ladder_weights_to_hf.py` | 761 | weight mapping shape |
| `transformers_plugin/src/transformers_plugin/configuration_mainline_ladder.py` | 178 | `model_type = "mainline_ladder"` |
| `vllm_plugin/src/vllm_plugin/mainline_ladder.py` | 548 | registers the arch on **stock vLLM** via a `vllm.general_plugins` entry point — no vLLM fork |

**What it does NOT have: MoE.** Grepping the whole plugin for
`expert|moe|router|num_experts|top_k` returns nothing. Routed experts, the
shared expert, the router, and MoE forward are the new work.

Ask Yashas whether to extend `mainline_ladder` or add a new architecture, and
whether a MoE variant exists elsewhere — that answer changes the estimate a lot.

## Why olmo-core's own converter won't do it

`olmo_core/nn/hf/convert.py:708` (`_require_no_peri_ln`) raises
`NotImplementedError` for peri-LN, deliberately: peri-LN maps the same olmo-core
norm parameter to different HF layernorms depending on whether a layer is dense
or MoE, which its conversion framework cannot express uniformly. s004 sets
`use_peri_norm: true` on every block type, so both
`convert_olmo3moe_state_{to,from}_hf` refuse it.

Note the flag is spelled `use_peri_norm` in olmo-core configs and `use_peri_ln`
in HF configs. Grep for both.

## The model

`gs://ai2-llm/checkpoints/olmo3moe/OLMoE3-dev-260614-s004_1536d2048a_30L1536M1536S_128E8K1S_gdn/step69000`
(also copied to `/weka/oe-adapt-default/abhishekr/s004/step69000`; 320 GB).
Jobs can read `gs://` directly — see `scripts/train/debug/probe_gcs.sh`.

- 26,688,158,976 params total, ~2.9B active. 30 layers, `d_model` 1536, vocab 100352.
- **Three block types**, `_CLASS_ = olmo_core.nn.ddp.block.OLMoDDPTransformerBlockConfig`:
  - `gdn_dense` (layers 0–1): GDN mixer + shared expert only (`hidden_size` 13824)
  - `gdn_moe` (28 of the remaining): GDN mixer + 128 routed experts + 1 shared (`hidden_size` 1536)
  - `attn_moe` (6, every 5th): full attention (`n_heads` 16, `n_kv_heads` 8, `head_dim` 128, gated) + same MoE
- Router: top-8 of 128, softmax, `normalize_expert_weights: 1.0`,
  `lb_loss_weight: 0.015`, `z_loss_weight: 1e-4`, `_CLASS_ = MoERouterConfigV2`.
- `use_peri_norm: true`, `use_pre_norm: false` on all three. Norm eps `1e-6`
  everywhere in config — **but check the HF reference for hardcoded values**
  (see the 1e-5 trap above).
- No RoPE on the attention layers (NoPE), like the hybrid.
- Native context 8192 (`dataset.sequence_length`). s004 is **pretrain-only** —
  no `-midtrain` or `-long-context` sibling exists, unlike s002/s003.

The checkpoint's own `config.json` needs migrating before current olmo-core will
load it: `scripts/train/debug/migrate_s004_config.py` applies 9 lossless
rewrites (it asserts losslessness rather than assuming it) and the result is
committed at `scripts/train/debug/s004_migrated.json`.

## Training status (works — do not redo)

Branch `spike/s004-moe`, olmo-core pinned to `8f508637b57e` (`akshitab/moe-v2-core`).
172 steps on one 8-GPU B300 node (`ai2/holmes`): exit 0, CE 1.315 → 0.823,
11,490 tok/s/device, expert load imbalance improving (mean 1.87 → 1.40 across 28
MoE blocks). Beaker `01M0941GK4GMHNV16B1ZSEYAMW`; checkpoints under
`deletable_checkpoint_states/mo3eoo1s`.

Constraints discovered the hard way, all committed on the branch:
- `OLMoDDPModel` **refuses FSDP2** (`prepare_experts_for_fsdp` raises) — it
  trains through olmo-core's `nn.ddp` train module.
- `OLMoDDPTrainModule` **rejects async checkpointing** — `--no_save_async`.
- MoE v2 permutation **requires transformer_engine**; the import is guarded but
  the call is not, so a missing TE surfaces as
  `TypeError: 'NoneType' object is not callable` deep in the first forward.
  TE must be **2.16.1** (≥2.17's `ep.cpp` unconditionally includes a header
  torch 2.10 does not ship), and the CUDA 13 base must be `13.3.1-cudnn-devel`
  (13.0.3's libcublasLt lacks a symbol TE's wheel needs).
- `throughput/device/TPS` as logged by this train module is **unreliable** —
  it reported 19, 26 and 733.7 in one run. Compute from step timestamps.

## Suggested order

1. Port `check_olmo_core_matches_hf.py` to s004 and get it running against the
   olmo-core model alone (so the harness is trusted before it judges anything).
2. Config + modeling for the dense (`gdn_dense`) layers only; verify numerically.
3. Add `attn_moe` attention; verify.
4. Add routed experts + router; verify. This is the bulk.
5. Weight conversion both directions; verify the **written** safetensors, not
   just the state dict.
6. vLLM plugin; confirm olmo-eval can serve it and that scores are non-zero —
   a missing chat template scores 0.0000 while reporting `Success`.
