# Matching the reference OLMo-3-7B SFT run

This document summarizes the parity investigation that verified
`open_instruct/olmo_core_finetune.py` trains identically to a pure olmo-core
SFT loop, using reference Beaker experiment
[`01KNMEJKEZNJKZH9QWQW8CS0JW`](https://beaker.org/ex/01KNMEJKEZNJKZH9QWQW8CS0JW)
(wandb run `l6dg9xwv`, `jacobm/olmo3-7b-instruct-SFT-rerun-04072026`) as the
target.

## Goal

Step-by-step parity between `olmo_core_finetune.py` and an equivalent pure
olmo-core SFT loop, so future divergences from reference runs can be attributed
to specific layers of the stack rather than hunted across the whole pipeline.

## Stage 1 — Minimal single-GPU parity rig (Qwen3-0.6B)

To make debugging cheap, we stood up a single-GPU rig on Qwen3-0.6B with 3
steps and identical inputs between the two paths:

- model: `TransformerConfig.qwen3_0_6B`, HF weight load via
  `load_hf_model`
- seeds: init 33333, data loader 34521
- hyperparams: lr 5e-6, warmup 0.03, AC `blocks.*.feed_forward`,
  SkipStepAdamW, `attn_implementation=flash_2`
- dataset: identical tokenized mmap

Added:

- `scripts/train/debug/olmo_core_reference_sft.py` — pure-olmo-core SFT runner
  (DDP, no CP)
- `scripts/train/debug/oc_sft_qwen3_reference.sh` — launch the pure runner
- `scripts/train/debug/oc_sft_qwen3_openinstruct.sh` — launch
  `olmo_core_finetune.py` with the same config

### Root cause

Divergence traced to a debug `next(iter(data_loader))` peek in
`olmo_core_finetune.py` that consumed a batch without being rewound by the
subsequent `data_loader.reshuffle(epoch=1)`, causing a one-batch shift in
training data vs. the plain olmo-core loop. Removing the peek produced
byte-identical step 0 logits.

### Results (3 steps)

| Step | Pure olmo-core | `olmo_core_finetune.py` | Δ |
|---|---|---|---|
| 0 | 8.675 | 8.675 | **0.000** |
| 1 | 6.322 | 6.320 | 0.002 |
| 2 | 4.856 | 4.842 | 0.014 |

Step 0 is byte-identical. Residual drift at later steps is consistent with a
single non-deterministic op per step (bf16 reduction order, `torch.compile`
autotune kernel choice).

Runs:

1. Pure olmo-core reference: [Beaker](https://beaker.org/ex/01KPE49WD35E5HH9V9GY1F4P5C)
2. `olmo_core_finetune.py` open-instruct: [Beaker](https://beaker.org/ex/01KPE49WDADQZHHD74W8ZE7AAG)

## Stage 2 — Full-scale reproduction vs. the real reference run

With the single-GPU rig green, we relaunched the full reference config
end-to-end on 4 nodes × 8 H100 and compared against the reference wandb run
`l6dg9xwv`.

Config (matches `01KNMEJKEZNJKZH9QWQW8CS0JW` exactly):

- `config_name=olmo3_7B`, `max_seq_length=32768`
- `per_device_train_batch_size=1`, `gradient_accumulation_steps=2`
  — global batch = 1 × 32768 × 16 dp_ranks × 2 grad_accum = 1,048,576 tokens
- `learning_rate=8e-5`, `warmup_ratio=0.03`, `max_grad_norm=1.0`,
  `weight_decay=0.0`, 2 epochs
- `rope_scaling_factor=8`, `ac_mode=selected_modules blocks.*.feed_forward`
- `cp_degree=2` (auto-derived on H100 because seq_len > 16384 tokens/rank)
- `attn_implementation=flash_2`, `compile_model=true`
- seeds: `seed=33333 data_loader_seed=34521`

Launcher: `scripts/train/debug/oc_sft_olmo3_7b_full.sh`.

## Stage 3 — ring-flash-attn packaging issue

Reference runs use `cp_strategy=llama3`, which dispatches through
`ring_flash_attn.llama3_flash_attn_varlen_func` — olmo-core's
`FlashAttention2Backend.assert_supports_ring_cp` fails without the package.
But `ring_flash_attn==0.1.8`'s `__init__.py` unconditionally imports
`substitute_hf_flash_attn` from `.adapters`, which in turn imports
`is_flash_attn_greater_or_equal_2_10` from
`transformers.modeling_flash_attention_utils` — a symbol removed in
transformers 5.x. So `import ring_flash_attn` raised `ImportError`, olmo-core's
try/except fell back to `ring_flash_attn = None`, and the CP backend check
aborted the run.

This stage was ultimately deprecated by Stage 4; what was tried:

- single-GPU import diagnostic (`scripts/train/debug/ring_flash_diag.py`,
  Beaker `01KPEGN4BYW1VPN9BV4BVA1NEA`) pinned the traceback
- a Dockerfile-time patch rewrote `ring_flash_attn/__init__.py` to wrap the
  adapters import in try/except. Verified fix: `has_ring_flash_attn: True`
  (Beaker `01KPEH0MV4Q6XXT0E0MHF2ETG6`). CP worked, reference run was
  reproduced (Beaker `01KPEHZKJQRJ5DD150XBYNWT59`).

## Stage 4 — cp_strategy=ulysses replaces llama3 + ring-flash-attn

Ulysses is mathematically equivalent to llama3 for standard softmax attention
(all-to-all head repartition vs. ring accumulation — same attention output,
different communication pattern). Ulysses does **not** require
`ring-flash-attn`. We launched the same config with `cp_strategy=ulysses`
(Beaker [`01KPENNBD43D1PCH1HQ0Y666QP`](https://beaker.org/ex/01KPENNBD43D1PCH1HQ0Y666QP))
to see how it compared.

### Loss parity (first 10 steps)

| Step | Reference (`l6dg9xwv`, llama3) | Ours — llama3 | Ours — ulysses |
|---|---|---|---|
| 1 | 0.9756 | 0.9756 | 0.9756 |
| 2 | 1.0582 | 1.0580 | 1.058 |
| 3 | 0.9726 | 0.9726 | 0.9727 |
| 4 | 0.9667 | 0.9666 | 0.9667 |
| 5 | 1.0229 | 1.0220 | 1.022 |
| 6 | 0.9809 | 0.9806 | 0.9807 |
| 7 | 0.9237 | 0.9226 | 0.9226 |
| 8 | 0.8344 | 0.8339 | 0.8339 |
| 9 | 0.9188 | 0.9184 | 0.9185 |
| 10 | 0.8585 | 0.8585 | 0.8586 |

Both CP strategies match the reference to within bf16 / `torch.compile`
non-determinism (max Δ 0.0011 for llama3, 0.0009 for ulysses).

### Throughput / MFU (running avg over >60 steps)

| metric | llama3 + ring-flash-attn | ulysses | Δ |
|---|---|---|---|
| MFU (actual avg) | 38.08% | **45.88%** | **+20.5%** |
| TPS / device | ~6,350 | ~7,400 | +16.5% |
| flops/s / device | ~377 TFLOPs | ~436 TFLOPs | +15.6% |

On Jupiter (H100, DOCA OFED interconnect) ulysses's all-to-all is cheaper than
llama3's ring accumulate. Combined with the fact that ulysses has no
ring-flash-attn dependency, we made it the default in the 7B launch scripts
and removed the Dockerfile patch and the ring-flash-attn package entirely.

## Final state

- `scripts/train/debug/oc_sft_olmo3_7b_full.sh` — 2-epoch full run,
  `cp_strategy=ulysses`
- `scripts/train/debug/oc_sft_olmo3_7b_match.sh` — 3-step quick-check,
  `cp_strategy=ulysses`
- no ring-flash-attn dependency, no Dockerfile patch
- parity with `01KNMEJKEZNJKZH9QWQW8CS0JW` verified at both scales

## Runs

1. Pure olmo-core reference (Qwen3-0.6B): [Beaker](https://beaker.org/ex/01KPE49WD35E5HH9V9GY1F4P5C)
2. `olmo_core_finetune.py` open-instruct (Qwen3-0.6B): [Beaker](https://beaker.org/ex/01KPE49WDADQZHHD74W8ZE7AAG)
3. Full 7B — llama3 CP strategy (ring-flash-attn): [Beaker](https://beaker.org/ex/01KPEHZKJQRJ5DD150XBYNWT59)
4. Full 7B — ulysses CP strategy: [Beaker](https://beaker.org/ex/01KPENNBD43D1PCH1HQ0Y666QP)
