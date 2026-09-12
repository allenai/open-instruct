# Sequence packing qualification, 2026-09-12

Feature branches: `robertb/miles-sequence-packing` in Open Instruct and Core.
Open Instruct base is `016318180`; Core base is `cfc42934d` on
`robertb/miles-rl-adapter`. Primary branches have not been changed.

## Numerical gate: passed

[Passing Beaker run](https://beaker.org/ex/01M2ABT2D0A5J9YX6F4C62WDNH), exit 0,
Open Instruct `ec71db9b24373516c1a8a22008ea8993981342d9`, Core
`3d35ab326` (full pin in `runtime/miles/runtime.lock.json`). Two Holmes GPUs,
about 9m22s execution, urgent priority and 1h minimum runtime.

Six `packing-ep*-rank*.json` reports contain all rank results for EP1/EP2,
recomputation on/off. Each compares unpacked policy-only, packed policy-only,
and packed policy+auxiliary arms on identical inputs, two updates each.
The fixture has KDA, RoPE/full attention, latent MoE, fixed replay and masked
response tokens. `replay-summary.json` summarizes all 18 rank/arm contract logs.

- All packed/unpacked scoring comparisons and document-isolation perturbations
  had mean and maximum absolute difference **0** on this fixture.
- Policy-only first-step gradient relative L2 error was at most about 0.20%
  for EP1; after independently updating, the second step reached 1.98%.
  EP2's maximum across gradient and final Adam-state groups was 0.0052%.
  These pass the preset 5% group-relative tolerance. Gradients are not bit-exact;
  changing batch shape changes floating-point accumulation. No tolerance was relaxed.
- Eight samples became three EP1 forwards. Four local samples became two
  EP2 forwards on both ranks, including a deliberately unequal initial pack count.
- Zero replay mismatches in all contracts; gradients remained enabled in training,
  recomputation re-entered the routed layers, and scoring modes were checked/skipped.
- Combined native auxiliary loss was finite and produced updates. It intentionally
  balances each packed token batch rather than preserving per-response auxiliary
  gradients. It is not an objective-equivalence claim for auxiliary loss.
- Warm second-update wall times on the toy policy-only fixture: EP1 0.254→0.165s
  without recomputation and 0.333→0.198s with it; EP2 rank 0 0.219→0.198s and
  0.294→0.197s. One tiny observation per arm is not a production speed estimate.

[First attempt](https://beaker.org/ex/01M2AAR9MJ3Z5613XWG9WG3PPJ), Open Instruct
`c75c2ba70422`, failed at the first packed full-attention call. The pinned FA4 API
has an optional `qv` argument ahead of sequence metadata, which Core supplied
positionally. Core now passes the four metadata arguments by keyword. Shared and
separate metadata regression tests pass. The failed run remains retained.

## Live async exercise: in progress

[First live attempt](https://beaker.org/ex/01M2ACJE4Q5CXKN507QN00S80X), Open Instruct
`82e7a0f27`, stopped at MILES argument validation: a 2048-token prompt cap
exceeded the shortened 1536-token context. No policy update ran. The corrected
config uses a 1024-token prompt cap, and plan now rejects this combination
before launch. A fresh `-r2` output root preserves the first attempt. Config:
`configs/miles/qualification/sequence-packing.toml`.
EP2 trainers + one TP1 engine, 3 async updates, 8×2 responses, lag 2, TIS,
4096-token pack budget, actual SGLang route capture/replay and recomputation.
Full-SFT KDA checkpoint used by the comparison runs; short responses bound cost.
A separate CPU audit on Saturn will verify retained samples, packing/replay,
policy clocks, rewards, publication equality and cleanup.

## Local checks

101 focused host tests, 33 combined pinned-runtime tests, and 2 Core API regression
tests passed. Open Instruct formatting/lint/type checks and targeted Core lint
passed. The combined runtime suite exposed an existing test's assumption that
PyTorch returns its set-backed serialization allowlist in a stable order; the
test now compares exact membership and still rejects unrecognized pickle types.

Not yet qualified: EP8 performance, long-context memory limits, learning quality,
checkpoint/resume with packing, and larger/multi-node production batches.
