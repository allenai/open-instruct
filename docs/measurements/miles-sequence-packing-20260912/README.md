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

## Live async exercise: passed

[First live attempt](https://beaker.org/ex/01M2ACJE4Q5CXKN507QN00S80X), Open Instruct
`82e7a0f27`, stopped at MILES argument validation: a 2048-token prompt cap
exceeded the shortened 1536-token context. No policy update ran. The corrected
config uses a 1024-token prompt cap, and plan now rejects this combination
before launch. A fresh `-r2` output root preserves the first attempt. Config:
`configs/miles/qualification/sequence-packing.toml`.
EP2 trainers + one TP1 engine, 3 async updates, 8×2 responses, lag 2, TIS,
4096-token pack budget, actual SGLang route capture/replay and recomputation.
Full-SFT KDA checkpoint used by the comparison runs; short responses bound cost.
[Corrected live run](https://beaker.org/ex/01M2ACWT28D6KFPW3N2HJ742ZP), source
`a318a75a8`, passed with exit 0. [Independent Saturn audit](https://beaker.org/ex/01M2AEGJJ24ARYVXZDHF4K7MNS)
also exited 0; `live-audit.json` has `passed=true` and `full_sample_audit=true`.
It checked all 48 consumed training responses (22,635 response tokens, 16 unique
prompts), both four-question evaluation rounds, immutable preparation hashes,
reward recomputation, masks, lag bounds, optimizer counters and seven publication
events (initial plus update/equality repeat per step). Workflow completion and
the Beaker exit establish successful run completion.

Every rank/update used two packs for eight samples: four times fewer forwards,
with unchanged real-token counts. All packed training replay observations passed
expert-ID equality, sample-tail coverage, gradient-mode and recomputation checks.
The first scoring-versus-training comparison was exactly zero over 8,015 active
tokens. The next two updates skipped standalone scoring. All three consumed
version-0 data, exercising lags 0, 1 and 2 without exceeding the configured bound.
TIS was enabled; its clipping fraction happened to be zero in this short run.

`live-runtime-summary.json` retains timing and memory observations. Rank-zero
optimizer time was 188.78, 40.68 and 2.13 seconds; the first two included cold
kernel compilation. Peak allocated trainer memory was about 166.2 GiB, including
the model, optimizer, diagnostics and activations, not just packing overhead.
Driver cycles were 362.64, 72.39 and 33.05 seconds; these also include publication
and the expensive full serving-weight equality checks. There is no matched
unpacked full-model timing arm, so these are observations, not a speedup claim.
The small greedy evaluations took 9.39 and 6.45 seconds. Their scores (1/4 then
3/4, all capped at 512 tokens) do not establish learning quality.

The job ran for about 24m37s including startup and teardown. Compiler-cache
publication timed out at 120 seconds for each of three workers, adding six
minutes after trainer disposal. All three reported `publish.status=unavailable`;
training still exited cleanly because cache publication is optional. Its cause
was not isolated here. Cache reuse/publication needs a separate follow-up; this
run does not qualify it or demonstrate warm startup. Serving's initial KDA
warmup health timeouts recovered before engine admission, and later health
probes stayed healthy.

## Local checks

101 focused host tests, 33 combined pinned-runtime tests, and 2 Core API regression
tests passed. Open Instruct formatting/lint/type checks and targeted Core lint
passed. The combined runtime suite exposed an existing test's assumption that
PyTorch returns its set-backed serialization allowlist in a stable order; the
test now compares exact membership and still rejects unrecognized pickle types.

Not yet qualified: EP8 performance, long-context memory limits, learning quality,
checkpoint/resume with packing, hero/dense architecture coverage, and larger/multi-node
production batches. Compiler-cache publication is an open follow-up as described above.
