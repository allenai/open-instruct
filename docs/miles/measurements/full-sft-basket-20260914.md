# Faster trainer qualification and frozen 200-update task baseline

The EP2 fast-settings live qualification completed all **24 optimizer updates**
on both ranks and exited zero: [Beaker](https://beaker.org/ex/01M2EXJV10NDMSQHXKETD3XSMV).
It used the full-SFT checkpoint, 6144-token packing, no activation recomputation,
guarded standalone-scoring skip, recorded router replay, TIS and mixed-policy
refresh. The initial check was bit-exact over 178,642 active tokens. Subsequent
updates skipped the redundant scoring pass; model/optimizer compilation and
experimental native kernel switches remained off.

Updates 6–23: median forward/backward/optimizer 16.36 s; 5,089 useful response
tokens/s across the complete 2+2 GPU allocation; 58.08% awaited collection;
0.79% stale response-token drops. This qualifies the faster trainer combination
on that workload, not checkpoint/export/resume or an EP8 mixture. The small
example adopts it; the larger defaults await their own exercise.

[Retained analysis](full-sft-basket-20260914/fast-qualification.json).

## Two-node baseline

[Submitted run](https://beaker.org/ex/01M2F0FJC8JCSFKV5CHDX5S10R), source
`34fb3a99a55d2548636602d3fa0d7155a53a302a`, image
`01M2CJG5RQQ93GEYNYAS7ASCQJ`, committed overlay. This attempt failed after one update; see the outcome below. It is not a
completed learning baseline.

- Eight EP8 trainers on node one; seven TP1 policy engines and one fixed Qwen3-32B
  judge on node two. Both full nodes are B300 on Holmes, urgent,
  `ai2/open-instruct-dev`, minimum runtime four hours, timeout 24 hours.
- Same full-SFT HF as the throughput campaign, not SFT1000 or hero:
  `/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1/hf`.
- 200 optimizer updates, 64 prompts × 4 responses, batch 256: 51,200 consumed
  training responses if every update completes. FIFO buffer 256 responses,
  producer admission 1,024, lag limit two, refresh/TIS enabled.
- Packing 6144, recomputation off, guarded scoring skip checked every 20 steps.
  Router replay stays enabled. Native compile/reduction candidates remain off.
- Seven engines with 32 concurrent requests each, decode graphs through 32,
  radix extra-buffer strategy, 786432 token slots and 1024 state slots per engine.
- 2048 prompt / 4096 response budget; this is a short-context efficiency baseline,
  not the 32K released Think recipe. LR 1e-6 constant, no KL, response-averaged
  policy loss, no GRPO std normalization, clipping 0.2/0.28, existing router aux losses.
- Initial/every-20 greedy held-out evaluation, 128 prompts/domain, one response
  each; judge rubric sampling remains temperature 1.0 and thus is stochastic.
- Native checkpoints every 100 updates and final HF export; no automatic
  multi-node restart until qualified. Online W&B project `ai2-llm/olmo-rl-comparison`,
  group `full-sft-basket-baseline-20260914`; retained generations and phase metrics.

[Run configuration](../../../configs/miles/qualification/full-sft-basket-fast-200.toml),
[frozen submission](full-sft-basket-20260914/train-launch.json).

## Frozen data

[CPU preparation](https://beaker.org/ex/01M2F09GMR27A0NMM4ZAWG5T2Q) on Saturn
passed. It adopted the audited Dolci manifest, preserving source weighting and
training duplicates while expanding held-out sets to 128/domain. All matching
held-out prompts/source identities were removed from training. It produced
101,434 training rows and 512 held-out rows; no further overlength exclusions
were needed beyond the source preparation. Both function/stdio positive and
negative code-service canaries passed.

Training verifier counts: math 30,018; IF 29,675; function code 12,046; stdio code
9,206; no-reference general 3,944; reference general 16,545. Evaluation code has
74 function/54 stdio prompts; general has 21 no-reference/107 reference prompts.

Source/model/template hashes, held-out identities and output hashes are in
[preparation.json](full-sft-basket-20260914/preparation.json). Both GPU replicas
verify frozen source/output hashes before starting. Future checkpoint comparisons
must reuse this exact artifact, seed, objective and held-out sets. Tokenizer or
chat-template changes require separately validated preparation, not silent reuse.

Pending: EP8 warm timings, per-domain rewards/lengths/caps, actual judge and code
latency, dropped/terminal samples, native saves, final export/reload, and resume.


## First attempt outcome

The 200-update attempt **failed after one optimizer update**. The producer
encountered `httpx.ReadError` at 04:19:49 UTC on its connection to the fleet router,
then stopped and joined its outstanding generation tasks. The trainer completed
its cold first update; the error propagated when requesting collection two at
04:39:47. Both nodes exited (driver 1, peer 143 by failure propagation).

All eight ranks completed the first optimizer step without skipping. The initial
scoring-skip comparison was bit-exact on 852,263 active response tokens. This is
one-step evidence, not EP8 steady-state qualification. No native checkpoint,
final export or resume was reached. Initial scoring took 426.9 seconds and
forward/backward/optimizer 753.1 seconds including cold compilation; neither is
a steady-state performance result. Large-profile defaults remain unchanged.

The exception is on **client → router**, distinct from the router → engine
transport errors in the earlier high-concurrency probes. The underlying cause
is not established. Do not claim an OOM, corrupted model or failed score check.
The separate 2+4 GSM8K follow-up remains a controlled throughput experiment.

Initial held-out results were math 0.78%, IF 23.76%, code 9.38%, general 65.94%
(the IF/general rubrics can award fractional credit). Capped responses were
95.3%, 69.5%, 72.7%, and 36.7%, respectively. The short 4096-token response budget
is a material limitation for this harder mixture; these are not GSM8K scores
and cannot establish a useful full-recipe learning baseline.

[Retained first-attempt measurements](full-sft-basket-20260914/first-attempt.json).

## Integration status

The winning EP2 settings and their supporting Open-Instruct changes are merged
into `robertb/miles-olmo-core` (merge `917fb0a44`; later documentation removal
`2c477efd5`). The MILES backend branch `robertb/olmo-core-backend` now includes
refresh, metadata retention, queue decision instrumentation and transport failure
diagnostics at `b18b71b18`. OLMo-core remains `robertb/miles-rl-adapter`,
`3d35ab326`; this promotion requires no additional Core implementation changes.

Validation: 454 host tests passed (one skipped), 171 focused runtime tests passed,
and 113 tests passed inside the rebuilt image without source mounts or runtime
patch injection. Ruff and compile checks passed. The initial host `ty` invocation
resolved an unrelated installed Core and reported three missing/incompatible APIs;
it passed against the locked Core source via `--extra-search-path`, with no rule
suppression. The documentation build and generated-reference check passed.

New immutable runtime image: `01M2F1RKZFZVJYAS0XQGEC3SEJ`, application source
`2c477efd5`, Docker digest
`sha256:60aa54bfdeba71c49a863e393d7715c567d70cf293512f76f3a6939a6502759f`.
The CPU image tests used the preceding `917fb0a44` build; only documentation
removal differs in the final build. The first 200-update attempt used its frozen
older-image/committed-overlay provenance, avoiding changes mid-experiment.

Live tracking: [W&B](https://wandb.ai/ai2-llm/olmo-rl-comparison/runs/moxmo8wg).
