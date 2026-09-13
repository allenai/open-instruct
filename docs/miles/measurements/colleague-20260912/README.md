# Colleague exercise results, 2026-09-12

Status is evidence as of submission/startup, not a completed qualification.
[Campaign and gates](../../plans/colleague-exercises-20260912.md).

| Case | Beaker | State / claim |
| --- | --- | --- |
| CPU preparation A | [01M2BX13SSQNY91TWM39GWP562](https://beaker.org/ex/01M2BX13SSQNY91TWM39GWP562) | Failed: only one eligible general-quality held-out prompt, two requested. Dense fixture and service/import canaries passed. |
| CPU preparation B | [01M2BX9TAJD31D652BVSDP7DF4](https://beaker.org/ex/01M2BX9TAJD31D652BVSDP7DF4) | Passed: explicit one-per-domain MoE eval quota; completed dense fixture reused only after content/hash checks. |
| 1: dense KL/multi-update, 1T + 3I | [01M2BXECF7QDB19Z2F9071W6K9](https://beaker.org/ex/01M2BXECF7QDB19Z2F9071W6K9) | Reached six optimizer steps, including nonzero policy-gradient and reference-KL terms; lifecycle/audit pending. |
| 2: MoE packing/replay/six-task/judge, 2T + 2I + 1J | [01M2BXF4PWKTPKNP7GYKHN1HZP](https://beaker.org/ex/01M2BXF4PWKTPKNP7GYKHN1HZP) | Failed before training: Ray GCS unreachable at advertised host IP; launcher networking defect identified. |
| 3: matching warm cache | Not submitted | Requires case 2 cache publication and audit. |
| 4: radix bundle | Not submitted | Requires case 2 numerical/service audit. |

Both GPU runs use candidate `c143bea53fc1`, image `01M2BX9JPT44DGQ9QWY5C0HF52`.
Each has four collections with initial/final evaluation. Two GPU attempts used;
eight remain. The CPU retry is an explicit extra preparation allocation.

Host tests: 255 passed plus three fixture tests. Pinned-image tests: 94 passed,
2 skipped; these were CPU tests inside the runtime image, not GPU qualification.
All four TOMLs validate in a fresh Python 3.12 environment without training deps.
Image A and B differ only in preparation/test/docs sources; training code and all
runtime dependency pins are identical. The retained runtime test log is image A.

Local follow-up found and repaired two existing type errors in code-service
length diagnostics by retaining the extracted program as a string. No reward
behavior change; this typing cleanup is not in the running image. Eight focused
code/fixture tests pass afterward, and full Open Instruct formatting, lint and
type checks pass with the local development dependency paths installed.

No lifecycle, learning, restart, cache speedup, or full-model numerical pass is
claimed from job submission. Expand each case with retained sample audits and
stage timings once its run ends.

## Expanded EP / pool-size wave

Submitted with source `06ee61b0d` and the same immutable runtime image
`01M2BX9JPT44DGQ9QWY5C0HF52`. The new host-networking setting lives in the
submitted Beaker spec, so this launcher correction does not require a new image.
88 host launch/judge/config tests and targeted formatting/lint passed.

| Exercise | Beaker | GPUs | State at submission check |
| --- | --- | ---: | --- |
| Mixed EP2 + two engines + judge, retry | [01M2BZ6J54ACHF4MTNZ7PHR15Y](https://beaker.org/ex/01M2BZ6J54ACHF4MTNZ7PHR15Y) | 5 | Scheduled; numerical/training outcome pending |
| EP8 + 8 engines | [01M2BZ6Q7ZRE6QY9FDR0DSAZCX](https://beaker.org/ex/01M2BZ6Q7ZRE6QY9FDR0DSAZCX) | 16 | Both replicas scheduled |
| EP8 + 16 engines | [01M2BZ6W14S0V6YXPN6EB7T238](https://beaker.org/ex/01M2BZ6W14S0V6YXPN6EB7T238) | 24 | Three replicas scheduled/acquiring resources |
| EP4 + 8 engines, eight trainers | [01M2BZ711YRWVYZA9MS1WAX1NX](https://beaker.org/ex/01M2BZ711YRWVYZA9MS1WAX1NX) | 16 | Queued: workspace slot limit, 57/64 occupied, each task needs eight |
| EP8 + 24 engines | Not submitted | 32 | Staged; only if 16 engines still leave meaningful trainer starvation |

The three sizing runs use identical 12-collection workloads, batch/lag/admission,
packing/replay, sampling and image controls; the EP4 comparison changes only EP
within the same eight trainer GPUs. Neither 8 nor 16 engines is declared optimal.
Report fastest cadence and GPU-efficient near-saturation choice separately. Judge
service throughput is intentionally excluded from this controlled sizing family;
managed-judge mixtures need their own end-to-end confirmation.

Six GPU experiments have now been submitted in the campaign, counting the failed
original and its retry. Three of these are the added sizing family. Keep launch
receipts here and update outcomes from the latest job of every replica.


## Completed first wave (2026-09-13 UTC)

Four cases completed every driver stage and final evaluation with exit 0 on every
replica: dense (eight optimizer steps), mixed EP2/judge retry (four), and EP8 with
eight/sixteen engines (twelve each). Native contract files contain 230, 6224 and
6216 replay observations respectively, with zero mismatches. All MoE optimizer
steps record nonzero sampled parameter changes on every rank. Dense changes seven
of eight times; its first warmup step deliberately uses LR zero. These are runtime
contract/lifecycle results, not a completed independent retained-sample reward audit.

| Mean after first two collections | EP8 + 8 engines | EP8 + 16 engines |
| --- | ---: | ---: |
| Sum of training/wait/publication phases | 191.1 s | 148.2 s |
| Training phase | 62.1 s | 60.0 s |
| Waiting for generated data | 85.2 s | 37.2 s |
| Publication phase including full diagnostics | 43.8 s | 51.1 s |
| Actual publication transfer (actor timeline) | 1.42 s | 1.57 s |

Sixteen engines increase warm throughput about 29%, using 50% more GPUs (24 vs
16 total), about 16% more warm allocated GPU-time per update. This does not settle
the production optimum: diagnostic_interval=1 intentionally snapshots, resets,
republishes and compares full serving weights every update. That adds a large
publication barrier in both arms. Next timing runs should retain startup checks
and disable periodic full diagnostics, matched across arms.
No 24-engine run is launched on the basis of these diagnostic-heavy numbers alone.

Cold first optimizer steps were 642/749 s; steady optimizer work was about 60 s.
Compiler publication succeeded for all 16/24 workers, taking about 44 s total at
shutdown; the mixed judge run published four workers in 15 s. Warm reuse still
requires case 3. Full raw results were downloaded to /tmp/colleague-final-*;
compact runtime-contract summaries are in completed-first-wave.json.

EP4 failed at rendezvous before training: one task was blocked by workspace/budget
slot limits while its peer acquired a node, started and hit the 20-minute startup
deadline. This does not establish an EP4 model/collective failure. Retry with a
fresh output root after the larger allocations finish; keep the failure bounded.

## Throughput follow-up configuration

`pool-ep8-i8-throughput.toml` and `pool-ep8-i16-throughput.toml` retain
`check_weight_update_equal=true` and set `core.diagnostic_interval=0`. Each has a
fresh output identity. The completed qualification configs remain unchanged;
the unsubmitted 24-engine config also uses startup-only audits. These changes
are staged for subsequent launches, not changes to already submitted jobs.

This removes periodic snapshot/reset/republish/compare and the extra gradient
norm/update sampling governed by the same interval. Replay diagnostics, policy
version/logprob gates, and scoring-pass checks retain their previous settings.
A new timing measurement is required; the 44–51 s publication phase above remains
the result for every-update audits. No startup-only performance result is claimed.

The throughput follow-ups now explicitly supply 64 requests per TP1 engine:
512 outstanding responses for eight engines, 1024 for sixteen, and 1536 in the
staged 24-engine config. All use sample-level replenishment; the optimizer batch
remains 512 responses. The previous runs inherited a producer limit of 64 prompt
groups (512 responses), even with sixteen engines, and replenished only after a
whole group finished. Consequently the new runs change both audit cadence and
producer scheduling; comparison with the old runs is an overall configuration
comparison, not an isolated measurement of any one change. Publication still
cancels unfinished groups in this runtime; a larger producer budget can also
increase discarded work. Retain cancellation counts alongside useful throughput.

Sixteen-engine throughput follow-up submitted as
[01M2C867KWDKSN1XRH228A6VGM](https://beaker.org/ex/01M2C867KWDKSN1XRH228A6VGM),
source `04b427592`, unchanged runtime image `01M2BX9JPT44DGQ9QWY5C0HF52`.
It uses 8 EP8 trainers + 16 TP1 engines, 1024 outstanding responses, sample
replenishment, and startup-only full weight audits. Twelve collections, batch 512,
lag budget 2, TIS, packing and replay remain enabled. Submitted is not passed.
The matching eight-engine follow-up and 24-engine candidate remain staged.
77 focused host tests, lint, and all three config translations passed.

Warm-cache case 3 completed with exit 0
([01M2C5Q1C437KT0S3GGM6H1PCM](https://beaker.org/ex/01M2C5Q1C437KT0S3GGM6H1PCM));
its detailed cache/reward audit remains pending. EP4 retry
[01M2C5Q6FF95BFBZJBT7WRP5W1](https://beaker.org/ex/01M2C5Q6FF95BFBZJBT7WRP5W1)
was still running at this submission check.

## Latest completed/blocked outcomes (2026-09-13 UTC)

EP4 retry completed 12/12 optimizer updates, both replicas exit 0, with no traceback
in the trainer log. The corrected sixteen-engine attempt did not reach training:
its third replica never acquired slots; another replica raised
`TimeoutError: Cluster startup/readiness deadline expired`, propagating cancellation
to its peers. No throughput result is available for the corrected configuration.
The next retry is held pending capacity rather than repeating partial allocation.
See the campaign plan's current summary for outstanding readiness coverage.
