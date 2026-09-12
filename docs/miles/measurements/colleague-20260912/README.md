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
