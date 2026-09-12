# Colleague exercise results, 2026-09-12

Status is evidence as of submission/startup, not a completed qualification.
[Campaign and gates](../../plans/colleague-exercises-20260912.md).

| Case | Beaker | State / claim |
| --- | --- | --- |
| CPU preparation A | [01M2BX13SSQNY91TWM39GWP562](https://beaker.org/ex/01M2BX13SSQNY91TWM39GWP562) | Failed: only one eligible general-quality held-out prompt, two requested. Dense fixture and service/import canaries passed. |
| CPU preparation B | [01M2BX9TAJD31D652BVSDP7DF4](https://beaker.org/ex/01M2BX9TAJD31D652BVSDP7DF4) | Passed: explicit one-per-domain MoE eval quota; completed dense fixture reused only after content/hash checks. |
| 1: dense KL/multi-update, 1T + 3I | [01M2BXECF7QDB19Z2F9071W6K9](https://beaker.org/ex/01M2BXECF7QDB19Z2F9071W6K9) | Started; attention preflight passed, creating Ray placement group. |
| 2: MoE packing/replay/six-task/judge, 2T + 2I + 1J | [01M2BXF4PWKTPKNP7GYKHN1HZP](https://beaker.org/ex/01M2BXF4PWKTPKNP7GYKHN1HZP) | Started, loading managed judge. No training result yet. |
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
