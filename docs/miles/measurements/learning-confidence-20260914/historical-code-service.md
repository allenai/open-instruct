# Historical Olmo 3 code-service audit — 2026-09-14

The three Think/code runs initially inspected from `scripts/train/olmo3/README.md` used the
shared AWS code-execution endpoint. The released 7B Instruct run used local execution,
as confirmed by the follow-up below. This is confirmed by both submitted
`--code_api_url` arguments and runtime warning lines naming that endpoint;
it is no longer an inference from the checked-in recipe.

| Release-linked run | Inspected driver job | Read-timeout warnings | HTTP 500 | HTTP 413 | HTTP 503 | Last observed training step |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [7B Think, November 19–20](https://beaker.org/ex/01KADRVRYEPW4YPKNN0RRNS137) | `01KADRVS2QN40WPQW4SM9Q31BJ` | 206 | 253 | 88 | 2 | 1456 |
| [7B Think, no pipeline, October 2–18](https://beaker.org/ex/01K6JZVN4EN3VHTJ820BV23HGC) | `01K6JZVND7B151PFJVVFVJDNPQ` | 2892 | 2353 | 736 | 4 | 1577 |
| [7B RL Zero Code, October 14–15 attempt](https://beaker.org/ex/01K7FSWM4717FAR9KF6GE958CA) | `01K7HRK5MKKTPDY5B41YKAESB4` | 4713 | 6896 | 623 | 1 | 2042 |

Counts are warning lines in the downloaded driver logs. They are not a
unique-sample count, a fraction of code requests, or a measurement of lost
accuracy. Training and evaluation verifier calls are not separated. The
code-only experiment had earlier attempts; this table covers its latest
replica-0 attempt only. All three inspected attempts were manually canceled;
do not describe their terminal exit codes as code-service crashes.

All three runtime argument dumps show `code_max_execution_time=1.0`.
The timeout warnings explicitly show `read timeout=30`. The Think runs use
pass-rate threshold 0.99; code-only uses 0.0.

The Think runs' recorded source revisions (`c4878185`, `e698112d`) both contain
`http_timeout = max(30, min(300, code_max_execution_time * 10))` and the broad
exception handler that logs `Error verifying code sample` and returns
`VerificationResult(score=0.0)`. Training progress is logged after the errors.
For example, the no-pipeline run logs a timeout at 2025-10-18 21:03:05 UTC and
training step 1577 at 21:16:03 UTC. Thus continued operation did not imply
successful grading. We cannot infer how many affected programs deserved a
nonzero reward: HTTP 500 may arise from harness/program errors, whereas 413
indicates an oversized request; the specific causes of read timeouts are not
recorded per payload.

## Implication for the MILES baseline

Our MILES adapter retained the old timeout formula but deliberately raises
on exhausted transport/gateway failures. The original runs tolerated those
failures by assigning zero. This is a material difference in the reward
contract, not evidence of an OLMo-core numerical failure.

A separate bounded probe today showed the AWS endpoint answering tiny
function and stdio checks in 0.65 and 0.15 seconds, but returning HTTP 503
at 30.11 seconds for 35 sequential one-second per-test timeouts, despite a
50-second client timeout. This demonstrates a service-side failure for a
legitimate grading duration. It does not establish the service's configured
gateway deadline or prove the cause of each historical timeout.

The later `origin/ngu` work records shared-service overload (`1ae99daa6`),
a local nginx timeout increase from 5 to 300 seconds (`d703db162`), and a
client timeout budget based on 120 tests (`9ba44477e`). These changes belong
to later DeepCoder work, not to the audited 2025 Think revisions. Its early
notes infer that Olmo 3 used local services, but the actual Beaker launch
arguments and HTTP errors above establish AWS usage for these runs.

Evidence: [structured counts, timestamps, source lines and log hashes](historical-code-service-evidence.json).
Re-fetch logs with `beaker job logs <driver-job-id>`; no new training jobs or
service configuration changes were made by this audit.

## Follow-up: local execution in the released 7B Instruct run

A colleague recalled eventual local-service use. Expanding the audit confirms
that this is true for at least one released Olmo 3 arm:

- [7B Instruct RL, November 17–19](https://beaker.org/ex/01KA8BY8MMAQWENWY4087MAPFE),
  driver `01KA8BY8R5QS2CZG1MVH7DR8A7`, records
  `code_api_url=http://10.95.1.109:8070/test_program`. HTTP request warnings
  name this same local endpoint. This proves actual local use, not merely
  local-server startup. It also has HTTP 504 warnings; local placement alone
  does not establish correct timeout handling.
- [Later 7B Think continuation, November 20–21](https://beaker.org/ex/01KAHGKA74GFJZ15G40271VDCK),
  latest driver `01KAHRW77BSP6HQ8ZJ303ZC9FS`, continues W&B `buq6ny46`.
  It starts a local service at `10.93.1.18:8070`, but its effective
  `code_api_url` and verifier errors still name AWS. Both startup and actual
  request destinations must be checked.

The released 32B Think and 32B Instruct Beaker specifications also explicitly
pass the AWS URL (`01KA4ZXT7MCVK493Y2B3K0BC82` and
`01KAJQH1X2PRZP5VYZ1F0Z96KK`); their runtime logs were not downloaded in this
follow-up. Do not equate that specification evidence with the runtime checks
performed for the 7B arms.

Corrected conclusion: Olmo 3 used both local and AWS code execution across
runs. The original three-run audit was not sufficient to generalize across
all released models or later continuations.

## Baseline policy update

Following user direction, MILES now defaults to the original Open Instruct
behavior: exhausted code-service failures and invalid replies receive zero
reward and do not terminate training. `OI_MILES_CODE_FAILURE_POLICY=raise`
(or verifier `failure_policy="raise"`) restores strict service failure handling.
Existing per-sample HTTP rejection handling is unchanged. Configuration errors
and task cancellation still propagate. Known-answer preparation checks explicitly
use strict mode.

Every fallback is logged and tagged `status="service_error"`, with HTTP status,
exception type, request/response stage and elapsed time. Rollout tracking records
`rollout/code_verifier/service_errors` and `service_error_fraction`, separate from
`rejected` and successful grading. The fraction's denominator is code verifier
calls represented in the consumed collection, not all generated samples or
all HTTP retry attempts. These metrics make the comparison's reward-service
limitations visible; this change does not repair the external execution service.
