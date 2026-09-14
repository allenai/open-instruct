# HTTP keep-alive candidate — September 14, 2026

The first EP8 + seven policy engines + judge basket failed after one optimizer
update. At 04:19:49 UTC the producer received `httpx.ReadError` while reading
headers from the MILES router. Around the same time the router logged
`ClientDisconnect` while reading an incoming body, before its forwarding call.
These original errors lack a shared request ID; the disconnect may have resulted
from cancellation of sibling requests. Engine delivery of the original failed
request is unknown. The router and engines continued completing other requests.

## Candidate and scope

- Isolated MILES branch `robertb/http-keepalive-20260914`, revision
  `f13bb03c105f57f165bba097fb17a4dede95df26`.
- Router honors the existing `SGLANG_TIMEOUT_KEEP_ALIVE` environment setting,
  retaining its five-second default. The candidate run sets it to 60 seconds for
  both router and SGLang. This controls idle HTTP connections, not generation time.
- Producer sends its generated request ID in `x-miles-request-id`. Router logs
  receipt, forwarding start and completed upstream response with that ID.
  Forwarding start does **not** establish engine admission. Upstream completion
  does **not** establish successful delivery to the producer.
- Incoming disconnect/cancellation and upstream HTTP errors log phase, selected
  worker and elapsed time. Producer HTTP errors log request/group/sample identity,
  error class, HTTP status if present, and traceback. Payload text is not logged.
- Errors still propagate with the original cause. No automatic resampling, changed
  policy provenance, or general engine-replacement recovery is introduced here.
  A transport error remains fatal if it recurs; this is an instrumented candidate.

The launch overlay applies the complete router diff onto immutable base image
`01M2CJG5RQQ93GEYNYAS7ASCQJ`, with committed Open Instruct source. The primary
MILES runtime lock is unchanged pending qualification. The original failed run and
its output remain intact; the candidate uses a fresh run and output identity.

## Local evidence

The [reproducer](../../../../scripts/miles/diagnostics/http_keepalive_probe.py)
ran in the installed runtime under Docker with four CPU cores and 12 GiB RAM.
Each of three paired trials issued 672 requests (224 concurrent slots, three
requests per slot) returning 8,388,645-byte synthetic JSON bodies.

| Idle keep-alive | Read errors per trial | Total successes |
|---|---|---|
| 5 seconds | 29, 6, 1 | 1,980 / 2,016 |
| 60 seconds | 0, 0, 0 | 2,016 / 2,016 |

[Raw measurements](local-probe.json) support an idle-connection reuse hypothesis;
this single-server synthetic test does not establish the production root cause.
It is not a GPU throughput measurement.

The exact legacy-image source overlay applied cleanly. Twenty-six runtime tests
passed, including injected producer ReadError/503, router body disconnect before
forwarding, cancellation cleanup, correlation, and existing refresh/provenance
checks. For these CPU tests only, the unused Megatron debug plugin was disabled
because the host Docker runtime lacks a CUDA driver; HTTP/rollout code was real.
Seven host baseline/loader tests also passed; the candidate config validated.

## Relaunch

Config: `configs/miles/qualification/full-sft-basket-fast-200-keepalive60.toml`.
Same checkpoint, frozen training and held-out sets, 200 updates, EP8 + seven TP1
policy engines + one judge on two B300 nodes, Holmes/urgent, four-hour minimum.
Recipe unchanged except new identity/output and the keep-alive setting. The [launch receipt](launch.json) records source revision `455d5cd0c` and
[Beaker experiment](https://beaker.org/ex/01M2F7N19DQ3YJMRJAXJ1K4H59).
Both replicas were scheduled on Holmes at 05:54 UTC. Training results are pending. This short-context basket is
not the full 32K Dolci Think recipe.

## Terminal outcome observed during dense comparison launch

Experiment `01M2F7N19DQ3YJMRJAXJ1K4H59` completed 18 optimizer updates, then its
driver failed at 06:58:57 UTC on September 14. A code-execution POST exhausted
eight HTTP retries with 30-second read timeouts and exponential backoff.
The final exception originated in `open_instruct/miles/code_rewards.py`, for
`/prod/test_program`. Earlier retries are visible from approximately 06:50 UTC.
This is a separate failure from the fleet-router header-read error motivating
the keepalive change. The 200-update endurance criterion remains unmet.
Cause within the code service or a specific execution payload is unmeasured.
Logs/results were retained in `/tmp/full-sft-basket-keepalive-results` locally.
