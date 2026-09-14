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
Recipe unchanged except new identity/output and the keep-alive setting. Launch
receipt is recorded separately after submission. This short-context basket is
not the full 32K Dolci Think recipe.
