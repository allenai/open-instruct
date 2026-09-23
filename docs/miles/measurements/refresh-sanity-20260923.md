# Refresh sanity checks, September 23, 2026

The reviewed upstream migration completed eight optimizer updates on both B300
and H100. Refresh produced responses spanning policy versions, and neither run
reported aborted rollout groups.

| Check | B300, Holmes | H100, Jupiter |
| --- | --- | --- |
| Experiment | [B300 run](https://beaker.org/ex/01M36F541HWJKDWT85MPKGVRF6) | [H100 run](https://beaker.org/ex/01M36F49XMPRJ3V27902E1MGXY) |
| Exit code | 0 | 0 |
| Optimizer updates | 8 | 8 |
| Training responses | 64 | 64 |
| Mixed-policy training responses | 10 | 16 |
| Tokens generated before each response's replay version | 1,306 | 1,298 |
| Reported aborted groups | 0 | 0 |
| Reported stale groups filtered | 1 | 0 |

A [read-only audit on Saturn](https://beaker.org/ex/01M36FS4RQFZ89T065RFR4ZNRK)
passed for both runs. It checks all 64 retained responses per run: contiguous
native token-version spans cover each response, relative refresh metadata agrees
with those spans, rollout log probabilities have the right length, and versions
stay within the configured staleness bound. Each batch contains four distinct
two-response groups with rewards zero and one; groups are not reused across
updates. Workflow records confirm completion of rollout IDs zero through seven.

## Configuration and provenance

Both jobs use one trainer GPU and one SGLang GPU, a two-layer, four-expert tiny
Olmo MoE checkpoint, Torch attention, and disabled CUDA graphs. Each update uses
four prompt groups with two responses each, capped at 256 response tokens and
512 context tokens. Refresh uses fully asynchronous collection, maximum policy
staleness two, TIS correction, and 32 producer samples. Synthetic alternating
rewards exercise zero-variance filtering and nonzero advantages; these are
mechanics checks, not learning-quality measurements.

Evaluation runs before training and every four updates. W&B is offline.
Checkpoint saving and HF export are disabled; rollout capture is retained.
Compiler-cache setup and publication succeeded in both trainer and serving
processes. Both runs started with cache misses; cache restoration was not
qualified by these runs.

- Open Instruct: `af151b168e0b0c7f7398341767fc6769963f90b6`.
- MILES: `cd0cbe5cc08de85128ed2b56db0c76e6681ef50b` on the private fork's `main`.
- OLMo-core: `e505356353aa7ce1f6ff83e24d6eb945f463714e`.
- olmo-sglang: `72f194a35045f02cc7d87980819bd0e4652cc931`.
- Immutable Beaker image: `01M36E342G8ZWZ8ZR53YAEFDFM`
  (`robertb/open-instruct-miles-core-af151b168e0b`).

The submitted specifications, resolved plans, workflow completion records,
training contracts, rollout-flow metrics, and logs are in each experiment's
results. Full rollout tensors remain under
`/weka/oe-training-default/robertb/open-instruct/runs/`, in
`upstream-refresh-sanity-b300-20260923` and
`upstream-refresh-sanity-jupiter-v2-20260923`. Disposable local configurations
and audit scripts are under Git-ignored `runs/miles-sanity-20260923/`.

## Review fixes exercised

Core now sets the serving weight version with `abort_all_requests=False`.
The producer controls barrier draining; refresh preserves active requests.
Focused tests cover the actual actor call in barrier, refresh, and engine-drain
modes. Empty token-version spans are ignored; training still rejects responses
without any version attribution. Startup-cache configuration fails explicitly
if either expected worker pool is absent.

MILES restores depth-two IPC restrictions for LoRA and mixed colocated/distributed
engines, drains pending transports after receiver failures, and readmits
quarantined native-router engines only after confirmed publication. These paths
have focused unit coverage; this two-GPU disaggregated run does not exercise
depth-two IPC or quarantine recovery. Follow-up fork commits separate transport,
router recovery, and restoration of upstream formatting.

Focused validation passed: 155 fork tests and 71 Open Instruct adapter tests,
plus the relevant lint/format checks and generated-documentation check. The
[migration report](miles-upstream-20260922.md) records older full-suite failures
and missing dependencies; these results do not imply that the whole suite passes.

## Startup observations and scope

B300 briefly reported detokenizer health-check timeouts during startup, then
recovered without intervention. This run does not establish the cause of that
delay or compare throughput across GPU types.

The first H100 attempt failed CUDA preflight because its host driver was older
than the image's CUDA runtime. The successful retry uses the same image with
`/usr/local/cuda/compat` first in `LD_LIBRARY_PATH`, followed by the NVIDIA and
CUDA library directories. That override was confined to the disposable H100
configuration; it is not required by the successful B300 configuration.

These runs qualify tiny full-weight refresh mechanics. They do not repeat
multi-node, expert-parallel, LoRA, checkpoint/resume, export, or engine-drain GPU
qualification. The documented overlapping engine-drain controller-window
limitation remains; this work does not change its scheduling algorithm.
