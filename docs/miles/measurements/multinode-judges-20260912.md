# Tiny multi-node mixed-task and named-judge exercise

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

Status: **both the wrapper retry (r2) and native MILES backport (r3) passed**.
All four GPU replicas exited zero, and both corrected retained-sample audits
passed on Saturn. The native implementation is the retained project path.

| Run | GPU exercise | Corrected artifact audit | Retained report |
| --- | --- | --- | --- |
| r2: wrapper, per-expert publication | [passed](https://beaker.org/ex/01M29Y35Z257VCZ6KQTW7G3PY0) | [passed](https://beaker.org/ex/01M2A1ZYTM0KTTPB07GNJE2J1A) | [r2 audit](multinode-judges-20260912/r2-audit.json) |
| r3: native MILES, fused publication | [passed](https://beaker.org/ex/01M29ZPW6Z04D27MFTGMGB7GGD) | [passed](https://beaker.org/ex/01M2A2BHH3F428616A1D85AV4B) | [r3 audit](multinode-judges-20260912/r3-audit.json) |

The dated sections below retain the failure and repair history; their pending
statuses describe what was known at that time.

This exercises the researcher TOML launcher on real Dolci data. It is deliberately
small: two updates on two physical Holmes B300 nodes, with two Core EP2 trainer
GPUs, one TP1 policy engine, and one fixed Qwen3-32B judge. The purpose is placement,
reward delivery and cross-node training/publication, not learning or throughput.

## Reproducibility

- [Run configuration](../../../configs/miles/qualification/multinode-judges.toml)
- [Frozen launch receipt](multinode-judges-20260912/launch.json): source
  `7c8dd09792a4446d462ee4b72b810cfc84c690eb`, image `01M29VAV2PH85A9WRTF1ZR1GK4`.
- [GPU exercise](https://beaker.org/ex/01M29VB7YZXNZWK28W9H6CZSJ4).
- [CPU preparation](https://beaker.org/ex/01M29ST07SWYHMGFWBKJEJ3TC1), exit 0 on
  Saturn; [retained report](multinode-judges-20260912/preparation.json).
- Run root: `/weka/oe-training-default/robertb/open-instruct/runs/multinode-judges-20260912`.
- Coordination attempt: `22d7a6a56b31413181a27361393c9a79`.

The checkpoint is the recent full-SFT KDA model, not SFT1000 or hero. The immutable
preparation has 16 training prompts and six held-out prompts: math, modern IF,
function code, stdio code, general quality and reference-based general quality.
The exercise requests 8 prompts × 2 responses per collection, async lag at most
one update, TIS and rollout router replay. Evaluation runs before training and
after update two. Checkpoint saves and HF export are disabled. The ordinary
researcher examples retain their 8 × 8 batch; this bounded exercise is an exception.

## Issues caught before this attempt

1. CPU preparation exposed a JSONL reader splitting valid string contents at
   Unicode line/paragraph separators. Parsing now splits on physical LF records;
   the regression covers Unicode separators and CRLF. The first preparation
   experiment `01M29SDC44GZJ43BNTDBGTP1TV` failed before GPU allocation.
2. [First GPU attempt](https://beaker.org/ex/01M29SYFJPN98CZE91BCR0426A)
   reached two-node Ray readiness, but a generating judge health probe timed out
   during the first grading request. Both replicas stopped; no optimizer updates
   completed. Judge liveness now uses non-generating health mode, three consecutive
   failures, and a separate actual-grading deadline. Child logs are streamed to
   Beaker and retained on WEKA.
3. [Second allocation](https://beaker.org/ex/01M29TP88N1CGDHX1BF42R0YJT)
   scheduled both partial-node replicas on the same host. It was canceled before
   model startup. The launcher now uses explicit tasks with disjoint hostname
   pools drawn from the requested cluster. It also verifies the actual Ray node
   addresses and per-node GPU counts. Beaker rejects simultaneous cluster and
   hostname constraints, so hostname-constrained tasks use the resolved cluster
   inventory rather than specifying both fields.

4. The first attempt to reach training, `01M29VB7YZXNZWK28W9H6CZSJ4`,
   completed both updates but failed during final evaluation. At 04:14:18 UTC,
   the stock router marked its busy engine DEAD after three health failures;
   decoding continued. Final-eval HTTP retries exhausted and the driver exited 1.
   This matches the already reproduced olmo-miles connection-pool starvation
   failure documented in `docs/measurements/v02-discrete-experiments-20260905.md`
   around its “Replay failure attribution” section. The actual pinned router
   shares its bounded generation client with health. Our CPU regression
   reproduced `PoolTimeout` without any health request reaching the server,
   then passed with the ported independent health client under the same load.
   The GPU run did not log the underlying exception, so this is a demonstrated
   reachable cause consistent with its timeline, not direct observation of that
   exception in the GPU run.

The failed attempt's [rank 0](multinode-judges-20260912/failed-attempt/training_contract_rank0.jsonl),
[rank 1](multinode-judges-20260912/failed-attempt/training_contract_rank1.jsonl),
[publication](multinode-judges-20260912/failed-attempt/publication.jsonl), and
[stage timing](multinode-judges-20260912/failed-attempt/driver_timing.jsonl)
records are retained. They explicitly record final evaluation as failed.

A local continuation monitor, `/tmp/watch-multinode-judge-retry.py`, waits for both
retry replicas to exit zero, then launches the committed `--stage audit` through
the standard build wrapper using the exact retry image. This read-only audit runs
on Saturn and checks retained samples, reward accounting, policy versions, both
rank contracts, placements and cleanup. It issues no new judge requests.
Monitor state/log: `/tmp/multinode-r2-monitor.json` and `.log`; successful audit
results go under `/tmp/multinode-r2-audit-results`. A submitted retry is not a pass.

## Observed training checks (failed final-evaluation attempt)

Both ranks completed updates 1 and 2, with finite objectives, no skipped steps,
and nonzero gradients/parameter changes in dense, expert and router groups.
The first standalone-score/training-forward check was bit-identical over 47,190
active tokens. The second update skipped standalone scoring and checked behavior
agreement on its training forward. Mean trainer/behavior log-probability gaps
were 0.009084 and 0.007490, below the configured mean-gap gate of 0.05. Individual
token maxima were 0.16335 and 0.28505; the gate is not a per-token maximum bound.
Both consumed collections used behavior version 0, so the second exercised lag 1.

Initial publication transferred 37,028,386,304 bytes in 4.75 seconds. Fresh
versions 1 and 2 took 3.33 and 3.98 seconds, respectively; diagnostic repetitions
and equality-check overhead are additional. These are cross-node observations,
not a many-engine throughput qualification.

The first training forward/backward/optimizer interval took 270.5 seconds, plus
its separate cold scoring pass. The second took 43.7 seconds with no standalone
pass. Cold SGLang/FLA/TileLang compilation was visible in the logs. These two steps
cannot establish steady-state performance or attribute every second to compiling.

## Scope

This does not qualify EP8, 32K responses, many policy engines, judge throughput,
learning quality, checkpoint/resume, or automatic coordinated restart. Those
remain separate screens. The config and allocation planner support additional
inference nodes, independently of trainer GPU count. Homogeneous node allocations
can leave unused GPUs: 8+7+1 packs into 16; 8+8+1 currently reserves 24.

Local validation: 147 focused tests and `make style && make quality` passed.
Eight real-runtime router/async tests also passed inside the exact retry image;
[retained output](multinode-judges-20260912/router-tests.txt).
The saturation fixture observes the parent’s `PoolTimeout`, then verifies the
managed router reaches the healthy server while generation stays in flight.
This exercise is not the repository GPU-pytest suite.

## Native MILES backport and optimization integration

The next exercise (`-r3`) uses MILES `d29c04a944216c6309770fc9cf0d6ca37c066e17`
on `allenai/miles:robertb/olmo-core-backend`. This branch already descended from
the fetched fork primary; rebasing reported it up to date. The Open Instruct
change starts from project primary `446189de7` (`robertb/miles-olmo-core`),
including the merged scoring-pass and publication work. Core remains
`cfc42934d818036728d63f7ccdcd3b541eab9880`; serving remains
`02ccb5dcf641cbabc9b78a5bc65dacf8690707a7`. No generic upstream-main rebase or
compiled dependency upgrade is part of this backport.

The independent health transport and worker lifecycle now live in MILES itself;
the Open Instruct wrapper and installation hook are removed. The native router,
manager, lifecycle, Open Instruct spawn-target and async tests passed together:
34 passed in the pinned image with the new sources mounted. The test includes a
real occupied one-connection generation pool while health requests reach the
server, as well as stale-probe rejection, sticky quarantine and cancellation.

The fresh `-r3` output root reuses the immutable prepared six-task dataset. It
retains the two-update, two-node 2+1+1 topology, async/TIS/replay and equality
checks. It enables the primary profiles' already-qualified fused expert
publication and 2 GiB weight buckets. This exercises the integrated optimized
runtime, so its timing is not an isolated router-only A/B comparison. The native-backport exercise is submitted as
[01M29ZPW6Z04D27MFTGMGB7GGD](https://beaker.org/ex/01M29ZPW6Z04D27MFTGMGB7GGD),
source `aff77a6bd9a9854bf37b83d2e4df62dea5cd008b`, image
`01M29ZPAFZDZZYZ45GEMARYZJK`. Both replicas allocated distinct Holmes nodes.
The [frozen receipt](multinode-judges-20260912/native-router-launch.json)
records the complete configuration and placement pools. The exact built image,
without source mounts, also passed the
[34 regression tests](multinode-judges-20260912/native-router-tests.txt).
`make style`, `make quality`, and 147 focused host tests passed.

GPU and retained-sample audit results are pending. A local continuation monitor
waits for both GPU replicas to exit zero, then launches the existing strict audit
on Saturn through the build wrapper using the same immutable image. It records
state at `/tmp/multinode-r3-monitor.json`; the launch checkout is frozen at
`/tmp/oi-multinode-native-launch`. Do not remove that checkout while the monitor
is active. The earlier `-r2` wrapper run has its own independent monitor.

## Progress and audit-loader correction

The wrapper retry `01M29Y35Z257VCZ6KQTW7G3PY0` finished both GPU replicas
with exit 0 at approximately 05:26 UTC on September 12. Its logs contain no
router quarantine messages. The follow-up Saturn audit
`01M2A19CZENZN9WF6SK6RSTFSA` failed loading the first rollout: restricted
PyTorch deserialization did not allow the NumPy int32 expert-assignment arrays
retained by router replay. The earlier counter/placement/cleanup gates passed
before this load. Full sample validation was not completed by that audit.

Commit `2c4be8e394b587d3c243c1caec8f9aba9980c57f` scopes a minimal NumPy
reconstruction allowlist to the loader and keeps `weights_only=True`. Two new
runtime regressions verify int32 replay round-trip and rejection of unknown
pickle classes, with no allowlist leakage. Nineteen workflow-audit tests, lint,
and type checks passed. The corrected audit image is
`01M2A1ZQSQVABE9NBEK4KZ26WA`; r2's rerun is
[01M2A1ZYTM0KTTPB07GNJE2J1A](https://beaker.org/ex/01M2A1ZYTM0KTTPB07GNJE2J1A).
The r3 continuation now uses that same corrected audit image from frozen checkout
`/tmp/oi-mixed-audit-fixed`, retaining the original training image/run artifacts.

As of the 05:34 UTC r3 logs, both optimizer updates completed with nonzero finite
dense/expert/router gradients and parameter changes. Mean trainer/rollout score
gaps were 0.00943 and 0.00783, below the 0.05 gate. Native fused publication
sent 523 tensors/20 buckets; fresh versions 1 and 2 took 1.09 and 1.34 seconds
in the Core publication timer, versus r2's 29,669 tensors/35 buckets and
3.89/2.67 seconds. These are tiny-run observations, not a controlled steady-state
benchmark or the full driver publication stage (which includes other checks).
Final evaluation/cleanup and corrected retained-sample audits remain pending.

## Final qualification

Both reports set `passed=true` and `full_sample_audit=true`. Each covers 32
training responses across two updates, plus six initial and six final evaluation
responses. All six task types were consumed in training. Both named judge
bindings supplied actual trained rewards, reparsed from their retained replies.
Prompt/label/verifier identity and token hashes matched prepared data, policy
versions respected lag 1 (and exact evaluation versions), and reward components
matched the trained totals. Both trainer ranks completed two finite, non-skipped
optimizer updates with parameter changes. Each report records five publications,
distinct-node placement, judge/policy GPU separation, four judge canaries, and
successful final cleanup on both nodes. Neither completed run's logs contains a
router quarantine message.

Native r3 consumed: math 8, ifeval 10, function code 4, stdio code 4,
general-quality 2, and reference-quality 4 training responses. Async selection
produced a slightly different mix in r2: 8, 8, 4, 4, 4, 4 respectively.
This is another reason these runs are not a strict numerical/performance A/B.

Native r3's two driver cycles were 836.94 and 118.14 seconds; r2's were 879.43
and 133.48 seconds. Initial/final mixed evaluation took 241.69/330.61 seconds
in r3 and 234.66/441.33 seconds in r2. Startup/cold work dominates this tiny
exercise; judge-backed evaluation is still expensive. These measurements do not
establish sustained throughput. Qualification remains two EP2 updates, not
EP8, long context, learning quality, checkpoint recovery, or injected failures.
The audit reparses stochastic judge outputs and checks deterministic reward
accounting; it does not independently rerun the deterministic verifiers.
