# Tiny multi-node mixed-task and named-judge exercise

Status: the first training attempt completed both optimizer updates but failed
during final evaluation. A retry with the managed router is being prepared.

This exercises the researcher TOML launcher on real Dolci data. It is deliberately
small: two updates on two physical Holmes B300 nodes, with two Core EP2 trainer
GPUs, one TP1 policy engine, and one fixed Qwen3-32B judge. The purpose is placement,
reward delivery and cross-node training/publication, not learning or throughput.

## Reproducibility

- [Run configuration](../../configs/miles/qualification/multinode-judges.toml)
- [Frozen launch receipt](miles-multinode-judges-20260912/launch.json): source
  `7c8dd09792a4446d462ee4b72b810cfc84c690eb`, image `01M29VAV2PH85A9WRTF1ZR1GK4`.
- [GPU exercise](https://beaker.org/ex/01M29VB7YZXNZWK28W9H6CZSJ4).
- [CPU preparation](https://beaker.org/ex/01M29ST07SWYHMGFWBKJEJ3TC1), exit 0 on
  Saturn; [retained report](miles-multinode-judges-20260912/preparation.json).
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
This exercise is not the repository GPU-pytest suite.
