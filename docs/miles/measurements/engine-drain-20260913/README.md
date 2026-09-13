# Engine-drain qualification — isolated branch

Implementation branch: `robertb/miles-engine-drain`; MILES runtime branch:
`robertb/engine-drain` (`bc582bc5c`). Core and serving arithmetic are unchanged.
These attempts use fresh identities and do not modify the colleague campaign.

## Attempts

| Attempt | Image | Result |
| --- | --- | --- |
| [A](https://beaker.org/ex/01M2CAMD21MHPDYSDZPEG6SA4D) | `01M2CAKXN7XTV41HA4ZXHF0ANH`, source `a41f371ad87e` | Stopped while queued to fix admission-clock ordering before allocation. No GPU evidence. |
| [B](https://beaker.org/ex/01M2CB2NK3KW4K1HY127YK943N) | `01M2CB2E68DZFCDXHG4W4KHCH4`, source `441d93839bdd` | Startup and legacy initial weight equality passed. Failed before new transport setup: `delivery_location()` assumed `CUDA_VISIBLE_DEVICES` was set. No optimizer updates. |

Attempt B ran from 02:58:01 to 03:08:27 UTC, September 13. This is unsuccessful
startup evidence, not the new transport's performance. The fix identifies the
trainer's selected physical GPU using Torch's device UUID, then exposes it as
local device zero to the independent sender. A regression test covers unset CVD.

| Attempt | Image | Result |
| --- | --- | --- |
| [Transport probe](https://beaker.org/ex/01M2CCHBGWRXJ7CZJ31DHSBQTY) | `01M2CCH3KXG5M2N8FJ48JRZ46X`, source `e5bba0f2ea72` | Passed, exit 0. Real two-GPU Ray/NCCL transport, synthetic receiver. Source was mutated after capture; received BF16 bytes matched the original exactly. |
| [C](https://beaker.org/ex/01M2CCPFJ1XYNC48G7W7KENZPC) | Same image/source as transport probe | Three optimizer steps completed. Stopped at the checkpoint join after reproducing the already-fixed completed-buffer deadlock. No committed checkpoint; not a successful lifecycle run. |
| [Barrier control](https://beaker.org/ex/01M2CCQY4DJBX0B62QWYRV213N) | Same image/source as C | Passed, exit 0: six optimizer steps, initial/final evaluation, saves and shutdown. |

The transport probe transferred only 32 KiB in 1.46 seconds including its first
collective. This validates process placement, immutable object-store ownership,
wire layout and communicator teardown. It is not a bandwidth measurement or a
SGLang model-loading test. See [its report](transport-probe.json).

## Current local evidence

- Wrapper suite: 309 passed, one skipped.
- Focused runtime/config/lifecycle suite: 63 passed in the pinned runtime image
  with the current source mounted; no GPU exposure.
- Ruff and type checking pass.
- Request IDs now distinguish repeated admissions of the same prompt group.

The six-update rolling lifecycle and independent engine progress now pass (D).
Fresh-process resume, complete optimizer overlap, and bounded real-engine failure
remain gates for the follow-up runs listed below.

## Additional review and test findings

A saturated completion queue can contain several finished tasks awaiting insertion,
none of which remain in `_active_tasks`. Quiescent capacity now uses the full
`_producing_groups` ownership ledger. A regression exercises four such completions
with no active decodes and verifies all survive the boundary without cancellation.
This correction postdates image C. C subsequently reproduced this exact failure
(`active_groups=0`, worker blocked) at its first save boundary and was stopped.
Image D includes the fix and completed; resume fixtures read D, not C.

The full CPU runtime suite initially could not collect
`test_core_policy_contract.py`: the base image lacks its cross-backend
`olmo_miles.evaluation.policy_contract_schema` helper. This is the same documented
base-image limitation as the earlier scoring-pass gate; it remains an explicit
exception, not a new skip added to the tests.

With only that module excluded, the first broad run produced 748 passes, 53 skips
and two failures: a historical cache fixture still expected `tmp-30d`, and the
retained-rollout loader removed NumPy entries from a pre-existing Torch allowlist.
The fixture now checks the current shared `tmp-7d` root, and the loader scopes only
new allowlist entries. Its regression also preloads entries to catch the latter
failure independent of test order. The corrected broad rerun passed: **752 passed, 53 skipped, zero failures** in 264 seconds, with only the documented cross-backend module excluded. Wrapper tests also passed again: **309 passed, one skipped**.

## First full-model rolling update

Trial C reached generation through the new transport after its initial full
weight equality check. Its first optimizer step completed on both EP2 ranks, with
74,454 active tokens and exact standalone/training scoring agreement (max and mean
absolute difference zero). Behavior/teacher scoring mean absolute difference was
0.00557; that is a different comparison from the exact scoring-path check.

At version 1, snapshot capture took **43.67 s** for **37,028,386,304 bytes**.
Independent engine delivery took **4.58 s** and **5.36 s**, with a reported sender
GPU allocation peak of **1,247,805,440 bytes** each. Both reopened version 1.
The first drain had no outstanding requests; it is not independent slow-peer
progress evidence. Cold training took 270 s and cannot be used as steady-state
publication performance.

The next revision packs owned BF16 tensor copies on the source GPU and copies one
whole bucket to CPU. This targets the measured per-tensor CPU copy/concatenation
cost without retaining a second full GPU model. CPU snapshot tests pass; the
revised two-GPU probe now uses two buckets and a noncontiguous FP32 source to check
BF16 capture and source mutation. The revised GPU transport probe passed; full-model capture timing remains pending.

The slow fixture now waits for admission to close before starting its delay.
The earlier fixed-at-admission delay in image D could expire during cold backward
compilation, before a publication drain began. Image D remains recorded as that earlier fixture, not evidence for the revised one.

## Additional attempts

| Attempt | Image / source | Status |
| --- | --- | --- |
| [D, original slow fixture](https://beaker.org/ex/01M2CDFK13AV7WV82EX4AMAJTR) | `01M2CDFB37KCSGKJZ36M12TK48`, `3c97e913b` | Passed, exit 0: six updates, two saves, initial/final evaluation and shutdown. Independent engine progress observed. |
| [GPU bucket probe](https://beaker.org/ex/01M2CEMH82TJRT3DV4C1SV5319) | `01M2CEM9RZ5CGQ1NVMTM7PY4R0`, `019e66609` | Passed: 8,392,704 bytes in two buckets, noncontiguous FP32 source, BF16 capture, post-capture mutation, exact receipt and teardown. |

The bucket probe's 2.02 s includes first-collective setup; it is not a throughput
benchmark. The final runtime image keeps this transport and adds bounded completion
backlog handling across repeated save boundaries: `01M2CFNVJAJ84SNR66MJ7Z90GV`,
source `2b8845852cf225355dca0809690e15982452b642`.


## Six-update result and control

D completed from 03:40:07 to 04:21:36 UTC. The control completed from 03:27:07
to 03:56:49. Both used the same initial checkpoint, GSM8K selection/seed, EP2/two
TP1 topology, optimizer, packing, replay, TIS and lag budget. They share the same
base runtime and Core/serving pins; their Open Instruct images differ as recorded
above. D includes the rolling-only lifecycle fix and a deliberate slow request.
Scheduling, sampled responses and compilation misses differ. This is a functional
control, not a steady-state speed or learning-quality comparison.

| Measurement | Barrier control | Rolling D (old snapshot capture) |
| --- | ---: | ---: |
| Optimizer updates / trained responses | 6 / 384 | 6 / 384 |
| Active response tokens | 716,422 | 802,587 |
| Mixed-reward groups (nonzero GRPO advantages) | 23 | 19 |
| Gradient norm range | 0.100–0.306 | 0.105–0.229 |
| Routers with changed FP32 master weights | 19 | 19 |
| Consumed lag | 0–2 | 0–2 |
| Mean training reward | 0.779 | 0.760 |
| Held-out correct, initial → final (16 questions) | 14 → 13 | 15 → 14 |
| Publication, mean driver-blocking seconds | 0.87 | 65.23 |
| Independent delivery, mean seconds per engine | not separate | 4.80 |
| Checkpoint boundary, seconds | 141 / 119 | 173 / 173 |
| Initial training stage, including scoring/compilation | 398 s | 611 s |
| Total allocation time | 29.7 min | 41.5 min |

The small sampled held-out scores do not establish learning equivalence. Both
runs performed useful learning work: finite nonzero gradients, mixed-reward groups,
finite behavior log probabilities, replay covering every token and real router
master changes. The initial router storage and import were both BF16; the audit
compares to the actual BF16 import before measuring FP32 master changes.

D reserved and completed **1,384 responses / 2,998,004 generated tokens** with no
unknown outcomes, engine failures, mixed groups, request ownership errors or
unreleased snapshots. Engine 0 reopened version 3 while engine 1 drained version 2;
**71 version-3 responses completed during that interval**. The longest drain was
58.49 s. No *entire* driver training interval fit inside that drain; the gated
follow-up below tests that remaining criterion conservatively.

Only 384 responses were trained in this deliberately short run: 1,000 completed
responses / 2,195,417 generated tokens were not consumed before termination.
These are completed work, not aborted partial decodes. The async buffer reported
46 stale groups filtered and retried, versus zero for the barrier control. Lifecycle
quiescence and slow capture allowed substantial overproduction; the latest image
also prevents repeated saves from expanding the completion backlog indefinitely.
Avoid treating zero cancellations as zero waste. Useful-token throughput and queue
sizing still need longer, warmed measurement.

Capture was the main regression: six frozen snapshots averaged 64.92 s, whereas
delivery averaged 4.80 s per engine. The old capture performed many CPU tensor
copies/concatenations. GPU bucket packing is qualified separately and is being
measured in the resumed run. Peak retained snapshot bytes in D were 37,028,386,304;
this is live object-store payload, not process/host peak RSS. Configured capacity is
two versions, requiring approximately 74 GB plus staging. Sender GPU allocation
peaks were about 1.25 GB per engine, excluding CUDA-context memory.

Exact stage arrays, lag counts, gradients, queue/staleness metrics and protocol
summary are retained in [six-update-comparison.json](six-update-comparison.json).
CPU read-only audits passed for the [control](https://beaker.org/ex/01M2CFT4DPKJ37YH20AE0W54NF)
and [D](https://beaker.org/ex/01M2CG5W7KFK4QKJSGT0P4CKZ5); their `audit.json` and
provenance are Beaker results. Two earlier control-audit attempts failed because
the checker assumed the wrong router dtype/layout; the checker now uses the
canonical flat Core layout and explicit BF16 import. Those were checker failures,
not altered training evidence.

## Final-image gates in progress

- [Fresh process, steps 7–12, gated slow engine](https://beaker.org/ex/01M2CG415X6NA0NJXDV3PCRAQR).
  Reads D's completed step-six checkpoint, publishes before admission, tests GPU
  bucket capture and complete optimizer overlap, then saves/evaluates/shuts down.
- [Deliberate owned SGLang engine loss](https://beaker.org/ex/01M2CG4GZ3C2CF38C2BBN8CMCQ).
  Independent output directory and process; reads D, then retires exactly its own
  engine after a trained publication. Success requires bounded driver failure and
  quarantine with no subsequent reopen. A successful harness exit means the
  expected failure was observed, not that training finished normally.

Final-image CPU runtime suite: **769 passed, 53 skipped**, with only the previously
documented cross-backend module excluded. Wrapper: **309 passed, one skipped**.
Ruff and type checking passed. The additional resume audit compares final masters
against the actual saved step-six masters, so pre-resume changes cannot certify
an inert resumed optimizer. No primary branch or shared example has been changed.


D's engine load/latency is included in the comparison JSON. Engine 0 completed
816 responses / 102 groups; engine 1 completed 568 / 71. Both reached 128 owned
requests at their busiest point (includes client/server queued work). Median
reservation-to-decode latency was 32.6 / 29.9 s; median whole-group completion was
49.6 / 43.8 s. The deliberate hold appears in engine 1's 125.3 s maximum. These
latencies include queue/prefill/decode, not isolated model-kernel time. Grading after
the final response averaged 30 / 25 ms. The buffer's reported staleness metrics
include examined rejected entries; the bounded-consumption claim uses actual
trainer contracts and retained samples instead.


## One-step lag checkpoint correction

A further CPU regression reproduced a checkpoint deadlock with lag one: after the
optimizer clock advances, already-submitted producer tasks can await admission at
the new version. Quiescing before publishing that version waits on those tasks.
The driver now submits rolling publication before checkpoint quiescence; the
barrier path retains its ordering. The test fails with the previous order and
passes after the fix; all 45 lifecycle/rolling-runtime tests passed together.

[Two resumed updates with lag one and saves every update](https://beaker.org/ex/01M2CHHVJ1D0A6SFF2EQCHVKAC)
exercise this on image `01M2CHD46881NR0DKT7KN96YWZ`, source `0bbc4e833cad`.
The previous lag-two resume and deliberate-loss runs remain on image F; their
results must not be mislabeled as qualification of this ordering correction.
