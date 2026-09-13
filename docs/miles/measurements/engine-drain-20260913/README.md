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

Complete rolling-run lifecycle, slow-engine overlap, fresh-process resume, and
bounded GPU engine failure remain unqualified. Initial SGLang weight equality
and optimizer execution have passed in C; the old-mode control completed.

## Additional review and test findings

A saturated completion queue can contain several finished tasks awaiting insertion,
none of which remain in `_active_tasks`. Quiescent capacity now uses the full
`_producing_groups` ownership ledger. A regression exercises four such completions
with no active decodes and verifies all survive the boundary without cancellation.
This correction postdates image C. C subsequently reproduced this exact failure
(`active_groups=0`, worker blocked) at its first save boundary and was stopped.
The running image D includes the fix; resume fixtures now read D, not C.

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
compilation, before a publication drain began. Image D's running attempt remains
recorded as that earlier fixture, not evidence for the revised one.

## Additional attempts

| Attempt | Image / source | Status |
| --- | --- | --- |
| [D, original slow fixture](https://beaker.org/ex/01M2CDFK13AV7WV82EX4AMAJTR) | `01M2CDFB37KCSGKJZ36M12TK48`, `3c97e913b` | Running with corrected completion ledger and original fixed delay. |
| [GPU bucket probe](https://beaker.org/ex/01M2CEMH82TJRT3DV4C1SV5319) | `01M2CEM9RZ5CGQ1NVMTM7PY4R0`, `019e66609` | Passed: 8,392,704 bytes in two buckets, noncontiguous FP32 source, BF16 capture, post-capture mutation, exact receipt and teardown. |

The bucket probe's 2.02 s includes first-collective setup; it is not a throughput
benchmark. The next full-model exercise reuses this immutable image.
