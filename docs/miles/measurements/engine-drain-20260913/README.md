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
| [C](https://beaker.org/ex/01M2CCPFJ1XYNC48G7W7KENZPC) | Same image/source as transport probe | Full EP2 plus two TP1 engines; launched, result pending. |
| [Barrier control](https://beaker.org/ex/01M2CCQY4DJBX0B62QWYRV213N) | Same image/source as C | Matched old-mode control; launched, result pending. |

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

Full SGLang transport, learning, slow-engine overlap, control performance,
fresh-process resume, and bounded GPU engine failure remain unqualified.

## Additional review and test findings

A saturated completion queue can contain several finished tasks awaiting insertion,
none of which remain in `_active_tasks`. Quiescent capacity now uses the full
`_producing_groups` ownership ledger. A regression exercises four such completions
with no active decodes and verifies all survive the boundary without cancellation.
This correction postdates image C and must be exercised in a later image.

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
