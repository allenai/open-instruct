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

## Current local evidence

- Wrapper suite: 309 passed, one skipped.
- Focused runtime/config/lifecycle suite: 63 passed in the pinned runtime image
  with the current source mounted; no GPU exposure.
- Ruff and type checking pass.
- Request IDs now distinguish repeated admissions of the same prompt group.

Full SGLang transport, learning, slow-engine overlap, control performance,
fresh-process resume, and bounded GPU engine failure remain unqualified.
