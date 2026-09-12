# Core native scorer versus serving, full SFT checkpoint at update zero

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

Core's actual native scorer and its SGLang serving process were evaluated on identical tokens before any optimizer step, with router replay disabled. The CPU comparison validated both native rank files, payload hashes, token/mask/response alignment, complete routed-layer coverage, and uninstrumented serving controls. [Compact results](core-native-routes-20260911.json).

| Cohort | Cases | Sampled token/layer positions | Individual expert agreement | Exact top-16 set agreement | Active-token logprob mean absolute difference |
|---|---:|---:|---:|---:|---:|
| Fixed divergence prefixes | 4 | 10,944 | 97.314% | 63.752% | 0.04382 |
| Retained rollout 0 | 16 | 43,776 | 97.351% | 63.626% | 0.02008 |

The retained cohort has 31,905 active response tokens; maximum absolute logprob difference is 0.69929. Native routes are sampled at the first 16 and last 128 input positions of each sequence; serving routes use those same positions. Counts are token/layer observations, not distinct response tokens. Rank-strided diagnostic partitioning does not recreate historical training-rank assignment.

A top-16 set can differ by one expert while 15 selections still agree. Thus 36.4% differing complete sets corresponds to only 2.65% differing individual selections here. Exact-set agreement falls from 91.06% in logical routed layer 1 to 39.54% in layer 19. This is an observation of the natural replay-off forwards, not evidence that replay was broken.

Serving used the original Core recipe, without the new seven-tuner pin. The earlier independent-serving-process probes also showed route changes from numerical/autotuner variation. **Do not attribute this disagreement specifically to the trainer, or conclude that Core disagrees more than Megatron, until the matched Megatron measurement is available.** These are bounded eager-prefill observations, not original decode traces or an optimizer-gradient comparison.

The capture completed all measurement stages, then failed distributed teardown after 180 seconds. The report explicitly records `observations_verified=true` and `cleanup.completed=false`; the GPU experiment exited 1. The independent CPU comparison exited 0. No backward or optimizer step ran.

Runs:

1. Core native/serving capture: [Beaker](https://beaker.org/ex/01M27J5YARGEAV1RNQRYFGJT6T).
2. Independently finalized CPU report: [Beaker](https://beaker.org/ex/01M27N5DYPESAEW41TDVKVNW9V).
3. Paired CPU report, waiting for Megatron: [Beaker](https://beaker.org/ex/01M27MX7930HVMZYVMSKCMB49X).
4. Megatron native/serving probe, still queued at 07:18 UTC: [Beaker](https://beaker.org/ex/01M27J6E1FTNGJG9ECTE0FH58Q).
