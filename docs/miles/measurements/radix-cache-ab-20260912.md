# KDA radix cache A/B on a math/IF/code mixture

Three serving configurations, same prepared prompts (1,280 training prompts, 320 each
of math, instruction following, function code and stdio code, interleaved; 48 held-out),
same seed, same two-node layout (8 Core trainers at EP8 plus 8 TP1 engines), 64 × 8
collections with lag ≤ 2, 20 updates each. Run files:
`configs/miles/qualification/mixed-cache-ab-20260912-{off,radix,radix-mixed}.toml`;
preparation by `scripts/miles/prepare_mixed_cache_exercise.py`
([01M2BHNQPZX3A5STSA04MQNXES](https://beaker.org/ex/01M2BHNQPZX3A5STSA04MQNXES)).

| Arm | Serving | Beaker (training complete) |
| --- | --- | --- |
| off | radix cache off, 128 KDA state slots, round-robin router | [01M2BMGMSZB4MC9339AE59PHME](https://beaker.org/ex/01M2BMGMSZB4MC9339AE59PHME) |
| radix | cache on, `extra_buffer`, 332 slots, `cache_aware` router 0.8 / 4 / 1.5 | [01M2BMSSGSJYCEHVTSHBMES03J](https://beaker.org/ex/01M2BMSSGSJYCEHVTSHBMES03J) |
| radix+mixed | the radix arm plus `enable_mixed_chunk` | [01M2BS1T7HXQ3VN53W0H2DPBVQ](https://beaker.org/ex/01M2BS1T7HXQ3VN53W0H2DPBVQ) |

All three completed their 20 updates; all three then lost the final held-out
evaluation to the external code service (a persistent 500 for one sample), on images
that predate the fix in `de07a88a4`. Warm measurements below cover updates 3–20
([summary JSON](radix-cache-ab-20260912/three-arm-summary.json)).

| Warm updates 3–20 | off | radix | radix+mixed |
| --- | ---: | ---: | ---: |
| Wall time for 18 updates (min) | 31.5 | 32.0 | 31.7 |
| Cadence between updates (s) | 111 | 113 | 112 |
| Trainer step (s), different trainer node per arm | 45.7 | 50.2 | 53.8 |
| Trainer waiting for data, mean (s) | 52.0 | 48.7 | 39.5 |
| Collections ready without waiting (of 17) | 3 | 3 | 8 |
| Engine response tokens per GPU per second | 22,500 | 24,900 | 30,300 |
| Prefix cache hit rate | 0 | 0.128 | 0.198 |
| Cached tokens per sample | 0 | 45 | 68 |
| Publication transfer, mean (s) | 1.31 | 1.07 | 1.00 |
| Producer join stalls (30 s each, retried) | 0 | 1 | 3 |
| Response tokens, mean | 3,672 | 3,639 | 3,660 |
| Truncated at 4,096 | 80% | 80% | 81% |
| Training reward, mean | 0.088 | 0.087 | 0.085 |
| Trainer-versus-behavior log-prob gap | 0.024 | 0.024 | 0.024 |
| TIS clip fraction | ~0 | ~0 | ~0 |

## Reading

- **The controls work and are numerically clean.** The cache-on arms serve with the
  same log-probability agreement and reward as the baseline; the run-file aliases,
  capacity validation and router settings all resolved and ran.
- **Engine throughput rose with the cache and again with mixed chunk**, +11% and +35%
  in response tokens per GPU-second, and the trainer waited less, 52 s → 39 s per
  cycle with both on. Cached prefix per sample is small in absolute terms, 45–68
  tokens against 3,650 generated, so the throughput gain is mostly the cache-aware
  router and mixed chunking, not saved prefill.
- **Wall time did not move.** Each arm ran on a different trainer node and the
  trainer step varied 45.7–53.8 s across them (the node-to-node spread also seen in
  earlier runs); the generation-side savings in the mixed arm were consumed by a
  slower trainer step and by three 30 s producer-join stalls. The stall count rose
  with the cache on (0 / 1 / 3); each was recovered by the boundary retry.
- **This mixture is not a learning workload at a 4,096 cap.** 80% of responses hit
  the cap and 73% of groups carried no signal; the arms are a throughput and
  correctness comparison only.

## Consequences

Enable the cache and the cache-aware router by default: they cost nothing measurable
and are the validated olmo-miles path. Leave mixed chunk opt-in until its stall
interaction is understood. The producer-join stall now has a correlate (cache on)
worth following up in the boundary code. The code service's rejections are now
recorded per collection as `rollout/code_verifier/*` metrics.
