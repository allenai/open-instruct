# Experimental policy-scoped prefix caching

Our long-context RL trials generate several responses per prompt. This integration aims to reuse their prefill while preserving active requests across weight updates. It combines a cache plugin in `allenai/olmo-sglang`, publication checks and sibling routing in `allenai/miles-olmo-core`, and configuration validation here. **SGLang itself is unchanged** at `b84bec5761a6cdc8b1df86b9b89ead80a7a9bb02`; there is no dependency on `sgl-project/sglang:main`.

## Repository boundaries

| Repository | PR | Responsibility |
|---|---|---|
| `allenai/olmo-sglang` | [#32](https://github.com/allenai/olmo-sglang/pull/32) | Register `olmo_policy_versioned_v1` through the pinned engine's existing radix-cache backend API; scope sharing to the serving policy; preserve and release retained private state. |
| `allenai/miles-olmo-core` | [#33](https://github.com/allenai/miles-olmo-core/pull/33) | Route siblings consistently; verify backend and committed weight version before resuming generation. |
| `allenai/open-instruct` | [#2215](https://github.com/allenai/open-instruct/pull/2215) | Permit the explicitly selected backend, validate memory/concurrency settings, and preserve existing cache-off defaults. |

These branches form a candidate integration. This configuration PR preserves the production runtime lock: the stock image cannot run this backend. Land the companion runtime changes and advance the lock/options provenance before enabling it in a normal application build. The successful probe used a separate integration image containing the cache changes, startup-error handling and previously exercised LC scoring fixes; those unrelated scoring/verifier changes are not part of this PR. SGLang stayed pinned throughout.

## Configuration

Use the existing non-colocated one-node recipe, with these overrides merged into the existing tables:

```toml
[inference]
profile = "generic"
radix_cache = true
mamba_radix_cache_strategy = "extra_buffer"
sglang_max_running_requests = 64
sglang_server_concurrency = 64
sglang_max_mamba_cache_size = 512
router_policy = "consistent_hashing"

[miles]
pause_generation_mode = "in_place"
sglang_radix_cache_backend = "olmo_policy_versioned_v1"
sglang_schedule_policy = "lpm"
```

The capacities are a conservative starting configuration, not a measured throughput profile. Configuration validation requires explicit running-request and recurrent-slot limits and the `extra_buffer` strategy; recurrent slots must exceed five times the per-engine running-request limit. Long-context KV capacity still needs separate measurement. The application currently accepts this backend only with non-colocated, in-place publication and single-GPU TP1/PP1/DP1/EP1 inference engines. The plugin additionally validates resolved model/engine state at startup, rejecting sessions, speculation, sliding-window attention, adapters and host/external offload. Existing measured profiles remain cache-off.

## Policy changes and cold siblings

The backend combines the existing engine `weight_version` with the client's salt. New requests bind to the committed version. Active requests retain their old/mixed-policy state privately and cannot publish it to a new-policy namespace. Releasing or canceling a request reclaims its private state; old cache entries remain subject to ordinary bounded eviction.

This uses policy labels, not a separate transaction epoch. Changed weights require an increasing integer label; repeated publication of a label is valid only for the same unchanged policy. MILES checks the engine's existing server-info readback after finalization and before resume. It does not add an SGLang endpoint. Retained-state approximation and importance-sampling accounting remain unchanged.

Consistent-hash routing supplies a common prompt-group key to all siblings. SGLang's built-in LPM in-batch scheduling deprioritizes matching cold siblings to give one a chance to establish a reusable prefix. This is an opportunity for reuse, not a guarantee that all eight requests perform only one prefill.

## Validation

The rebased application branch passes 283 focused configuration, cache, profile, run-spec, refresh and constraint tests. Coverage includes missing explicit capacities, incompatible publication/placement/parallelism, measured-preset rejection, advisor behavior and unchanged cache-off defaults. Earlier companion checks covered MILES publication, retained-state and routing (45 tests), and the plugin's real CPU KV/KDA pools, cross-policy misses, completion/cancellation cleanup and built-in cold-sibling scheduling (17 tests).

The two-B300 DocQA integration probe completed four optimizer updates with exit code 0. The cache initialized, reused long prefixes (one request needed 15 new prefill tokens after 12,416 cached tokens), and published versions 0–4. Thirty-two logged responses continued across policy boundaries. Weight publications after training took 1.57–2.45 seconds. The first driver training call took 396 seconds including first-use compilation; subsequent calls took 19–26 seconds. This is a small mechanics probe, not a speedup comparison.

All four consumed training batches in that first probe came from policy version 0 because asynchronous prefetch ran ahead during compilation. A subsequent one-node comparison completed 24 optimizer updates with caching off and 24 with caching on, including training consumption of cross-version responses (50 off, 19 on). There were no retractions; the cache-on arm reused 41.2% of prompt tokens. Controlled policy-cache results matched native-cache outputs and logprobs exactly. Cached and uncached trajectories differed, so this is not a claim of bitwise equivalence to cache-off training or long-run learning equivalence.

The controlled 256-output-token serving benchmark showed 1.82–3.52× speedups across 12K–40K prompts. In training, responses averaged about 4,350 tokens and the completed-group buffer was full about 98% of sampled time in both arms. Mean warm update time was 83.09 seconds off versus 85.81 seconds on: no end-to-end speedup was established in the two-trainer/six-serving-GPU layout. Keep caching opt-in; capacity allocation and longer-run learning behavior need separate evaluation. TP>1 and speculation remain unqualified.

Runs:

1. Four-update DocQA probe: [Beaker](https://beaker.org/ex/01M6Y5Q3BWZQVRVN1CG0M7NNHM). Image `01M6Y5NN3FMQGCVV2WX92AMJS6`; application `a2feaaef3`, MILES `0c2b405`, plugin `f0444a8`, Core `903e57d`. Immutable result dataset `01M6Y5Q3C5VM98YMG402TE8WSY` contains publication, driver-timing and training-contract records. These are the tested integration revisions, not a claim that the subsequently separated PR heads were GPU-tested.
2. Completed cache-off/on training and serving panel: [Beaker](https://beaker.org/ex/01M6YCQWJC7Q3ESTATPQZRFKGY), result dataset `01M6YCQWJNX08T1N63Q3EBNWZR`. Application `13d47a043`, MILES `8802440b2a8ce7845f8dc07ff81bd345b3890e71`, plugin `f0444a8`. This also used the integration branches; the rebased configuration-only PR head has CPU regression coverage, not a new GPU run.
