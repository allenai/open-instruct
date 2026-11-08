# Experimental policy-scoped prefix caching

Our long-context RL trials generate several responses per prompt. This integration aims to reuse their prefill while preserving active requests across weight updates. It combines a cache plugin in `allenai/olmo-sglang`, publication checks and sibling routing in `allenai/miles-olmo-core`, and configuration validation here. **SGLang itself is unchanged** at `b84bec5761a6cdc8b1df86b9b89ead80a7a9bb02`; there is no dependency on `sgl-project/sglang:main`.

## Repository boundaries

| Repository | Branch | Responsibility |
|---|---|---|
| `allenai/olmo-sglang` | `codex/policy-prefix-cache` | Register `olmo_policy_versioned_v1` through the pinned engine's existing radix-cache backend API; scope sharing to the serving policy; preserve and release retained private state. |
| `allenai/miles-olmo-core` | `codex/policy-cache-publication` | Route siblings consistently; verify backend and committed weight version before resuming generation. |
| `allenai/open-instruct` | `codex/policy-cache-config` | Permit the explicitly selected backend, validate memory/concurrency settings, and preserve existing cache-off defaults. |

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

The capacities are a conservative starting configuration, not a measured throughput profile. Recurrent slots must exceed five times the per-engine running-request limit; long-context KV capacity still needs separate measurement. The plugin initially restricts serving to device-only TP1/PP1/DP1 text generation without sessions, speculation, sliding-window attention, adapters or host/external offload. Existing measured profiles remain cache-off.

## Policy changes and cold siblings

The backend combines the existing engine `weight_version` with the client's salt. New requests bind to the committed version. Active requests retain their old/mixed-policy state privately and cannot publish it to a new-policy namespace. Releasing or canceling a request reclaims its private state; old cache entries remain subject to ordinary bounded eviction.

This uses policy labels, not a separate transaction epoch. Changed weights require an increasing integer label; repeated publication of a label is valid only for the same unchanged policy. MILES checks the engine's existing server-info readback after finalization and before resume. It does not add an SGLang endpoint. Retained-state approximation and importance-sampling accounting remain unchanged.

Consistent-hash routing supplies a common prompt-group key to all siblings. SGLang's built-in LPM in-batch scheduling deprioritizes matching cold siblings to give one a chance to establish a reusable prefix. This is an opportunity for reuse, not a guarantee that all eight requests perform only one prefill.

## Validation

Focused Open Instruct configuration/cache/profile tests pass (120 tests). MILES publication, retained-state and routing tests pass (45 tests). Plugin tests against the unchanged pinned SGLang source exercise real CPU KV/KDA pools, cross-policy misses, completion/cancellation cleanup, and built-in cold-sibling scheduling (17 tests).

The two-B300 DocQA integration probe completed four optimizer updates with exit code 0. The cache initialized, reused long prefixes (one request needed 15 new prefill tokens after 12,416 cached tokens), and published versions 0–4. Thirty-two logged responses continued across policy boundaries. Weight publications after training took 1.57–2.45 seconds. The first driver training call took 396 seconds including first-use compilation; subsequent calls took 19–26 seconds. This is a small mechanics probe, not a speedup comparison.

All four consumed training batches came from policy version 0 because asynchronous prefetch ran ahead during compilation. Cross-version serving was exercised, but training on those cross-version responses was not. Remaining checks are fixed-weight cache-off/on output parity, fresh requests against a cold new-policy engine, training consumption of retained responses, repeated-update memory stability, and representative throughput. TP>1 and speculation remain unqualified.

Runs:

1. Four-update DocQA probe: [Beaker](https://beaker.org/ex/01M6Y5Q3BWZQVRVN1CG0M7NNHM). Image `01M6Y5NN3FMQGCVV2WX92AMJS6`; application `a2feaaef3`, MILES `0c2b405`, plugin `f0444a8`, Core `903e57d`. Immutable result dataset `01M6Y5Q3C5VM98YMG402TE8WSY` contains publication, driver-timing and training-contract records. These are the tested integration revisions, not a claim that the subsequently separated PR heads were GPU-tested.
