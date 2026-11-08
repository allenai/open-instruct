# Experimental policy-scoped prefix caching

Our long-context RL trials generate several responses per prompt. This integration aims to reuse their prefill while preserving active requests across weight updates. It combines a cache plugin in `allenai/olmo-sglang`, publication checks and sibling routing in `allenai/miles-olmo-core`, and configuration validation here. **SGLang itself is unchanged** at `b84bec5761a6cdc8b1df86b9b89ead80a7a9bb02`; there is no dependency on `sgl-project/sglang:main`.

## Repository boundaries

| Repository | Branch | Responsibility |
|---|---|---|
| `allenai/olmo-sglang` | `codex/policy-prefix-cache` | Register `olmo_policy_versioned_v1` through the pinned engine's existing radix-cache backend API; scope sharing to the serving policy; preserve and release retained private state. |
| `allenai/miles-olmo-core` | `codex/policy-cache-routing` | Route siblings consistently; verify backend and committed weight version before resuming generation. |
| `allenai/open-instruct` | `codex/policy-versioned-prefix-cache` | Permit the explicitly selected backend, validate memory/concurrency settings, and preserve existing cache-off defaults. |

These branches form a candidate integration. The separate `codex/policy-cache-lc-probe` experiment branch advances its candidate lock to the companion plugin/MILES revisions and the previously exercised LC Core scoring fixes (`903e57d`). It also includes the DocQA verifier from LC support revision `cb77cbeea`. SGLang and the binary base stay unchanged. Production main and its images are unaffected. Build the candidate image before enabling the backend. The old local SGLang-main prototype is superseded and must not be included.

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

GPU qualification remains: fixed-weight cache-off/on output comparison, weight updates with waiting and active requests, a fresh-request comparison with a cold engine at the new policy, repeated-update memory checks, then representative prefill and sustained-throughput measurement. Existing LC runs have not been changed, and no performance gain is claimed yet.
