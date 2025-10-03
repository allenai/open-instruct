# TODO: Implement Pauseable Weight Updates for vLLM v1 (v0.10.2)

> Goal: let the v1 `AsyncLLM` engine accept in-place weight updates (e.g. from an online trainer) without tearing down the engine and while keeping request KV caches alive, using a pause→broadcast→resume flow inspired by PipelineRL.

## 1. Understand Existing PipelineRL Flow
- [ ] Read `pipelinerl/vllm1.py` to confirm worker extension API, `collective_rpc_async` usage, and expected `WeightUpdateRequest` format.
- [ ] Inspect `pipelinerl/finetune_loop.py` (or wherever `WeightUpdateRequest` is produced) to understand tensor naming, sharding, and dtype assumptions.
- [ ] Verify that our trainer can emit compatible `parameters_info` (names, shapes, dtype string, shard index) per tensor-parallel rank.

## 2. Design Pauseable Architecture for v0.10.2
- [ ] Decide where to situate the pause logic: likely a thin wrapper around `AsyncLLM` exposing `async def pause_and_update(request: WeightUpdateRequest) -> None`.
- [ ] Define a synchronization strategy. Proposed: single `asyncio.Lock` held by all inference calls and by weight updates so they do not overlap.
- [ ] Map base executor APIs: confirm `AsyncLLM.engine_core` is an `AsyncMPClient` exposing `collective_rpc_async` and `stop_remote_worker_execution_loop_async`.

## 3. Implement Worker Extension
- [ ] Create `pauseable_v1_worker_extension.py` (or similar) next to our service code. Class must importable via dotted path for `worker_extension_cls`.
- [ ] Implement `init_actor_update_group(self, actor_idx, actor_ngpus, init_method, world_size)` with NCCL process-group setup mirroring PipelineRL but adapted to our rank layout.
  - [ ] Use `torch.cuda.device_count()` or config to compute per-node GPU count.
  - [ ] Cache `self.pg_rank` and `self.process_group`.
- [ ] Implement `receive_weight_update(self, request)`:
  - [ ] `torch.cuda.synchronize(self.device)` before updates.
  - [ ] Pre-compute dictionary mapping parameter names → `nn.Parameter`. Consider caching on first call (`self._param_map`).
  - [ ] For each `parameters_info` entry: allocate buffer, broadcast from rank 0, copy into parameter (`param.data.copy_(buffer)`).
  - [ ] If using fused or quantized layers, ensure the weight name resolves correctly; add error handling/logging when `info.name` not found.
  - [ ] Handle MultiStep runner or other wrappers (similar to PipelineRL’s `_base_model_runner` fallback) if we need multi-step scheduler support.

## 4. Integrate Worker Extension with AsyncLLM
- [ ] Allow CLI/config to pass `worker_extension_cls="<module>.PauseableWorkerExtension"`.
- [ ] Ensure module is importable by the engine subprocess (install package or adjust `PYTHONPATH`).
- [ ] Smoke-test engine startup to confirm workers instantiate without errors.

## 5. Implement Engine Pause & Resume Flow
- [ ] Add wrapper class (e.g. `PauseableAsyncLLM`) that composes an existing `AsyncLLM` instance.
  - [ ] Provide `generate`, `create_chat_completion`, etc., by delegating to the wrapped engine.
  - [ ] Add `async def pause_and_update(self, request)` that:
    1. Acquires global `asyncio.Lock` (shared with all inference entrypoints).
    2. Calls `await engine.engine_core.stop_remote_worker_execution_loop_async()` to drain ongoing loops.
    3. Issues `await engine.engine_core.collective_rpc_async("receive_weight_update", args=(request,))`.
    4. Releases the lock (next inference call will lazily restart execution loop).
- [ ] Wrap each inference entrypoint with `async with lock` to serialize against updates.
- [ ] Consider adding fast-path `try_lock`: if lock unavailable, queue requests or return error depending on SLA.
- [ ] Expose explicit `resume` only if needed; otherwise rely on lazy restart inside executor.

## 6. Update Serving Layer / Control Plane
- [ ] Extend HTTP (or gRPC) server with `/receive_weight_update` endpoint.
  - [ ] Deserialize `WeightUpdateRequest` payload.
  - [ ] Invoke `await pauseable_async_llm.pause_and_update(request)`.
  - [ ] Return success/failure JSON with timing metrics.
- [ ] Add request auth/validation (optional but recommended).
- [ ] Log durations for pause, broadcast, resume; expose metrics for monitoring.

## 7. Trainer-Side Adjustments
- [ ] Update trainer to partition updated tensors per tensor-parallel rank and send `WeightUpdateRequest` to the serving endpoint.
- [ ] Ensure broadcast source is rank 0 for each PG (Trainer must set `src=0` semantics by pushing data into the request only on that rank).
- [ ] Throttle update frequency; avoid overlapping updates while one is in flight.

## 8. Testing Strategy
- [ ] Unit tests for worker extension:
  - [ ] Mock `parameters_info` with small tensors; run `receive_weight_update` and confirm parameter data changes.
  - [ ] Ensure dtype verification triggers on mismatch.
- [ ] Integration tests:
  - [ ] Start multi-process engine in test harness; issue prompt, capture partial output, trigger weight update mid-stream, continue decoding; confirm no deadlock/crash.
  - [ ] Validate that cached KV reuse does not throw (even if output quality changes).
  - [ ] If multi-GPU: confirm all ranks update weights (inspect parameter checksum per rank).
- [ ] Stress test repeated pause/update/resume to watch for memory leaks or NCCL errors.

## 9. Operational Considerations
- [ ] Add metric counters for pause duration, update throughput, failure counts.
- [ ] Document runbook for rollback (e.g., restart engine if update fails).
- [ ] Evaluate whether we need versioning of weights to avoid stale updates; consider storing last applied step.

## 10. Documentation
- [ ] Write developer docs describing pause protocol, required config flags, and how to trigger updates.
- [ ] Add warnings about potential divergence when reusing KV cache with changed weights.

## 11. Follow-Up Enhancements (Optional)
- [ ] Support partial updates (only subset of tensors) with name filtering and default skip behaviour.
- [ ] Explore per-layer locking so we can update small adapters without full pause.
- [ ] Investigate automatic cache invalidation strategy for layers whose weights changed significantly.
