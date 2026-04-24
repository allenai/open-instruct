# Weight-Sync Hangs in olmo-core GRPO (`open_instruct/grpo.py`)

Debugging notes for the deterministic hang(s) in `VLLMWeightSyncCallback.post_step` around step 1 of olmo-core GRPO training with FSDP2 + torch.compile on Qwen3-4B-Base.

Branch: `finbarr/post-training-experiments`.
Script: `scripts/train/qwen/qwen3_4b_dapo_math_oc.sh`.
Config: `fsdp_shard_degree=4`, `num_replicas=1`, `vllm_num_engines=4`, `vllm_tensor_parallel_size=1`, `inflight_updates=True`, `async_steps=4`.

## Symptom

Training reaches end of step 1. `VLLMWeightSyncCallback.post_step` starts the vLLM weight sync. The job then hangs forever: no NCCL heartbeat timeout fires, no traceback, no forward progress. Without instrumentation the stdout simply goes silent.

## Debugging the visibility problem

Before the real bugs could be diagnosed we had to fix log suppression so we could see per-rank state:

1. olmo-core's `logger_utils.setup_logger` only emits INFO on rank 0 (other ranks get WARNING). Moved phase markers from `logger.info` to `logger.warning`, then ultimately to direct `print(..., flush=True, file=sys.stderr)` via a `_phase()` helper.
2. `TORCH_LOGS=+dynamo,+inductor,+recompiles` was drowning phase markers in torch.compile verbose output. Removed.
3. `PYTHONUNBUFFERED=1` was added to guarantee flush.
4. Ray log deduplication (`RAY_DEDUP_LOGS=1`, the default) was collapsing identical phase lines across ranks so ranks 2/3 appeared silent. Added `RAY_DEDUP_LOGS=0` to the launch script.
5. `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180` was lowered from the 2-hour default to force a faster NCCL-timeout crash if a collective actually hung (turned out none did).

`_phase()` markers were added at every phase boundary in `open_instruct/grpo_callbacks.py::VLLMWeightSyncCallback.post_step` and `open_instruct/vllm_utils.py::broadcast_weights_to_vllm` (entering, `engine.sleep` start/done, `unshard` start/done, `trainer_send_weights` start/done, `reshard` start/done, returning, `ray_get(refs)` start/done, `wake_up` start/done, exit barrier start/done).

## Bug #1 — FSDP2 unshard deadlock on non-rank-0 ranks (FIXED)

**Observed**: Rank 0 last log was `unshard start (37 blocks)`. Ranks 1/2/3 went straight from `entering broadcast_weights_to_vllm` back to `ray_get(refs) start` in the caller — they never logged `unshard start/done`, `reshard start/done`, or `returning`. Clearly they had taken a different branch inside `broadcast_weights_to_vllm` and skipped the FSDP2 collective.

**Root cause**: `open_instruct/vllm_utils.py:1411` had

```python
if model_update_group is None:
    return _broadcast_weights_ipc(...)
```

`model_update_group` is only initialized on **rank 0** in `open_instruct/grpo_fast.py::setup_model_update_group` (lines 511-542); on ranks 1-3 it stays `None`. So ranks 1-3 took the IPC early-return and skipped `block.unshard()`. Rank 0 then blocked forever in the unshard all-gather because ranks 1-3 never participated.

**Fix** (commit `531a363f6`): gate the IPC fallback to rank 0 only.

```python
if model_update_group is None and is_rank_0:
    return _broadcast_weights_ipc(...)
```

Non-zero ranks now fall through into the FSDP2 branch and participate in unshard/reshard. `trainer_send_weights` on rank 0 stays gated by `is_rank_0`.

## Bug #2 — FSDP2 weight-sync dtype mismatch (FIXED)

Surfaced only after Bug #1 was fixed, because before the fix ranks 1-3 never reached the NCCL path where the dtype-mismatched metadata was sent.

**Observed**: `[DIAG]` logs at step 1 in experiment `01KQ04CX5NTZH30WQ95JW753JP` showed

- Trainer rank 0: every `p.data.dtype == torch.bfloat16` (`[DIAG] param ... dtype=torch.bfloat16`).
- All 4 vLLM engines received `dtype_names=['float32', 'float32', ...]`.

Names (399 on both sides) and shapes matched; only the dtype was wrong.

**Root cause**: `_collect_weight_metadata` (`vllm_utils.py:1604`) read `param.dtype` from `model.named_parameters()` **before** `block.unshard()`. With FSDP2 mixed precision, the sharded master weights are `fp32`, but `unshard()` materializes a `bf16` compute buffer. So metadata reflected the fp32 master dtype while `_prepare_params_for_sync` (called after unshard) cloned and sent the bf16 buffer. vLLM allocated fp32 receive buffers for bf16 bytes; the NCCL recv side waited for 2× the bytes the trainer ever sends, so `collective_rpc` never returned.

**Why grpo_fast.py doesn't hit this**: DeepSpeed ZeRO's `GatheredParameters` gathers the master-precision tensor; `param.dtype` pre-gather matches what the trainer then sends. No asymmetry between metadata time and send time. FSDP2's two-dtype split (fp32 master / bf16 compute buffer) is the anomaly.

**Fix** (commit `1fda6231a`): in the FSDP2 branch of `broadcast_weights_to_vllm`, reorder so that after `unshard()` the metadata (`names`, `dtype_names`, `shapes`) is derived from the cloned tensors produced by `_prepare_params_for_sync`, and `engine.update_weights.remote(...)` is fired with that metadata *before* `trainer_send_weights`. The non-FSDP2 (DeepSpeed/FSDP1) path is unchanged since pre-gather dtype is authoritative there.

Verified in experiment `01KQ06VPYV0PKN514MAKMPPS2M`: `[DIAG] dtype_names[:5]=['bfloat16', 'bfloat16', ...]` on every engine, matching the trainer's `torch.bfloat16`.

## Bug #3 — vLLM `update_weights` RPC still never returns (OPEN)

With correct metadata and a correct NCCL send, the job **still hangs** at step 1. Same signature as before: rank 0 reaches `trainer_send_weights done`, ranks 1-3 complete through `wake_up done → exit barrier start`, and all 4 engines sit at `collective_rpc pending` forever. NCCL heartbeat never trips — no in-flight collective on the send side. So the dtype fix was necessary but not sufficient.

Engine-side heartbeat in `01KQ06VPYV0PKN514MAKMPPS2M` (1100 s+):

```
[vllm engine-core utility] collective_rpc pending heartbeat=225 elapsed=1125.1s
[vllm frontend update_weights] collective_rpc pending heartbeat=225 elapsed=1125.1s
[vllm actor update_weights] waiting for coroutine heartbeat=225 model_step=1 future_done=False
    active_tasks=32 done_active_tasks=0 loop_thread_alive=True elapsed=1125.1s
```

`active_tasks` is 22-45 per engine and `done_active_tasks=0` — nothing drains during the hang.

### Leading hypothesis (still the sleep/pause one)

`LLMRayActor.sleep` calls `self._run_async(self.llm_engine.sleep(level=0, mode="keep"))`. Inside vLLM engine-core, `mode="keep"` sets `PauseState.PAUSED_ALL`; if `has_work()` is true at pause time, the engine registers an idle-state callback that fires only when `has_work() == False`. But `PAUSED_ALL` stops the scheduler from stepping while keeping requests alive, so `scheduler.has_requests()` stays true indefinitely and the callback never fires. `_invoke_utility_method` then defers the RPC reply via `add_done_callback` on the returned Future; the reply never ships.

The first `sleep` RPC (step 1 pre-sync) returns synchronously because the scheduler is empty at startup, which is why we see `engine.sleep.remote() done`. By the time `update_weights` arrives, the actor has queued real generation work that keeps the scheduler non-empty.

### Why grpo_fast.py doesn't hit this (unresolved)

grpo_fast.py runs reliably with the same `LLMRayActor.sleep → update_weights → wake_up` machinery and the same `--inflight_updates --async_steps 4` config. That rules out the sleep mode itself as the fundamental issue — something sequence-level about how grpo_fast.py calls the actor must let the scheduler drain before (or during) sleep. Next concrete step is to trace grpo_fast.py's pre-weight-sync sequence and compare it to `VLLMWeightSyncCallback.post_step`.

### Fixes considered but rejected

- **`sleep(mode="abort")`**: would call `scheduler.finish_requests(..., FINISHED_ABORTED)` and reply immediately. Rejected by user: if grpo_fast.py doesn't need to abort in-flight work to unhang this, grpo.py shouldn't either — the fix should target the actual grpo.py-specific difference, not paper over it.
- **Drain active tasks before calling sleep**: same objection — grpo_fast.py doesn't pre-drain.

### Hypotheses diagnosed and closed

- **FSDP2 DTensor passed to NCCL send**: closed. `[DIAG] pdata_type=Tensor` on every trainer param (rank 0) after `block.unshard()`. Total numel `4,411,424,256 × 2 bytes = ~8.8 GB` matches the Qwen3-4B-Base footprint in bf16; no truncated-shard send.
- **`olmo_core_to_hf_name` fallback leaking unmapped names**: closed. `[DIAG] _prepare_params_for_sync: 399 params, total_numel=4411424256, unmapped_count=0` — every OLMo-core param mapped cleanly into an HF name. No fallback path taken.
- **Param-set count mismatch between OLMo-core `Transformer` and vLLM Qwen3**: closed. Both sides see exactly 399 names with matching shapes.
- **NCCL recv dtype mismatch (Bug #2)**: fixed as above.
- **NCCL recv on engine side not matching the send (other causes)**: send completes in ~110 ms; no NCCL heartbeat timeout. Not NCCL-level.
- **FSDP2 `reshard()` freeing buffer before async NCCL send completes**: `_prepare_params_for_sync` clones into independent memory; reshard can't free what NCCL is reading. Would manifest as corruption, not a hang.

## Self-inflicted diagnostic regression (logged for honesty)

An earlier iteration of the engine-side diagnostics added `async def _update_weights_async_with_diag` as a method on `LLMRayActor`. Ray treats any class with an `async def` method as an async actor and spawns it on an asyncio event loop, which violates `assert_threaded_actor(self)` at the top of `LLMRayActor.__init__`. All 4 engine actors died during creation in experiment `01KQ02SA4W920XJW622J7XDSQN`:

```
AssertionError: LLMRayActor must run in a threaded Ray actor (no running event loop).
Detected RUNNING loop=<uvloop.Loop ...> on thread='AsyncIO Thread: default'.
```

**Fix** (commit `a5f8f483e`): moved the async wrapper out of the class to a module-level coroutine `_update_weights_coroutine_with_diag(llm_engine, request, model_step)` that `update_weights` schedules via `asyncio.run_coroutine_threadsafe`. Note for future diagnostics on this actor: any `async def` method added to `LLMRayActor` will flip Ray into async-actor mode and break the threaded-actor assertion.

## Current state

- Bug #1: fixed (commit `531a363f6`).
- Bug #2 (dtype mismatch): fixed (commit `1fda6231a`).
- Bug #3 (sleep/pause idle-callback never fires): **OPEN**. Hang reproduces cleanly with correct metadata. Next step is to compare grpo.py's and grpo_fast.py's weight-sync call sequences to find the grpo.py-specific trigger rather than patch over with `mode="abort"` / pre-drain.

## Files touched while debugging

- `open_instruct/grpo_callbacks.py` — phase markers in `VLLMWeightSyncCallback.post_step`; removed the unnecessary entry barrier (kept exit barrier).
- `open_instruct/vllm_utils.py` — phase markers in `broadcast_weights_to_vllm`; Bug #1 fix; Bug #2 fix (post-unshard metadata); DIAG heartbeat/param logging instrumentation; module-level update_weights coroutine (async-method regression fix).
- `scripts/train/qwen/qwen3_4b_dapo_math_oc.sh` — env vars for log visibility (`PYTHONUNBUFFERED`, `RAY_DEDUP_LOGS=0`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180`); removed `TORCH_LOGS`.

## Relevant experiments

- `01KPY3CFBFGNQFJ9VJEXB0PSX0` — hung on Bug #1 (before fix).
- `01KPY63Z27H7EG4YDWE8TF79WM` — hung on Bug #1 (with `RAY_DEDUP_LOGS=0`, confirming diagnosis).
- `01KPY7F5H65YMHJ5FNFP9MG2H3` — hung post-Bug #1 (what was then called Bug #2; now understood as dtype mismatch + sleep-pause).
- `01KPZVW93Z4EJXFNPEP3ZKZ3FA`, `01KPZWYYEQ9S68Z9Z4SXVJ1E3Y` — failed early on `ENOSPC` on shared Weka cache; infra-only.
- `01KQ02SA4W920XJW622J7XDSQN` — failed on the self-inflicted `async def` regression in `LLMRayActor`.
- `01KQ04CX5NTZH30WQ95JW753JP` — diagnosed Bug #2 (dtype mismatch): DIAG showed trainer `bfloat16` vs engines `float32`.
- `01KQ06VPYV0PKN514MAKMPPS2M` — Bug #2 fix verified (engines now see `bfloat16`); Bug #3 (sleep-pause) still hangs.
