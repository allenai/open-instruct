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

## Bug #2 — vLLM `update_weights` RPC never returns (OPEN)

After the Bug #1 fix, rank 0 advances to `trainer_send_weights done` (in 110ms — the NCCL broadcast completes) and then `reshard done` → `broadcast_weights_to_vllm returning` → `ray_get(refs) start`. Ranks 1/2/3 complete their (empty) `ray_get` and sit at `wake_up start`. Then everything hangs. NCCL heartbeat timeout (180s) never fires — so no NCCL collective is actually pending. The vLLM engines' `update_weights.remote()` ObjectRefs simply never resolve; queued `wake_up.remote()` calls on each engine actor sit behind them and also never fire.

### Leading hypothesis

`LLMRayActor.sleep` (`open_instruct/vllm_utils.py:782-783`) calls

```python
self._run_async(self.llm_engine.sleep(level=0, mode="keep"))
```

Inside vLLM's engine-core (`vllm/v1/engine/core.py::pause_scheduler`), `mode="keep"` sets `PauseState.PAUSED_ALL` and, if `has_work()` is true at the time of the pause, registers an idle-state callback that fires only when `has_work() == False`. But `PAUSED_ALL` stops the scheduler from stepping while *keeping* queued requests alive, so `scheduler.has_requests()` remains true and the callback can never fire. `_invoke_utility_method` then defers the RPC reply via `add_done_callback` on the returned Future. The first `sleep` call happens to return synchronously (empty scheduler at startup), which is why rank 0 logs `engine.sleep.remote() done`. The subsequent `update_weights` RPC lands while residual scheduler state prevents the idle callback, and the deferred reply never ships.

### Proposed fix (not yet tried)

Change `open_instruct/vllm_utils.py:783` from `mode="keep"` to `mode="abort"`:

```python
def sleep(self) -> None:
    return self._run_async(self.llm_engine.sleep(level=0, mode="abort"))
```

With `mode="abort"`, `pause_scheduler` calls `scheduler.finish_requests(..., FINISHED_ABORTED)`, drains the scheduler, returns `None` synchronously, and the RPC replies immediately. `actor_manager.set_should_stop(True)` is already called before `broadcast_weights_to_vllm` in `grpo_callbacks.py:102`, so aborting in-flight vLLM generations is safe — the data-prep layer has already stopped consuming.

### Hypotheses considered and rejected

- **NCCL recv on engine side isn't matching the send**: `trainer_send_weights` returned in 110ms (consistent with an 8 GB broadcast over NVLink). No NCCL heartbeat timeout ever fires. If NCCL were the blocker, the 180s heartbeat would have tripped.
- **FSDP2 DTensor `.clone()` sends a local shard**: after `block.unshard()` on a 1D FSDP2 mesh, `param.data` is a plain `Tensor`, not a DTensor. `.contiguous().clone()` returns a full local tensor. Ruled out by both code reading (pytorch `_fsdp_param.py`) and by the fact that the broadcast completed in ~110ms for the full model (~8 GB).
- **`reshard()` deallocating unsharded buffer before async NCCL send completes**: `_prepare_params_for_sync` at `vllm_utils.py:1330` clones into independent memory, so reshard doesn't free what NCCL is reading. Would also manifest as corruption, not a hang.
- **Param ordering / name mismatch between `_prepare_params_for_sync` and `_collect_weight_metadata`**: both iterate `model.named_parameters()` in the same order and apply the same `name_mapper`. No obvious divergence.

## Current state

- Bug #1: fixed and committed.
- Bug #2: diagnosed, fix proposed, **not yet tried**. Next step is to change the `sleep(..., mode="keep")` call to `mode="abort"`, relaunch `scripts/train/qwen/qwen3_4b_dapo_math_oc.sh`, and confirm rank 0 advances past `ray_get(refs)` into step 2.

## Files touched while debugging

- `open_instruct/grpo_callbacks.py` — phase markers in `VLLMWeightSyncCallback.post_step`; removed the unnecessary entry barrier (kept exit barrier).
- `open_instruct/vllm_utils.py` — phase markers in `broadcast_weights_to_vllm`; Bug #1 fix.
- `scripts/train/qwen/qwen3_4b_dapo_math_oc.sh` — env vars for log visibility (`PYTHONUNBUFFERED`, `RAY_DEDUP_LOGS=0`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180`); removed `TORCH_LOGS`.

## Relevant experiments

- `01KPY3CFBFGNQFJ9VJEXB0PSX0` — hung on Bug #1 (before fix).
- `01KPY63Z27H7EG4YDWE8TF79WM` — hung on Bug #1 (with RAY_DEDUP_LOGS=0, confirming the diagnosis).
- `01KPY7F5H65YMHJ5FNFP9MG2H3` — hung on Bug #2 (with the Bug #1 fix applied).
