# NCCL Timeout Investigation Plan

## Context
- Experiment `01K6FYH98Z7BS2BDFTM1P0972N` (10/01/2025) on branch `async-older` crashed with a 600 s NCCL watchdog timeout during learner step 3.
- PyTorch 2.7 raised new warnings (`ProcessGroupNCCL.cpp:4715`) about unknown device mapping; failure reproduced only on multi-node / multi-GPU runs.

## Goals
1. Reproduce the timeout deterministically.
2. Identify whether rank↔device mapping or async scheduling causes the unsynchronized all-reduce.
3. Land minimal fixes and guardrails.

## Investigation Steps
1. **Confirm repro envelope**
   - [ ] Re-run current config with verbose NCCL logging (`TORCH_NCCL_DEBUG=INFO`, `NCCL_DEBUG=INFO`, `TORCH_NCCL_TRACE_BUFFER_SIZE=67108864`). See “Run Instructions” for command skeleton.
   - [ ] Run reduced configs (single learner per node, single node multi-GPU) to see first failing combination.

2. **Instrument learner device mapping**
   - [ ] In `PolicyTrainerRayProcess.from_pretrained`, log `self.rank`, `self.local_rank`, `CUDA_VISIBLE_DEVICES`, `torch.cuda.current_device()`.
   - [ ] Pass explicit `device_id=torch.cuda.current_device()` (or `self.local_rank`) into `deepspeed.init_distributed` and rerun.
   - [ ] Compare logs against main branch to ensure mappings match.

3. **Validate minibatch alignment**
   - [ ] Before `ray_get_with_progress` call in `PolicyTrainerRayProcess.train`, log per-rank `len(collated_query_responses)`, `accumulation_steps`, `num_mini_batches`.
   - [ ] If any rank reports zero minibatches, inspect packing thread (`B = len(...) // world_size`) and adjust to avoid dropping data.

4. **Check weight-sync pauses**
   - [ ] Instrument `weight_sync_thread` to log when `set_should_stop(True/False)` fires and insert a `torch.distributed.barrier()` after broadcasts.
   - [ ] Ensure generate thread resumes and all learners exit the barrier before training resumes.

5. **Capture NCCL stack traces if needed**
   - [ ] If timeout persists, enable `TORCH_SHOW_CPP_STACKTRACES=1` and re-run to collect offending call stacks.
   - [ ] Optionally set `NCCL_ASYNC_ERROR_HANDLING=1` to force early aborts for clearer diagnostics.

6. **Patch & verify**
   - [ ] Land device-id fix and any batching adjustments.
   - [ ] Add sanity checks (assert matching batch counts across ranks, warn if not) to prevent silent divergence.
   - [ ] Run soak test on target configuration to confirm no further NCCL timeouts.

## Run Instructions
- Launch Beaker job with additional environment overrides:
  ```bash
  beaker experiment create <spec.yml> \
    --env TORCH_NCCL_DEBUG=INFO \
    --env NCCL_DEBUG=INFO \
    --env TORCH_NCCL_TRACE_BUFFER_SIZE=67108864 \
    --env NCCL_ASYNC_ERROR_HANDLING=1
  ```
- For reduced configs, adjust `--num_nodes` / `--num_learners_per_node` arguments passed through `large_test_script.sh` and note whether the NCCL warning still appears.
- After each run, grep for `[distributed_init]` and `[train_setup]` in learner logs to confirm consistent rank/device mappings and minibatch stats.

## Deliverables
- Log bundles demonstrating root cause & fix.
- PR with device-id fix, logging, and batch-alignment guardrails.
- Follow-up issue if further reliability work is needed.
