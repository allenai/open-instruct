# Native checkpoint performance qualification

Isolated branches: `robertb/miles-checkpoint-perf` in open-instruct and OLMo-core.
Bases: open-instruct `12235d54d`, Core `307d20590`. Parent worktrees remain untouched.

The initial trial preserves production defaults. Candidate save options are explicit harness
arguments, not new run-profile defaults:

| Mode | CPU storage | Replicated owner | Writer |
|---|---|---|---|
| baseline | unconditional clone | lowest rank | existing thread default |
| compact | clone only noncompact backing storage | lowest rank | existing thread default |
| balanced | compact storage | PyTorch planner balances | existing thread default |
| processes | compact storage | PyTorch planner balances | 2 spawned workers, 2 buckets |

Contiguity is insufficient to skip cloning: storage offset must be zero and backing storage
bytes must equal logical tensor bytes. A contiguous prefix otherwise serializes unrelated
storage. The shared writer defaults remain unchanged.

Profile fields distinguish state-dict construction, persistent buffers, local/global planning,
DCP total, writer wall time, coordinator metadata, optimizer reload and cache clearing. Per-item
resolve, blocking CPU copy, clone, serialization, flush and rename/upload durations are **summed
worker times**, which can exceed writer wall time. They do not by themselves establish GIL
contention. GPU boundaries synchronize only when profiling is requested.

The direct save now restores optimizer storage in a `finally` block if buffer preparation or
DCP writing fails. Construction of the optimizer state dict itself remains the optimizer's
responsibility. Asynchronous saves are deferred: the current state-dict method releases some
live storage and returns views, which cannot safely be mutated by subsequent training.

## Checks completed locally

- CPU writer/reshard and compact-storage tests: 11 passed, 4 GPU skips.
- Actual toy MoE on RTX 4090: both balanced/compact threads and spawned processes passed native
  save, live-state retention, HF-file content verification, fresh-process restore, and exact
  next-two-update model/optimizer state and log-probabilities.
- Toy writer saves: approximately 0.07 seconds with threads and 9.31 seconds with processes;
  process startup dominates a 9 MB checkpoint. These are correctness checks, not evidence about
  full-model performance.
- Open-instruct `make style quality` passed using the existing environment and Core source path.
- Changed Core files pass isort, Black and Ruff. Repository-wide `make checks` stops on existing
  import-order failures in unrelated hero example/attention/objective files.

## Full-model gate

`build_image_and_launch.sh --miles scripts/train/debug/miles_checkpoint_benchmark.sh`
launches two B300 GPUs on Holmes, urgent, open-instruct-dev, minimum runtime 60 minutes, maximum
3 hours. Native checkpoints and HF exports remain under a fresh WEKA experiment directory;
small reports are copied to Beaker results. Existing topology tests run first, including
EP1-to-EP2 save/load and both writer policies (EP4/8 require larger allocations).

Each arm trains two fixed batches, saves and exports, then continues for two more batches in
the **same process**. A fresh process loads that exact checkpoint, exports, and consumes the
same next batches. Compare all local master/moment/step bytes, model bytes, scheduler, clock,
data cursor, RNG, and scoring log-probabilities. The two exported HF artifacts must also match.
This avoids comparing two independently initialized/autotuned training trajectories.

Acceptance requires exact correctness plus max rank save time <= 340 seconds on the full SFT
model. Synthetic responses isolate the checkpoint contract; serving and GSM8K learning are
outside this experiment. No candidate is promoted based only on the toy test.
