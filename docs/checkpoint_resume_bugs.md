# Checkpoint Resume Bugs

Discovered from experiment `01KGTJAPR9MC7CMN637A870Z2P` (hybrid 7B DPO run).
The job crashed at step 2000 of 2031, retried from scratch, and wasted ~25 hours of compute.

## Bug 1: Race condition in checkpoint cleanup

**File:** `open_instruct/utils.py`, `clean_last_n_checkpoints`

All ranks call `shutil.rmtree` on the same checkpoint directory simultaneously.
One rank succeeds, and the others get `FileNotFoundError` because the directory is already gone.
This crashes the entire job.

**Fix:** Either gate the cleanup to rank 0 only, or catch `FileNotFoundError` in the `shutil.rmtree` call.

## Bug 2: Output directory changes on retry due to timestamp

**File:** `open_instruct/dpo_tune_cache.py`, line 168

The output directory includes `int(time.time())` in the path:
```
{exp_name}__{seed}__{int(time.time())}
```

When Beaker retries a failed job, `time.time()` returns a new value, so the retry
writes to a completely different output directory and never finds the previous checkpoints.

- First run:  `.../hybrid-7b-DPO-SFT-8e-5-1e-6__42__1770418359`
- Retry:      `.../hybrid-7b-DPO-SFT-8e-5-1e-6__42__1770560004`

**Fix:** Make the output directory deterministic (e.g. use a hash of the args instead of
wall-clock time), or pass an explicit `--output_dir` that doesn't change on retry.

## Bug 3: Silent failure to resume from checkpoint

**File:** `open_instruct/utils.py`, `get_last_checkpoint_path`

When the output directory exists but contains no valid checkpoint, the code logs a warning
and silently starts training from scratch:
```
WARNING - Output directory exists but no checkpoint found. Starting from scratch.
```

There is no way to distinguish an intentional first run from a failed retry that lost its
checkpoints. The job happily re-trains from step 0 with no error.

**Fix:** Add a flag or heuristic to detect that this is a retry (e.g. check if wandb run
already has logged steps, or require explicit `--overwrite_output_dir` to start fresh when
the directory already exists).
