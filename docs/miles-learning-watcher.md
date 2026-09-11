# Local learning-run watcher

`python scripts/miles/watch_learning_runs.py` observes the already-authorized Core500,
Megatron500, explicitly supplied Light Core200 experiment, and the authorized
scorer diagnostic `01M27BKYTF9N7JMYBTKAV0HS9A`. The scorer is capture-only and
does not gate submission or success of the paired learning audit. It polls every five
minutes for at most 24 hours from its original start, including after a restart.
It does not retrain, restart jobs, change priorities, delete artifacts, or send messages.

The output directory contains `state.json`, `process.json`, and an exclusive
`watch.lock`. Per-job directories retain immutable status snapshots, the final log,
and the Beaker result dataset. Environment values are excluded from status snapshots.

Only two successful finalized 500-update jobs permit audit submission. The watcher
writes and fsyncs its submission intent before invoking the standard committed
`build_image_and_launch.sh --miles ... --stage audit` wrapper. The existing launcher
places this CPU/WEKA audit on Saturn with a positive minimum runtime. Failed arms
never trigger an audit. A crash or ambiguous submission result requires manual
inspection; it is never automatically resubmitted. Read-only terminal artifact capture retries at most three times in separate staging
directories, promoting only a complete download. Exhausted capture attempts or
failed analysis leave an explicit attention record.

After a passing paired audit, the watcher invokes the existing analyzer in an
immutable local Docker image, with the pinned committed analysis source mounted
read-only. `analysis/` contains the comparison JSON, plot, brief Markdown report,
and the two independently verified audit inputs. Allocation durations come from
actual Beaker scheduled/exited timestamps. Light-SFT output is retained separately;
it does not enter the heavy-SFT comparison.

Run from a clean detached checkout using explicit paths and the approved light-run ID:

```bash
python scripts/miles/watch_learning_runs.py \
  --checkout /tmp/miles-learning-watcher-run-20260911 \
  --output /home/robert/proj/open-instruct/.artifacts/miles-learning-watch-20260911 \
  --analysis-image sha256:10b1a44ac7c871d84e0e1cdb3a3f8c681ed95921d83fa266584dd1a73ff4cbef \
  --base-image olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks \
  --light-experiment 01M2794XP15PHQSWN6M499NQYS
```

For a local restart, use exactly the same arguments and output directory. The lock
rejects concurrent watchers. An existing audit ID is resumed as an observation task;
no second audit is created. If `state.json` reports an ambiguous submission, inspect
`submission.log` and Beaker before any manual action. Do not clear the intent merely
to make the watcher submit again. A changed light-run identity or changed checkout
is rejected instead of silently replacing recorded provenance.

This process needs the local host, Docker, Beaker credentials, and filesystem to
remain available. Its PID and last successful poll are inspectable locally. It is
not a remote service and cannot survive host shutdown without an explicit restart.

An explicitly authorized light retry uses a separate output directory and
`--observe-only --light-experiment RETRY_ID --previous-watch-output ORIGINAL_OUTPUT`.
This sidecar captures only that experiment and cannot submit an audit, including
after restart. The original watcher remains the sole paired-audit owner, retaining
its immutable configuration and the failed original light run's evidence. The
sidecar records the original output path as provenance; it never edits that history.
