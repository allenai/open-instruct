# Checkpoint without inference drain: implementation and qualification

Primary `robertb/miles-olmo-core` includes the learning-confidence campaign through
merge `a50ced7f9`, pushed September 15 UTC. Its focused checks passed: 50 MILES
reward/judge/baseline tests, 20 historical Open Instruct comparator tests, and
Ruff checks/format verification over the MILES modules/scripts/tests. Other work
in the primary worktree was preserved.

The separate branch `robertb/miles-checkpoint-no-drain`, implementation
`75073fce1`, removes checkpoint quiescence from the driver for refresh and rolling
publication. It retains cursor-save, native trainer-save, final-commit ordering.
There are no native tensor format, optimizer, or OLMo-core kernel changes.

The prerequisite already existed: the async data source maintains pristine
pending prompts and snapshots them together with the dataset cursor under a
lock. Both incomplete and completed-but-unconsumed groups regenerate on resume;
responses are not persisted in this ledger. Normal in-memory generation continues
during save. This is a trainer/prompt-accounting guarantee, not exact reproduction
of an uninterrupted asynchronous sampling trajectory. Evaluation/export/shutdown
retain their existing lifecycle behavior.

## Local qualification

138 focused tests passed in `open-instruct-miles-core-298254e50a1e`:
`test_lifecycle`, `test_async`, `test_checkpoint_topology`,
`test_checkpoint_options`, `test_async_publication_boundary`, and
`test_policy_refresh_runtime`. Host libcuda was mounted for runtime imports;
these were CPU tests, not GPU training. Added cases exercise an unfinished
request during save, inference progress during the trainer write, atomic cursor
snapshot versus concurrent admission, post-snapshot mutations, correct pending
prompt regeneration, and failures at cursor/model/commit stages. Ruff passed.

## Live qualification: pending

[Beaker experiment](https://beaker.org/ex/01M2HTV62B7TBJRV763P9Y239J), job
`01M2HTV65VEYM2GCTAVZVT98F5`, uses immutable image
`01M2CJG5RQQ93GEYNYAS7ASCQJ` with committed overlay `75073fce1`.
Submitted through `build_image_and_launch.sh --miles` and
`launch_policy_refresh_trial.sh`: EP2 trainer plus two TP1 inference engines,
four initial updates with saves/evaluation, followed by a fresh-process resume
for the fifth update. Holmes, urgent, one-hour minimum runtime, 90-minute ceiling.

At 06:09 UTC it was queued. The scheduler reported the workspace group using
157/160 allowed slots; this job requires four. No GPU result is claimed yet.
Acceptance still requires actual completed native saves, a restored update clock
and successful next update. Inspect cursor pending groups, checkpoint stage
timings, and serving activity during writes; no `checkpoint_drain` stage should
occur. Retain GPU native same-batch resume equality checks as a separate numerical
qualification; this live async run does not reproduce identical sampled batches.
