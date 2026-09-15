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


## Recovery follow-up

The GPU qualification restored optimizer state and eight pending prompt groups
in a fresh process, then completed optimizer update 5 at 07:22 UTC September 15.
This qualifies continuation from the native checkpoint, not bit-exact async
sampling or a forced Beaker preemption. Final cleanup is tracked separately.

Campaign configs had explicitly disabled the launcher's default automatic resume.
The three Core campaign configs now enable it. An authenticated comparison of the
saved run specification permits changing only this recovery flag without relaxing
checks on model, data, objective or topology. The single-node GSM8K checkpoint
has a completed update-50 marker. Its first recovery submission selected an
incompatible newer image and failed before training; the campaign launcher now
rejects that image choice before allocating GPUs. The required overlay base is
`01M2CJG5RQQ93GEYNYAS7ASCQJ`.

Multi-node retries now rendezvous using fresh process identities for every rank
and receive a new shared coordination subdirectory. Staggered retries cannot
consume previous readiness, failure or completion markers. Old directories remain
as diagnostic evidence; wall-clock expiration is not used for correctness.
A missing restarted peer hits the existing startup deadline. Automatic preemption
recovery still relies on Beaker restarting the peer tasks together; this does not
introduce unlimited retries for application exceptions.


## Relaunches

The save/resume qualifier finished with exit code 0 at 07:32:49 UTC.
Nineteen focused recovery, rendezvous and campaign-launch tests passed; Ruff
checks passed. Forced multi-node preemption remains unqualified.

- Dense GSM8K resume: https://beaker.org/ex/01M2HZKS5S3EJTP6QY3KBERG7D
  (`dc1016c0c`, correct pinned base). Started at 07:37 UTC; restoration from the
  committed update-50 checkpoint is pending observation.
- MoE broad restart: https://beaker.org/ex/01M2J039655AS5BM5KZ9YQNQ3F
- Dense broad restart: https://beaker.org/ex/01M2J03FM72JY7CAWAXKDY7ATA

Both broad restarts use overlay `c1446a2b1` and the same pinned base. They retain
model/data/objective settings, enable automatic preemption recovery and use fresh
coordination state. Prior five-update attempts had no completed checkpoint;
their unsaved updates must not be stitched into the new learning curves.
At 07:40 UTC one dense replica remained pending because the workspace group had
156/160 slots occupied and the replica required eight. Other replicas were
scheduled or starting. Submission is not evidence of resumed training.
