# Multi-node automatic resume qualification

September 22, 2026. Closes the support-matrix gate "Multi-node/managed automatic restart is
configurable but remains unqualified", and resolves a contradiction: the guides said
multi-node runs require `launch.auto_resume=false`, nothing in the code enforced that, and
the maintained `medium.toml` shipped two replicas with `auto_resume = true`.

**Result: a two-replica run that is interrupted and started again from the same run
directory resumes from its newest native checkpoint, continues from the saved rollout
cursor rather than from zero, and finishes.**

## What was run

Two legs, same immutable image `01M33EBM5Z6SN0VNKH70A74ZPG` from source `37326d295`, same
run file, same output root. Two replicas of one GPU: one holds the trainer, the other the
policy engine, so the rendezvous, the cluster bootstrap and the resume are all real.
Tiny Olmo MoE, 32 GSM8K prompts, six updates, a native checkpoint after every update,
`checkpoint_keep_last = 2`, `auto_resume = true` and `max_retries = 3`.

| Leg | Experiment | Outcome |
|---|---|---|
| First | [01M34XG0CYP9JG5XVY9E0K3VBP](https://beaker.org/ex/01M34XG0CYP9JG5XVY9E0K3VBP) | Completed optimizer steps 1 to 3, rollouts 0 to 2, then cancelled deliberately |
| Second | [01M34Y06WCS7K4FTX9ZN52PCWD](https://beaker.org/ex/01M34Y06WCS7K4FTX9ZN52PCWD) | Began at optimizer step 4, rollout 3, ran to step 6 and exited zero |

The step counter continued at four rather than restarting at one, so the completed-step
clock was restored and not reset.

## Evidence

Checked by `runs/multinode-resume-20260921/verify.py` against the second leg's retained
artifacts, which carry the whole run directory because both legs share it.

| Check | Value |
|---|---|
| Starts recorded in `attempts.json` | 2, numbered 0 and 1 |
| Final workflow status | complete, recorded attempt 1 |
| Resume policy in the saved specification | `auto_resume` true, `max_retries` 3 |
| Checkpoint the second leg loaded | the run's own `checkpoints` directory |
| Optimizer rollouts recorded across both legs | 0, 1, 2, 3, 4, 5, each exactly once |

No rollout was repeated and none was skipped, which is the part that matters: a resume
that replayed or skipped an update would corrupt the training sequence while still
finishing.

## Boundaries

The interruption was a deliberate cancellation, not a forced preemption. From the
application's side the two are the same, because Beaker's automatic restart re-runs the
same command against the same output root, which is exactly what the second leg did.
**Forced multi-node preemption remains unqualified**, and Beaker's own restart of both
replicas together has been observed once, on a preempted multi-node MoE run recorded in
[checkpoint-no-drain-20260915](checkpoint-no-drain-20260915.md), rather than deliberately
exercised.

The model is tiny and randomly initialised, so every group had constant rewards and every
gradient norm was zero. This is a lifecycle qualification, not learning evidence. One GPU
per replica says nothing about EP8, larger trainers or the 32K mixed workload.

An earlier attempt at the same exercise,
[01M33EC0FWECVN9RQ06E8WSMB5](https://beaker.org/ex/01M33EC0FWECVN9RQ06E8WSMB5), was
cancelled about a minute after its replicas met, before any optimizer step, by a faulty
watcher in the operator's tooling. Nothing is concluded from it.
