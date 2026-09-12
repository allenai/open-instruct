# Operations and troubleshooting

## Establish completion

Use `python -m open_instruct.miles status run.toml`, then inspect all current
Beaker task attempts. A zero exit status, completed workflow state, expected
update/eval/checkpoint counts and retained audit evidence together establish what
finished. A run appearing in W&B or generating responses does not prove updates,
weight publication or clean shutdown succeeded.

| Artifact | What to inspect |
|---|---|
| Local launch receipt | Submitted specification, image ID, submitter revision, allocation and experiment |
| `workflow.json` under output.root | Preparing/training/complete/failed state; a killed job can leave stale state |
| `prepared/` | Model descriptor, immutable data preparation and verifier provenance |
| `rollouts/` | Per-collection samples, behavior versions, rewards and generations |
| `checkpoints/` | Completed native optimizer/model checkpoints for resume |
| `wandb/` | Offline W&B files or online tracking state |
| `driver_timing.jsonl` and contract reports | Stage timings, policy agreement, replay and optimizer/publication diagnostics |
| Beaker `/output` results | run.log, submitted-run.json and bounded JSON/log copies; large tensors stay on WEKA |

Use the exact artifact paths reported by the run; low-level/historical harnesses
can choose different subdirectories. `status` reports attempts and config identity,
not live WEKA metrics. Preserve receipts or set MILES_LAUNCH_RECEIPTS when moving
between laptop and session.

## Interpret measurements

Report rollout/evaluation latency, scoring, optimization, publication, checkpoint,
and orchestration costs separately. Async stages overlap: their durations cannot
simply be summed to derive wall time. Separate cold compilation/restore from warm
steady state, and compare tokens/second and GPU-seconds as well as update time.
Record generated length, cap fraction, mixed-reward groups, reward, lag and TIS
clipping to distinguish a workload change from a kernel speedup.

The policy-agreement guard named max_train_rollout_logprob_abs_diff measures the
**mean absolute active-token gap**, despite its legacy name. Replay diagnostics
check supplied expert IDs; native route agreement is a different experiment.
See [implementation contracts](core.md) for exact semantics and evidence limits.

## Failure triage

| Symptom | Next check |
|---|---|
| Configuration rejected | Error field/context, TOML types, conflicting aliases; rerun with --debug |
| Queued job | `beaker job events JOB_ID`; use the scheduler's reason and latest attempt |
| Import/model failure | Selected image provenance, lock, architecture/tokenizer descriptor |
| Out of memory | Trainer resident state, optimizer initialization, pack/context budget, actual KV/recurrent pools and graph capture |
| Engine unavailable | Per-engine logs, health/recovery events, published policy version and Ray actor state |
| Policy agreement/replay failure | Exact checkpoint/template, token alignment, precision, replay fields and version provenance |
| Save/resume failure | Completed-checkpoint marker, original run specification, output-root ownership and disk capacity |
| Slow shutdown | Compiler-cache phase records and cancellation result; do not attribute all delay to archive compression |

Do not bypass contract failures merely to finish a run. A healthy HTTP process is
not proof that it has the current weights. The runtime's health/recovery support
does not establish recovery of every Core trainer failure.

Compiler-cache publication is best effort. It stages locally, limits publishers
per node and uses a shared bounded publication wait; timeout must not turn completed
training into failure. See [cache guide](compiler-cache.md). Report metadata I/O
is outside that publication wait, so it is not an absolute shutdown deadline.
