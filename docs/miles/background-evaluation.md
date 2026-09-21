# Best-effort background olmo-eval

**Publication is currently blocked:** live qualification showed that W&B shared
writers reopen finished runs and replace metric definitions. Snapshots, external
evaluation and retained results work; the requested W&B lifecycle contract has
not passed. See the [qualification record](measurements/background-evaluation-20260920.md).

Opt in with `evaluation.mode="background"`. Training captures frozen HF weights,
then one driver submits independent Beaker jobs in a daemon worker. Evaluation
never borrows, drains or pauses rollout engines, and shutdown never joins the
submitter or waits for evaluator jobs. Missing points are expected: a busy
submitter drops new milestones, failures are not retried, and preemption can
interrupt submission. Snapshot export is a synchronous trainer collective;
checkpoint/collective failures still fail training.

Existing shared-engine evaluation remains the default. Remove explicit shared
settings (`training.eval_interval`, `miles.eval_*`, `skip_eval_before_train`,
`n_samples_per_eval_prompt`, `data.eval_prompt_data`, positive task `eval_count`)
when selecting background mode. Evaluation tasks are independent of training
`data.tasks`; no evaluation-only dataset needs to be added to the training mix.

```toml
[evaluation]
mode = "background"
interval = 20                 # Completed optimizer updates, not collections
initial = true
final = true
image = "IMMUTABLE_BEAKER_IMAGE_ID" # Replace with a qualified evaluator image
revision = "FULL_40_CHARACTER_OLMO_EVAL_COMMIT"
gpus = 1                     # Separate allocation; never training GPUs
cluster = "ai2/holmes"
submit_timeout = 30           # Hard wall-clock bound on a submission subprocess
timeout = "3h"                # Evaluator Beaker task lifetime
server_args = ["--attention-backend", "triton", "--disable-cuda-graph"]

[[evaluation.tasks]]
task = "gsm8k"
generation = { temperature = 0.0, max_tokens = 2048 }
scoring = { limit = 128 }

[[evaluation.tasks]]
task = "ifeval"
interval = 100
scoring = { limit = 128 }

[launch.secrets]
BEAKER_TOKEN = "your-beaker-token-secret"
WANDB_API_KEY = "your-wandb-secret"

# Optional task-specific credentials. These are secret references, never tokens.
[evaluation.secrets]
OPENAI_API_KEY = "your-judge-secret"
```

Workspace and budget default to `launch.workspace` and `launch.budget`. Evaluator
jobs inherit `launch.weka_mounts`, `WANDB_API_KEY` and `HF_TOKEN` secret references.
`evaluation.secrets` can override these or supply task-specific credentials. The
submitter needs Beaker credentials in the trainer allocation; evaluator jobs do
not inherit its Beaker token. Outputs and snapshots must be on a WEKA mount
accessible to both jobs. Inherit the main run's W&B entity, project and actual run
ID; never configure a separate evaluation run.

Generation overrides map to olmo-eval's flat per-task sampling overrides
(for example `-o max_tokens=2048`, not a nested `sampling_params` table); scoring maps to
olmo-eval task overrides (such as `limit`, scorer settings, or formatting).
Tasks with the same generation settings share a job; distinct settings form
separate jobs at that update. One HF snapshot serves all groups. A coincident
periodic/final point is submitted only once for each task group. `plan` displays
the milestones, task groups, resources and snapshot paths. `validate` checks the
schema; it does not establish image/model/task compatibility.

## Snapshots and receipts

Snapshots live at `OUTPUT/eval-snapshots/update-NNNNNNNN/hf`, including model
configuration, tokenizer and chat template. A `.complete` marker gates submission.
Update zero reuses the prepared initial HF checkpoint and records a reference
under `eval-snapshots/update-00000000/reference.json`; retain its source weights
as well. Snapshot paths are exclusive and never overwritten. An incomplete
snapshot from an interrupted export is retained and skipped on resume.

`OUTPUT/evaluation/update-UPDATE-GROUP.json` records checkpoint/update, task
configuration, evaluator image/revision, runner hash, training identity, and the
Beaker experiment ID or diagnostic. All recorded attempts suppress automatic
resubmission, including ambiguous timeouts and interrupted `pending` receipts.
No backlog or polling service runs in training. Submission failures produce
`BACKGROUND EVALUATION GAP` warnings. Evaluator failures appear in their Beaker
job and result directory and leave a gap in the dashboard.

Results are retained in `OUTPUT/evaluation/results/update-UPDATE-GROUP/` and
copied to the evaluator's Beaker results: aggregate `scores.json`, olmo-eval
`metrics.json`, predictions/requests, logs, command/configuration and provenance.
A failed W&B upload writes `publication.json` and preserves these files.

## W&B publication

The candidate publisher uses W&B's documented
[shared-mode secondary writer](https://docs.wandb.ai/models/track/log/distributed-training)
with `x_primary=False` and `x_update_finish_state=False`, the exact training run
identity, and an explicit `eval/checkpoint_update` axis without an internal
`step`. Live testing produced the correct curves and preserved training history
and public configuration, but reopened the finished run and replaced its metric
definitions. SDK 0.30.0 reproduces this behavior.

Publication is therefore blocked **before attaching**, for both automatic and
manual online publication. The evaluator saves scores, predictions, requests,
configuration and provenance, and writes `publication.json` with status `blocked`
and the reason. It does not restore run state after logging, which would race a
training writer. A supported state-preserving publication mechanism remains a
release blocker; do not use the earlier qualification image to bypass this gate.

Offline training still records publication as `deferred`. After training is
synced and the lifecycle blocker is resolved, the manual interface is:

```bash
python -m open_instruct.miles.evaluation_runner publish RECEIPT.json \
  --results /path/to/evaluation/results/update-UPDATE-GROUP \
  --wandb-run ENTITY/PROJECT/RUN_ID
```

The command currently fails safely with the same durable blocker. Results are
retained for later publication. Repeating publication once enabled can append
duplicate points; inspect `publication.json` first.

## Manual resubmission and cleanup

Inspect the receipt and Beaker experiments before resubmitting: a timeout may
mean the API accepted the job but the trainer missed its response. Check the
latest job attempt; if queued, inspect `beaker job events JOB_ID`.

```bash
python -m open_instruct.miles.evaluation resubmit RECEIPT.json
```

This creates a new receipt and results group, retaining the original diagnostic.
It requires the committed runner source matching `runner_sha256`, the recorded
image, secrets, mounts and completed snapshot. There are no automatic retries.

Snapshots are excluded from ordinary checkpoint retention and result collection.
There are no pins, leases, automatic deletion or retention controller. Manually
remove selected `eval-snapshots/update-NNNNNNNN` directories only after verifying
that **all** jobs referencing them have stopped (including manual resubmissions).
Do not delete snapshots while jobs are pending or running. Keep small receipts
and results as provenance. Initial checkpoint reuse additionally depends on the
original HF source remaining available.

## Evaluator image and qualification

`runtime/miles/Dockerfile.evaluator` builds a separate evaluator image from a
qualified MILES serving image and a pinned olmo-eval revision. It retains the
Olmo MoE SGLang architecture extension, installs olmo-eval in a separate CPU
virtual environment, and connects olmo-eval's `vllm_server` API client to its own
private SGLang server. It does not launch vLLM. The runner checks the olmo-eval
revision before starting. Build dependencies at image construction, never during
an evaluation job. Publish the image to Beaker and configure its immutable ID.

Compatibility must be qualified for the exported architecture, serving backend,
task and image; an OpenAI-compatible endpoint alone does not prove loglikelihood
or task correctness. Use generation tasks for the initial tiny MoE check. A
successful mechanics check is not evidence of model quality. Follow the normal
committed-image MILES wrapper for training qualification. Use ignored `runs/`
configs and fresh output paths; leave existing experiments unchanged.

Live GPU results and the W&B lifecycle blocker are recorded in the
[September 21 qualification](measurements/background-evaluation-20260920.md).
