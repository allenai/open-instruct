# Best-effort background olmo-eval

Opt in with `evaluation.mode="background"`. Training captures frozen HF weights,
then one driver submits independent Beaker jobs in a daemon worker. Evaluation
never borrows, drains or pauses rollout engines, and shutdown never joins the
submitter or waits for evaluator jobs. Missing points are expected: a busy
submitter drops milestones from the next collection, submission failures are not retried, and preemption can
interrupt submission. Submitted evaluator jobs automatically restart after
preemption. Snapshot export is a synchronous trainer collective;
checkpoint/collective failures still fail training.

Ready milestones from one collection are submitted as one batch, so they do not
compete with each other for the submission worker. A wall-clock-stop final
evaluation joins the last collection's batch, merging periodic and final tasks at
the same update. The worker submits accepted receipts sequentially; it does not
queue additional collections or wait at shutdown.

The [large example](../../configs/miles/examples/large.toml) opts into background
evaluation with a pinned sandbox-enabled image and the 25-entry post-training
development/text-safety panel. Its interval of 1,000 optimizer updates gives
initial and final evaluation for the template's 200-update run. Extending the run
also enables evaluation every 1,000 updates. Intervals remain positive integers;
omitting the field retains the default cadence rather than disabling it.
Dev and node retain shared-engine
evaluation, which also remains the low-level default. The large template is
provisional; check evaluator compatibility for the full model. Remove explicit
shared settings (`training.eval_interval`, `miles.eval_*`, `skip_eval_before_train`,
`n_samples_per_eval_prompt`, `data.eval_prompt_data`, positive task `eval_count`)
when selecting background mode. Evaluation tasks are independent of training
`data.tasks`; no evaluation-only dataset needs to be added to the training mix.

```toml
[evaluation]
mode = "background"
interval = 20                 # Completed optimizer updates, not collections
initial = true
final = true
image = "IMMUTABLE_BEAKER_IMAGE_ID" # Replace with a compatible evaluator image
revision = "FULL_40_CHARACTER_OLMO_EVAL_COMMIT"
gpus = 1                     # Separate allocation; never training GPUs
cluster = "ai2/holmes"
submit_timeout = 30           # Hard wall-clock bound on a submission subprocess
timeout = "3h"                # Evaluator Beaker task lifetime
server_args = ["--attention-backend", "triton", "--disable-cuda-graph"]

[[evaluation.tasks]]
task = "gsm8k"
generation = { temperature = 0.0, max_tokens = 2048, num_samples = 1, stop_sequences = [] }
scoring = { limit = 128, num_fewshot = 0 }

[[evaluation.tasks]]
task = "gsm8k:platinum"
generation = { temperature = 0.0, max_tokens = 2048, num_samples = 1, stop_sequences = [] }
scoring = { limit = 128, num_fewshot = 0 }

[[evaluation.tasks]]
task = "ifeval_ood" # IFBench; distinct from the IFEval task named "ifeval".
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

The limited GSM8K tasks above are small mechanics examples. The large template
uses individual task variants from `math:posttrain:dev`, `general:posttrain:dev`,
`hillclimb:dev` and `safety_thinking`, deduplicated to 25 entries. Individual names
avoid depending on the unmerged general/hillclimb suite definitions and do not
create synthetic cross-benchmark aggregate scores. It retains full task sizes
and sampling defaults from the pinned olmo-eval revision, including AIME/HMMT
pass@32 with 16K responses. The pinned IFBench defaults differ from newer upstream
revisions; changing the image/revision can change the benchmark protocol.

Generation overrides map to olmo-eval's flat per-task sampling overrides
(for example `-o max_tokens=2048`, not a nested `sampling_params` table); scoring maps to
olmo-eval task overrides (such as `limit`, scorer settings, or formatting).
Tasks with the same generation overrides, harness preset, harness overrides and
execution kind share a job. Empty generation overrides preserve each task's own
defaults; tasks execute separately while sharing the job's inference server.
The large panel creates five groups: 17 ordinary tasks, LiveCodeBench, SciCode,
five WildGuard-judged tasks and StrongREJECT. Each group requests one policy GPU;
judge service resources are additional. One HF snapshot serves all groups. A coincident
periodic/final point is submitted only once for each task group. `plan` displays
the milestones, task groups, resources and snapshot paths. `validate` checks the
schema; it does not establish image/model/task compatibility.

### Code and safety requirements

`harness_preset` selects an olmo-eval preset (default: `default`). The optional
task `harness` table passes flat harness overrides before the `--task` argument,
so auxiliary providers and sandboxes are not mistaken for scoring overrides.
The frozen checkpoint's `provider` settings cannot be overridden here. Harness
settings are recorded in receipts and checked when reusing completed tasks.

The large template selects `codex_python` for LiveCodeBench. Code jobs receive
Beaker's subcontainer environment flags and need an evaluator image containing
Podman and the olmo-eval sandbox dependencies. Container startup and reference-data
access still require qualification on the target cluster.

SciCode uses `kind="external"` and `arguments`, invoking olmo-eval's
`run-external` command against the same frozen SGLang server. Only SciCode currently
has a supported external artifact contract. Development runs require
`arguments.split="validation"` and an accessible `arguments.h5py_host_path`.
The evaluator checks per-problem predictions and step counts, rejecting
infrastructure errors even when upstream reports success with numeric zero scores.
SciCode's sandbox installs its pinned scientific dependencies from the upstream
lock; account for startup time and registry/package access when qualifying jobs.

Safety tasks use separately provisioned WildGuard and StrongREJECT endpoints.
Replace both `YOUR_*_HOST` placeholders before launching. The WildGuard endpoint
serves `allenai/wildguard`; the StrongREJECT endpoint serves Gemma-2B with the
`qylu4156/strongreject-15k-v1` adapter under the served name `strongreject` and uses
that adapter's tokenizer. These are evaluation judges, not the Qwen training
reward judge, and are not launched or allocated by this template. Endpoint model
versions and serving settings must be recorded with qualification results.

The image and CLI/config checks do not establish full-panel correctness, GPU
memory fit or completion within the starter three-hour timeout. Run small samples
of every task, inspect predictions and judge/code failures, then measure a full
panel before setting production resource bounds. For qualification, verify all
25 entries completed at both expected checkpoints and were published; training
completion alone does not establish evaluation completion.

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

### Recovery after preemption

Evaluator jobs use high priority, unallocated scheduling, and Beaker auto-resume.
On restart, the runner checks the existing output provenance against the
checkpoint, update, tasks/sampling/scoring, evaluator revision/image and server
configuration. It refuses incompatible results rather than overwriting them.
A process-owned filesystem lock prevents two evaluators from writing the same
group and is released when its holder dies.

Tasks execute separately while sharing one inference server. Each successful
task receives an atomic completion record after predictions and scores validate.
A restart skips those completed tasks. An unfinished task runs in a new attempt
directory; its previous partial files remain intact. Native olmo-eval response
resumption is not assumed, so an interrupted individual task is regenerated.
`task-results.json` selects only successful attempt directories for validation
and publication; abandoned attempts cannot introduce duplicate scores.

A fully completed group validates and reuses its results without loading the
model. A restart after the last task but before the group completion record also
finishes without loading the model. Publication may be retried, so duplicate
W&B points remain possible. Existing flat, completed result directories are
accepted only after the same artifact/coverage checks. The runner is embedded in
each Beaker spec, so retrying a spec runs the runner it was submitted with.

The runner rejects missing prediction files and empty `model_output` lists before
declaring evaluation complete. Some evaluator provider failures can otherwise
appear as zero scores with no failed-instance count. A returned EOS-only completion
is distinct from a missing provider response and remains a valid scored output.

## W&B publication

The publisher attaches to the exact main training run using W&B's documented
[shared-mode secondary writer](https://docs.wandb.ai/models/track/log/distributed-training)
with `x_primary=False` and `x_update_finish_state=False`. Late uploads may change
the dashboard status from `finished` to `running`; this accepted status change
does not restart training. The evaluator flushes its writer without declaring
an active training run finished. It does not restore or rewrite lifecycle state.

All MILES writers in background mode and every evaluator declare the same metric
schema: native training/rollout axes are retained, and `eval/*` uses the explicit
`eval/checkpoint_update` axis. This matters because W&B replaces metric-definition
metadata per writer. A common wildcard covers different task groups without
one evaluator erasing another group's axes. No internal W&B `step` is supplied,
so results can arrive out of checkpoint order. The publisher sends no training
configuration, name or group updates.

After a successful upload, `publication.json` records the destination and checkpoint
update. Resume skips an upload already confirmed for that same destination and
update; failed or deferred uploads can be retried. A crash after W&B accepts the
upload but before the local receipt is written can still cause a duplicate.

For offline training, sync the training run first, then publish to that existing
run with the manual command. The same command repairs a failed online upload:

```bash
python -m open_instruct.miles.evaluation.evaluation_runner publish RECEIPT.json \
  --results /path/to/evaluation/results/update-UPDATE-GROUP \
  --wandb-run ENTITY/PROJECT/RUN_ID
```

Uploads can change the dashboard status as described above. Failed uploads leave
`publication.json` and retain scores, predictions, requests, configuration and
provenance. Repeating publication can append duplicate points; inspect
`publication.json` first.

## Manual resubmission and cleanup

Inspect the receipt and Beaker experiments before resubmitting: a timeout may
mean the API accepted the job but the trainer missed its response. Check the
latest job attempt; if queued, inspect `beaker job events JOB_ID`.

```bash
python -m open_instruct.miles.evaluation.evaluation resubmit RECEIPT.json
```

This creates a new receipt and results group, retaining the original diagnostic.
It requires the committed runner source matching `runner_sha256`, the recorded
image, secrets, mounts and completed snapshot. Submission/application failures
are not automatically resubmitted; Beaker auto-resume handles preemption within
an already submitted experiment.

Snapshots are excluded from ordinary checkpoint retention and result collection.
There are no pins, leases, automatic deletion or retention controller. Manually
remove selected `eval-snapshots/update-NNNNNNNN` directories only after verifying
that **all** jobs referencing them have stopped (including manual resubmissions).
Do not delete snapshots while jobs are pending or running. Keep small receipts
and results as provenance. Initial checkpoint reuse additionally depends on the
original HF source remaining available.

## Evaluator image and compatibility

`runtime/miles/Dockerfile.evaluator` builds a separate evaluator image from a
MILES serving image and a pinned olmo-eval revision. It retains the
Olmo MoE SGLang architecture extension, installs olmo-eval in a separate CPU
virtual environment, and connects olmo-eval's `vllm_server` API client to its own
private SGLang server. It does not launch vLLM. The runner checks the olmo-eval
revision before starting. Build evaluator dependencies at image construction.
External benchmark sandboxes may have their own upstream setup steps. Publish
the image to Beaker and configure its immutable ID.

The large example sets `image` to the published evaluator image
`01M5KSXYY6BENK3GH8WV8B8KK0` and `revision` to olmo-eval
`710b2b5dab6ea09764c02647f3a4e72819b9ee7c`, which includes `gsm8k:platinum`
and the [stop-sequence normalization fix](https://github.com/allenai/olmo-eval/pull/450).
Keep `revision` matched to the olmo-eval commit installed in the selected image.
A serving-runtime refresh can change the image ID without changing that revision.

`runtime/miles/Dockerfile.evaluator-sandbox` extends a pinned evaluator with
Podman and the sandbox extra while preserving its serving environment. The
published panel image extends `01M45DMWV3FEH7TNAXKZWR2DF2`; its Docker ID is
`sha256:4461a621ebadba8f78ecee51a4e5f130a99623ef07e51c458e53e157e5726dc6`.
Build it with `--build-arg EVALUATOR_IMAGE=LOCAL_PINNED_EVALUATOR_IMAGE`.

Check compatibility for the exported architecture, serving backend, task and
image; an OpenAI-compatible endpoint alone does not prove loglikelihood or task
correctness. Use generation tasks for an initial tiny MoE check. A successful
mechanics check says nothing about model quality. Launch training through the
normal committed-image MILES wrapper. Use ignored `runs/` configs and fresh
output paths; leave existing experiments unchanged.

## Development gold labels and scoring

Use `scripts/miles/development_eval.py` for custom math/GSM8K development
scoring. Prepared math labels can be lists of acceptable strings. Preserve that
flat list rather than wrapping it again: `['42']` must not become `[['42']]`.
The scorer rejects malformed labels instead of silently recording zero.
GSM8K uses the training reward's exact decimal comparison, so `18.00`, `+18`
and `18` are equivalent. Its parity tests run in the normal CPU suite without
olmo-eval; math scoring still requires the olmo-eval environment.
Regression tests cover scalar/list/multiple gold answers and completion gates:

```bash
# Run inside the pinned olmo-eval environment; no GPU is needed.
python -m pytest --confcutdir=tests/miles_eval -o pythonpath=scripts/miles tests/miles_eval
```

These tests are separate from the MILES trainer suite because they use the
evaluator environment. `pythonpath` mirrors the standalone module layout in
custom evaluator bundles; a missing scoring module fails collection when
olmo-eval is installed. Without olmo-eval, ordinary CPU collection skips them.

Include this module in the custom worker's content manifest and import its
`score_one(row, item)` for development tasks. Check the actual panel's label
schema, not just synthetic scalar-label canaries. Version corrected bundles;
never modify a running experiment's immutable dataset in place.
