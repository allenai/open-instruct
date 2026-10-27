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

The [three-node example](../../configs/miles/examples/three-node.toml) opts into background
evaluation with a pinned sandbox-enabled image and the 25-entry post-training
development/text-safety panel. It sets `interval=0`, `initial=true` and `final=true`
to evaluate the full panel before and after training only, including when the run
is extended. Zero disables periodic evaluation; it does not disable either endpoint.
Positive intervals count completed optimizer updates, not rollout collections.
Omitting `interval` retains the default of 20 updates. Task-level intervals override
the global value: a task can opt into periodic evaluation under a global zero, or
set zero to opt out under a positive global interval. The global `initial` and
`final` flags still apply to every task. With both flags false and all intervals
zero, no background evaluations or evaluation snapshots are scheduled.
The [node example](../../configs/miles/examples/node.toml) runs a smaller GSM8K,
GSM8K-Platinum and IFBench panel initially, every 50 updates, and finally.
Dev retains shared-engine evaluation, which remains the low-level default. The three-node template is
provisional; check evaluator compatibility for the full model. Remove explicit
shared settings (`training.eval_interval`, `miles.eval_*`, `skip_eval_before_train`,
`n_samples_per_eval_prompt`, `data.eval_prompt_data`, positive task `eval_count`)
when selecting background mode. Evaluation tasks are independent of training
`data.tasks`; no evaluation-only dataset needs to be added to the training mix.

```toml
[evaluation]
mode = "background"
interval = 0                  # Initial/final only; positive values enable periodic evaluation
initial = true
final = true
image = "IMMUTABLE_BEAKER_IMAGE_ID" # Replace with a compatible evaluator image
revision = "FULL_40_CHARACTER_OLMO_EVAL_COMMIT"
gpus = 1                     # Separate allocation; never training GPUs
cluster = "ai2/holmes"
submit_timeout = 30           # Hard wall-clock bound on a submission subprocess
timeout = "3h"                # Evaluator Beaker task lifetime
provider_kwargs = { profile = "generic", attention_backend = "triton", disable_cuda_graph = true }

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

The limited GSM8K tasks above are small mechanics examples. The three-node template
uses individual task variants from `math:posttrain:dev`, `general:posttrain:dev`,
`hillclimb:dev` and `safety_thinking`, deduplicated to 25 entries. Individual names
make the panel explicit and do not create synthetic cross-benchmark aggregate scores. It retains full task sizes
and sampling defaults from the pinned olmo-eval revision, including AIME/HMMT
pass@32 with 16K responses. The pinned IFBench defaults differ from newer upstream
revisions; changing the image/revision can change the benchmark protocol.

Use the full panel at the initial and final checkpoints for official comparisons.
The example above shows how one task can additionally run every 100 updates; its
`scoring.limit` applies at every scheduled point, including initial and final.
The three-node template has no such limits or periodic overrides. Progress subsets are
not enabled by default: fewer prompts still incur separate-job and judge startup
costs. If adding a periodic development panel, choose fixed example IDs with a
documented seed and relevant category coverage, and distinguish its metrics from
the full-panel scores. A simple `limit` is not a seeded or stratified sample, and
this configuration does not switch sampling/scoring settings between endpoints
and periodic points. Keep release held-out evaluation separate from development
metrics used to select recipes or checkpoints.

Generation overrides map to olmo-eval's flat per-task sampling overrides
(for example `-o max_tokens=2048`, not a nested `sampling_params` table); scoring maps to
olmo-eval task overrides (such as `limit`, scorer settings, or formatting).
Tasks with the same generation overrides, harness preset, harness overrides and
execution kind and auxiliary GPU count share a job. Empty generation overrides preserve each task's own
defaults. Each unfinished task starts its own native SGLang engine; completed
tasks remain reusable after preemption. Model loading repeats between tasks.
The three-node panel creates five groups: 17 ordinary tasks, LiveCodeBench, SciCode,
five WildGuard-judged tasks and StrongREJECT. Each group requests one policy GPU;
each judge group also reserves one GPU with task `auxiliary_gpus=1`. The worker
exposes the policy and auxiliary GPUs to olmo-eval, whose planner assigns
non-overlapping GPU sets and manages judge startup and shutdown. One HF snapshot serves all groups. A coincident
periodic/final point is submitted only once for each task group. `plan` displays
the milestones, task groups, resources and snapshot paths. `validate` checks the
schema; it does not establish image/model/task compatibility.

### Code and safety requirements

`harness_preset` selects an olmo-eval preset (default: `default`). The optional
task `harness` table passes flat harness overrides before the `--task` argument,
so auxiliary providers and sandboxes are not mistaken for scoring overrides.
The frozen checkpoint's `provider` settings cannot be overridden here. Harness
settings are recorded in receipts and checked when reusing completed tasks.

The three-node template selects `codex_python` for LiveCodeBench and enables
`sandboxes.0.inject_swerex` to preinstall the sandbox server. The pinned runtime's
on-demand installation fallback otherwise invokes an unavailable executable.
Code jobs receive
Beaker's subcontainer environment flags and need an evaluator image containing
Podman and the olmo-eval sandbox dependencies. Container startup and reference-data
access still require qualification on the target cluster.

SciCode uses `kind="external"` and `arguments`, invoking olmo-eval's
`run-external` command against a dedicated frozen SGLang HTTP server. SciCode
requires `VLLMServerProvider` in the pinned olmo-eval revision and is the explicit
exception to native `sglang` routing. `evaluation.external_server_args` configures
only this server; native tasks use `evaluation.provider_kwargs`. Only SciCode currently
has a supported external artifact contract. Development runs require
`arguments.split="validation"` and an accessible `arguments.h5py_host_path`.
The evaluator checks per-problem predictions and step counts, rejecting
infrastructure errors even when upstream reports success with numeric zero scores.
SciCode's sandbox installs its pinned scientific dependencies from the upstream
lock; account for startup time and registry/package access when qualifying jobs.

Safety tasks configure olmo-eval managed WildGuard and StrongREJECT providers.
WildGuard serves `allenai/wildguard`; StrongREJECT serves `google/gemma-2b` with
the `qylu4156/strongreject-15k-v1` adapter and its tokenizer. The sandbox evaluator
image includes the pinned vLLM extra in an isolated environment selected through
`VLLM_PYTHON`, preserving the policy serving environment. Model access credentials
(including gated Gemma access) belong in `evaluation.secrets`. Record resolved
judge revisions and serving settings with qualification results.

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

`runtime/miles/Dockerfile.evaluator` builds a separate evaluator image on the
digest-pinned CUDA 13.0.2 / Torch 2.13 compiler-equipped olmo-eval base. It installs
the pinned `sglang` extra, including the engine and Olmo model plugin. Regular
background evaluations select `provider.kind=sglang` and
`provider.kwargs.model_registration=olmo_sglang:register`. SciCode retains its
explicit HTTP exception described above. The runner checks the installed
olmo-eval revision against the receipt before evaluating.

Both examples pin olmo-eval revision `3536fe751e40162cc33d5e647085e986375aadce`,
which includes the [native SGLang provider](https://github.com/allenai/olmo-eval/pull/506).
The node image is `01M5ZGD5K01GNW3ASDPNXXPEJ3`. Three-node uses sandbox-enabled
image `01M5ZSWYBJ5GPG1XBARFFC59MZ` with isolated vLLM judges. Both recipes build evaluator dependencies
at image construction; benchmark sandboxes may have their own upstream setup steps.
The revision change can also change task defaults; do not compare scores across
revisions as though the benchmark protocol were unchanged.

Replace the old `evaluation.server_args` with typed `evaluation.provider_kwargs`
using Python option names, for example `context_length = 34816` and
`disable_cuda_graph = true`. Use `profile = "generic"` when manually tuning
settings; automatic profiles reject conflicting overrides. For SciCode, retain
CLI arguments under `external_server_args` as well. The checkpoint, tokenizer,
model registration and GPU allocation remain owned by MILES. Task overrides
cannot change the policy provider. Native SGLang supports generation, not tools
or continuation likelihood; code scoring in a sandbox is separate from tool use.

```bash
docker build -f runtime/miles/Dockerfile.evaluator \
  --secret id=github_token,env=GH_TOKEN \
  -t open-instruct-miles-evaluator:sglang runtime/miles
docker build -f runtime/miles/Dockerfile.evaluator-sandbox \
  --build-arg EVALUATOR_IMAGE=open-instruct-miles-evaluator:sglang \
  -t open-instruct-miles-evaluator:sglang-panel runtime/miles
```

Keep each config's image and revision matched. Changing only the revision does
not upgrade an existing image. CPU tests check command parsing, GPU planning,
image dependencies and task resolution; GPU generation and full-panel resource
bounds still need qualification with these images.

The provider contract tests use the installed olmo-eval CLI and GPU planner,
without loading a model or downloading datasets. Run them inside the panel image
with this checkout mounted and pytest installed:

```bash
python -m pytest --confcutdir=tests/miles_eval tests/miles_eval/test_background_provider.py
```

Check compatibility for the exported architecture, serving backend, task and
image; successful provider initialization alone does not prove task correctness. Use generation tasks for an initial tiny MoE check. A successful
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
