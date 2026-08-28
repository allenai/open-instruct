# Training Observatory registry

This directory is the curated source of truth for the logical experiments shown by the Training Observatory. It records identity, run, provenance, and artifact locations.

The local viewer joins the registry to those live sources at runtime. See the
[viewer guide](../README.md) for the resulting UI and inspection features.

## Layout

```text
viewer/registry/
├── config.yaml
└── trainings/
    └── <stable-training-id>.yaml
```

`config.yaml` defines the registry schema version, repository root, and shared
W&B defaults. Every file under `trainings/` represents one scientific training
run. Its filename must equal its `id`.

```yaml
schema_version: 1
kind: training
id: q35-9b-bc72k-sv-llmj-8x32
```



## What counts as one run

One registry file may contain an initial launch, failed retries, and checkpoint
resumes. Keep them together when they preserve the experiment's scientific
identity and continue the same intended W&B run.

Create a new registry file when a comparison materially changes any of:

- base model or initialization checkpoint;
- training dataset or retrieval corpus;
- retrieval backend or available tools;
- verifier/reward construction or format policy;
- rollout geometry or optimizer recipe;
- intended W&B identity.

Do not merge runs because their names look similar, share a directory, or use
the same script. Conversely, a crash and resume is not a new experiment merely
because it produced a new Beaker experiment ID, image, process timestamp, or
rollout attempt prefix.

## Complete example

```yaml
schema_version: 1
kind: training
id: q35-9b-bc72k-sv-llmj-8x32
title: Qwen3.5-9B · BC-v2 72k · search/visit · LLM verifier · 8×32
classification: evaluated
visibility: default

tags:
  model: Qwen3.5-9B
  corpus: BM25 72k
  tools: search/visit
  verifier: LLM judge
  geometry: 8×32

wandb:
  run_id: t604arbl
  # entity, project, validation_metric, and rollout_artifact_offset inherit
  # from config.yaml unless this run overrides them.

launches:
- id: initial
  relation: initial
  script: scripts/search/train/experiments/train_bc_9b_example.sh
  git_repository: wu-ming233/open-instruct
  git_commit: 0123456789abcdef
  beaker_experiment: 01EXAMPLEINITIAL
  image: 01EXAMPLEIMAGE
  rollouts:
  - path: rl_rollouts/example-run
    attempts:
    - example_run__42__1780000000

- id: resume-60
  relation: resume
  script: scripts/search/train/experiments/train_bc_9b_example.sh
  git_repository: wu-ming233/open-instruct
  git_commit: fedcba9876543210
  beaker_experiment: 01EXAMPLERESUME
  image: 01EXAMPLEIMAGE2
  rollouts:
  - path: rl_rollouts/example-run
    attempts:
    - example_run__42__1781000000
  note: Resumed from the complete step-60 state checkpoint.

artifacts:
  furthest_step: 140
  evaluations:
  - benchmark: browsecomp-plus-bm25-830
    step: 100
    correct: 470
    total: 830
    checkpoint:
      step: 100
      path: /weka/.../checkpoints/step_100
    inference_artifact:
      path: output/browsecomp-plus/example/step100_question_only_300_256k
      schema: open_instruct_inference_v1
      judged_response_index: -1
  - benchmark: browsecomp-serper-jina-1266
    step: 100
    correct: 490
    total: 1266
    inference_artifact:
      path: output/browsecomp/example/step100_serper_jina_question_only_300_256k
      schema: open_instruct_inference_v1
      judged_response_index: -1
  best_evaluation:
    benchmark: browsecomp-plus-bm25-830
    step: 100
    correct: 470
    total: 830
    checkpoint:
      step: 100
      path: /weka/.../checkpoints/step_100
  latest_checkpoint:
    step: 140
    path: /weka/.../checkpoints/step_140
```



## Field reference



### Identity and visibility


| Field            | Meaning                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| `id`             | Stable URL/file identifier. Use lowercase letters, numbers, `.`, `_`, or `-`.                     |
| `title`          | Human-readable experiment title.                                                                  |
| `classification` | Conventionally `evaluated`, `substantive`, or `smoke`. Controls catalog grouping.                 |
| `visibility`     | `default`, `archive`, or `hidden`. Archived and hidden entries require explicit filters.          |
| `tags`           | Searchable string-to-string facts such as model, corpus, tools, verifier, geometry, or optimizer. |
| `note`           | Optional run-level explanation shown in the training workspace.                               |


Classification describes scientific status; visibility describes whether the
entry belongs in the default catalog. For example, a substantive superseded run
can be archived without relabeling it as smoke.

### W&B binding

The top-level `wandb` mapping identifies the logical run used for live progress,
validation, and metric charts.

```yaml
wandb:
  entity: stevenchenzijian-university-of-waterloo
  project: tmax
  run_id: t604arbl
  validation_metric: eval/objective/verifiable_correct_rate
  rollout_artifact_offset: -1
```

Defaults come from `config.yaml`. `rollout_artifact_offset: -1` means optimizer
step `N` corresponds to zero-based rollout artifact step `N - 1`.

A launch inherits the top-level W&B mapping. Give a launch its own `wandb`
mapping only when a retry or resume intentionally writes to a different run.
All registered W&B run IDs must be globally unique to one training run.

### Launches and provenance

Launches are chronological. Later registered attempts win when resumed rollout
ranges overlap earlier ones.


| Field               | Meaning                                                                                                                |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `id`                | Unique label inside the run, such as `initial`, `retry-1`, or `resume-60`.                                         |
| `relation`          | Human-readable relationship: normally `initial`, `retry`, `resume`, or `diagnostic`.                                   |
| `script`            | Current repository-relative script path when it still exists.                                                          |
| `historical_script` | Optional path/name retained when the original script moved or was deleted.                                             |
| `git_repository`    | GitHub `owner/repository` or HTTPS URL. Defaults to `allenai/open-instruct`; set the fork explicitly when appropriate. |
| `git_commit`        | Exact source commit used by the image. It is linked against `git_repository`, not assumed to exist upstream.           |
| `beaker_experiment` | Beaker experiment ID, without the URL prefix.                                                                          |
| `image`             | Beaker image ID.                                                                                                       |
| `rollouts`          | One or more persistent rollout-directory bindings.                                                                     |
| `note`              | Optional launch-specific incident or resume explanation.                                                               |


Never point a fork commit at `allenai/open-instruct`. The frontend constructs the
commit URL from both `git_repository` and `git_commit`.

### Rollout bindings and attempts

```yaml
rollouts:
- path: rl_rollouts/example-run
  attempts:
  - run_name__42__1780000000
  - run_name__42__1781000000
```

`path` names the persistent directory. Each `attempt` is the exact timestamped
prefix used by that trainer process and its metadata/shard files. Register
attempts in chronological order.

Use:

```yaml
configured_only: true
attempts: []
```

only when the launch configured that rollout path but produced no attempt
metadata at all. If a process created metadata or any shard—even if it failed
during startup—register the actual prefix instead. `configured_only` is not a
generic marker for missing, crashed, or deleted data.

The loader rejects one attempt assigned to multiple training runs.

### Checkpoints

```yaml
latest_checkpoint:
  step: 140
  path: /weka/.../checkpoints/step_140
```

Checkpoint paths name checkpoint directories. The frontend reports:

- `exists` when the directory is present;
- `complete` when `config.json` exists inside it.

It does not treat an existing parent directory as a valid deleted checkpoint.
Use optimizer-step numbers in checkpoint fields.

### Evaluation history

`artifacts.evaluations` stores every known BrowseComp or BrowseComp-Plus result,
not only the best score:

```yaml
- benchmark: browsecomp-plus-bm25-830
  step: 120
  correct: 441
  total: 830
```

The pair `(benchmark, step)` must be unique within a run. Keep older rows
when a new checkpoint finishes. The catalog and training page compute best
BrowseComp and best BrowseComp-Plus results independently from this history.

When the same benchmark/step also appears in
`evaluations`, the loader merges the checkpoint and inference-artifact metadata
instead of duplicating the row. New entries should still populate the full
`evaluations` list.

### Inspectable inference artifacts

An evaluation becomes clickable when it includes:

```yaml
inference_artifact:
  path: output/browsecomp-plus/example/step120_question_only_300_256k
  schema: open_instruct_inference_v1
  judged_response_index: -1
```

The registered directory must retain:

```text
<path>/
├── <query-id>.json                 raw inference record with responses[]
└── eval/
    ├── evaluation_summary.json
    ├── evaluation_results.jsonl
    └── judge_results/
```

The evaluator row must be joinable to a raw query JSON by query ID or by the
saved input basename. `judged_response_index` uses Python indexing; `-1` means
the last response. It must identify the response that produced the saved judge
verdict. The examiner defaults to the final response and exposes a compact
selector only when a question has multiple rollouts.

The directory is considered complete only when both evaluation JSON files and
the `judge_results/` directory exist. Register the exact inference output
directory, not its parent. Inference artifact paths must be globally unique to
one training/evaluation pair.

The examiner classifies rows as judged correct, judged incorrect, or incomplete
and warns when those categories do not account for the registry's `total`.
Therefore `correct` and `total` must describe the exact retained evaluation,
not a copied summary from another run.

## Paths

Relative paths resolve from the configured repository root (`../..` in the
checked-in `config.yaml`). Prefer repository-relative paths for scripts,
rollouts, and inference outputs. Keep external Weka checkpoint paths absolute.

The registry preserves missing paths as provenance. Do not redirect a deleted
artifact to a nearby surviving directory merely to make the UI green.

## Update workflow



### Starting a new training

1. Copy the closest training YAML and choose a new stable ID when the scientific
  identity changes.
2. Record the exact W&B run ID and searchable tags.
3. Record the rendered script, source repository and commit, Beaker experiment,
  image, rollout path, and timestamped attempt prefix.
4. Set `furthest_step` and checkpoint fields only after those artifacts exist.



### Retry or resume

1. Append a launch to the existing run; never replace the initial launch.
2. Record the new source commit/image and why the launch exists.
3. Append its rollout attempt prefix chronologically.
4. Add a launch-level W&B mapping only if it intentionally differs.



### Completing BrowseComp or BrowseComp-Plus evaluation

1. Append a new `artifacts.evaluations` row with the exact benchmark, checkpoint
  step, and correct/total counts.
2. Add the checkpoint path when known.
3. Add `inference_artifact` when the raw query JSONs and complete `eval/`
  directory were retained.
4. Update `best_evaluation` only when the curated best result changes; do not
  delete prior evaluation rows.
5. Change `classification` to `evaluated` when appropriate.



### Validate and refresh

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q \
  viewer/tests/test_training_registry.py \
  viewer/tests/test_registry_viewer.py \
  viewer/tests/test_evaluation_store.py
```

Then refresh a running local server with either **Refresh live data** or:

```bash
curl -X POST http://127.0.0.1:8082/api/refresh
```

Verify the catalog row, W&B history, launch links, checkpoint status, rollout
steps, evaluation counts, and at least one external-evaluation trajectory.

Refreshing the local server does not update a static hosted site. After local
verification, regenerate and republish the hosted snapshot separately. A live
hosted viewer requires an authenticated backend with access to Weka and W&B.

Automatic insertion by the training and evaluation launchers is not yet
implemented, so this workflow is required whenever a run starts, retries,
resumes, or gains a new external evaluation.

## Enforced integrity rules

The loader scans `trainings/*.yaml` in deterministic filename order and rejects:

- unsupported schema or `kind` values;
- filename/ID mismatches and duplicate training IDs;
- duplicate W&B run ownership across runs;
- one rollout attempt assigned to multiple runs;
- duplicate `(benchmark, step)` rows in one run;
- one inference artifact assigned to multiple evaluations;
- malformed GitHub repository values;
- invalid checkpoint/evaluation integers or inference schemas.

These checks are global across the registry. Splitting experiments into separate
YAML files does not weaken ownership validation.