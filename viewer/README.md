# Training Observatory

> **Terminal-RL adaptation (branch `command_center`).** This viewer was vendored from
> Steven's search/BrowseComp fork and adapted for Terminal RL runs:
>
> - `viewer/_compat.py` shims the BrowseComp-specific `ground_truth_utils` symbols that do
>   not exist on this branch (format gates become no-ops; the browsecomp-specific outcome
>   classification is pending replacement with the terminal truncation-vs-genuine logic).
> - Registry entries live in `viewer/registry/trainings/` and describe Terminal RL runs
>   (wandb `ai2-llm/oe-general-agents`, evals = Terminal-Bench 2.1 / TBlite trial counts).
> - Rollout shards are NOT read from the shared Weka dump directly (its ~200k files make the
>   startup scan crawl). Run `python -m viewer.tools.link_rollouts` to build the repo-local
>   `rl_rollouts/` symlink farm from the registry's `source` + `attempts` declarations, and
>   point `--rollouts-dir` at it. Nothing is copied or moved.
> - Catalog/eval columns are relabeled TB2.1 / TBlite (`terminal-bench*`/`tb2*` and `tblite*`
>   benchmark names map onto the two catalog slots).
>
> Run it as a detached daemon that survives SSH/Claude sessions (no systemd in the
> dev containers):
>
> ```bash
> ./viewer/serve.sh start        # stop | restart | status; log in .viewer_cache/server.log
> ```
>
> Defaults: 127.0.0.1:8090 (SSH-tunnel with `ssh -L 8090:127.0.0.1:8090 <vm>`); override
> with HOST/PORT/TOKENIZER. From a git worktree without a synced venv, point ENV_REPO at
> the main clone whose uv env is synced:
>
> ```bash
> ENV_REPO=/weka/nora-default/shashankg/code/open-instruct ./viewer/serve.sh start
> ```

The Training Observatory brings four kinds of
data into one workspace:

- experiment identity and file paths from `viewer/registry/`;
- live training scalars and validation histories from W&B;
- accepted and active-sampling-discarded training rollouts from JSONL shards;
- BrowseComp and BrowseComp-Plus inference trajectories and judge outputs from
retained evaluation directories.

The viewer does not copy those large artifacts into a database. The registry
records where they live, and the local Python server reads them lazily.

Read [the registry guide](registry/README.md) before adding or changing an
experiment.

## What runs where

```text
viewer/registry/trainings/*.yaml       curated identity and file paths
                  │
                  ├── W&B run IDs ───► validation and training-metric histories
                  ├── rollout paths ─► accepted/filtered training JSONL shards
                  ├── checkpoints ───► live completeness checks on Weka
                  └── evaluations ───► raw inference JSON + judge result folders
                                      │
viewer.server ◄───────────────────────┘
      │
      ├── RolloutStore       lazy training-rollout index and decoder
      ├── EvaluationStore    lazy inference/judge join and transcript builder
      ├── RegistryIndex      registry-to-W&B/rollout bindings
      └── ExperimentService  catalog and refresh orchestration
      │
      └── viewer/static/     browser UI
```

There is no write endpoint for experiments, checkpoints, or results. Updating
the Observatory means updating the registry or the underlying artifacts and
then refreshing the server.

## Quick start

From the repository root:

```bash
uv run --no-sync python -m viewer.server \
  --tokenizer hamishivi/Qwen3.5-4B \
  --port 8082
```

Open [http://127.0.0.1:8082](http://127.0.0.1:8082).

The tokenizer is used only when a stored training rollout contains token IDs
that must be decoded. It is prewarmed in a background thread and does not block
the catalog from opening.

Note that we expect the rl rollouts used during training to be in `--rollouts-dir`, default `<repo>/rl_rollouts`.

We also expect registry files that keep track of each experiment inside `--registry`, default `viewer/registry`.

```text
--registry PATH          Registry directory or config.yaml
                         (default: viewer/registry/)
--rollouts-dir PATH      Fallback/discovery root for rollout shards
                         (default in registry mode: <repo>/rl_rollouts)
--tokenizer MODEL        Tokenizer used for on-demand training-trace decoding
--response-limit N       Canonical response-token ceiling (default: 131072)
--cache-steps N          Classified step/source pairs held in memory (default: 16)
--host HOST              Bind address (default: 127.0.0.1)
--port PORT              HTTP port (default: 8080)
--verbose                Print HTTP request logs
```

`TRAINING_REGISTRY`, `ROLLOUTS_DIR`, and `WANDB_VIEWER_PATH` provide the
corresponding environment-variable overrides.

## The three primary views



### 1. Experiment catalog

The home page lists logical training runs rather than individual Beaker attempts. 

Each row summarizes:

- furthest optimizer step;
- best registered BrowseComp-Plus result;
- best registered BrowseComp result;
- best in-training validation score from W&B;
- retained rollout/checkpoint availability.



### 2. Training workspace

Opening a run goes directly to its rollout workspace while keeping global run
information above it:

- model, corpus, tools, verifier, geometry, and other tags;
- W&B and Beaker links;
- furthest step, latest checkpoint, and artifact status;
- all registered BrowseComp-Plus and BrowseComp results by checkpoint;
- launch history with script, source repository/commit, image, and rollout
paths;
- interactive W&B charts with hoverable points and optimizer-step axes.

The chart layer probes the metric names actually used by older and newer
training runs. It currently surfaces:

- validation score and training reward;
- mean prompt-group pass rate before and, when different, after trajectory
masking;
- rejected-group rate and the all-zero/all-one split among rejected groups;
- incomplete, terminal-format, and trajectory-format rollout ratios before
active-sampling selection and within the retained learner batch;
- average full-trajectory length, average terminal-turn length, and latest
token-capped fraction;
- tool calls per rollout plus search/visit failure rates;
- vLLM/local logprob drift, or policy entropy when that is the available
scalar.

The lower workspace can switch between individual **Trajectories** and prompt
**Groups**. It supports accepted learner batches and, when retained, discarded
active-sampling groups.

### 3. External evaluation examiner

This is optional. For each training, you may also attach evaluation runs (BrowseComp or BrowseComp-Plus) to the registry file, then you can inspect the evaluation trajectories. These trajectories have to satisfy a specific shape (see `/weka/oe-adapt-default/stevenc/output/browsecomp-plus/9b/question_only_300_256k` for example).



Every registered evaluation row is clickable when it has a complete
`inference_artifact`. The evaluation page joins:

1. the benchmark totals recorded in the registry;
2. `eval/evaluation_summary.json`;
3. `eval/evaluation_results.jsonl` and its saved judge records;
4. the raw per-question inference JSON containing `responses`.

The left pane filters and sorts questions. The right pane shows the selected
question, reference answer, terminal response, rollout state, evaluator input
and output, and the reconstructed turn-by-turn trajectory.

Evaluation outcomes are intentionally limited to:

- **Judged correct** — a completed rollout with a parseable positive verdict;
- **Judged incorrect** — a completed rollout with a parseable negative verdict;
- **Incomplete** — inference did not complete, so the judge did not run.

Most current evaluations have one rollout per question. If `responses` contains
more than one, a compact selector appears and defaults to the last rollout.
Only `inference_artifact.judged_response_index` inherits the saved judge verdict;
other responses are clearly marked as unjudged alternatives.

## Training rollout inspection



### Accepted versus discarded

The UI names the two populations by their training meaning:

- **Learner batch** (`accepted` in files and APIs) contains every trajectory
from prompt groups retained by active sampling. It is not a correctness
filter: a mixed retained group normally contains both reward-0 and reward-1
trajectories.
- **Discarded groups** (`filtered`) contains zero-variance prompt groups that
active sampling rejected. All-correct and all-wrong groups appear here when
`--save_filtered_rollouts` was enabled and those shards still exist.

Rollout artifact step `N` normally supplied optimizer step `N + 1`. The
registry's W&B offset makes this mapping explicit for unusual runs.

### Trajectory outcomes and review flags

The trajectory browser separates answer quality from failure to reach a
verifier:

- **Judged correct** and **Judged incorrect** reached the registered verifier.
- **Incomplete** stopped at the response, tool-step, context, reset, or
generation boundary, had no terminal turn, or did not stop cleanly.
- **Format Error** applies the verifier policy registered for that lineage.
- **No Tool Calls**, **Tool timeouts**, and **Gibberish** expose structural
anomalies.
- **Judge: negative has answer** and **Judge: positive has no answer** are
exact-reference containment screens for manual judge review, not proof that
the judge is wrong.
- **Token capped**, **Long**, **Tool errors**, and **Discarded** are context
filters and do not by themselves mark a trajectory suspicious.

When the rollout saved `verifier_input`, `verifier_skipped_reason`, and  
`judge_output`, the detail pane displays those exact values alongside the raw  
terminal model turn. 

## Evaluation transcript tools

- Reference-answer matches are counted separately in model reasoning/text,
tool calls, tool results, and final output. Matches are highlighted in red
and navigable with previous/next controls.
- The per-question search box performs case-insensitive literal search over the  
complete saved trajectory, even when the visible block is only a preview.
- A truncated tool result can be expanded to its complete saved content in a
floating pane without an inner scrollbar. This reveals the full persisted
segment; it does not refetch the source document from the web.
- Visit errors show a visible `404 not found` badge and, when a URL is  
recoverable from the corresponding call, a **Visit site** link.



### BrowseComp-Plus evidence and gold documents

BrowseComp-Plus evaluations additionally join against
`viewer/data/browsecomp_plus_urls.jsonl`:

- evidence documents use a pale-yellow marker and border;
- positive/gold documents, a subset of evidence, use a stronger gold marker
and star;
- red reference-answer highlights remain independent and can overlap either
document annotation;
- separate previous/next controls navigate evidence-bearing and gold-bearing
tool results.

The original BrowseComp benchmark does not use this mapping.

## Refreshing after a training or evaluation update

The **Refresh live data** button calls `POST /api/refresh`. In registry mode it:

1. reloads every registry YAML file;
2. clears evaluation and rollout caches;
3. rechecks registered files and checkpoints;
4. starts a fresh W&B validation refresh in the background.

Use this after editing a registry entry, after a rollout shard grows, or after a
registered evaluation directory becomes complete. Trainer scalar histories are
loaded lazily when a training page is opened.

Follow the [registry update workflow](registry/README.md) whenever  
a new run starts, resumes, or produces an external evaluation.