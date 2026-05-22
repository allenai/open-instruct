# mason.py Deep Reference

`mason.py` is the Beaker experiment launcher. It does significantly more than just submitting your command — it transforms the command, pre-caches datasets locally, injects environment variables and secrets, auto-manages output/checkpoint directories, and handles multi-node coordination. Understanding these implicit behaviors is important for debugging unexpected job behavior.

---

## The Arg Split

```
python mason.py [mason-args] -- <your command>
```

Mason uses `parse_known_args` to split everything before `--` as its own args, and everything after as your command. Multiple `--` separators create multiple commands, each becoming a separate Beaker task within the same experiment:

```bash
python mason.py [mason-args] -- python script_a.py [args] -- python script_b.py [args]
```

All tasks share the same image, cluster, budget, and priority, but each gets its own command string and task spec.

---

## What Mason Does Before Submitting

This is the core of what makes mason non-trivial. For each command, `make_internal_command` applies the following transformations in order:

### 1. W&B Environment Passthrough

If `WANDB_ENTITY`, `WANDB_PROJECT`, or `WANDB_TAGS` are set in your **local shell**, mason prepends them inline to the command string:

```
WANDB_ENTITY=ai2-llm WANDB_PROJECT=my-project python open_instruct/grpo_fast.py ...
```

This means your local W&B env vars are baked into the submitted command, not passed as Beaker env vars.

**However, this does not work for overriding `wandb_project`.** All Open Instruct training scripts (`grpo_fast.py`, `dpo.py`, `finetune.py`) pass `project=args.wandb_project` explicitly to `wandb.init()`. The default for this field is `"open_instruct_internal"` (defined in `LoggingConfig`). Because the value is always passed explicitly — never as `None` — W&B ignores the `WANDB_PROJECT` env var entirely. `HfArgumentParser` also does not populate dataclass fields from env vars, so setting `WANDB_PROJECT` in your shell has no effect on `args.wandb_project`.

To override the W&B project, pass it explicitly in the training command:

```bash
python scripts/update_command_args.py scripts/train/tulu3/grpo_fast_8b.sh \
    --wandb_project my_project | uv run bash
```

### 2. Special Character Escaping

- Any argument containing `</` (e.g. `--stop_strings "</answer>"`) is wrapped in single quotes.
- Any argument containing `{` (e.g. JSON dataset mixers) is wrapped in single quotes. This handles cases like `--dataset_mixer '{"dataset": 1.0}'` being passed correctly through bash.

### 3. Auto-injection of Entity Args (Open Instruct commands only)

For recognized training scripts (`grpo_fast.py`, `finetune.py`, `dpo.py`, etc.), if your command does not already include `--hf_entity` or `--wandb_entity`, mason appends:

```
--hf_entity allenai --wandb_entity ai2-llm
```

These are needed for HuggingFace uploads and W&B logging. You will not see these in your training script — mason adds them silently.

### 4. Local Dataset Pre-Caching

Before submitting to Beaker, mason **runs your training command locally** with `--cache_dataset_only` appended. This tokenizes and caches the dataset on the machine where you invoke mason (typically a compute node with Weka access). The Beaker job then finds the cache already populated on Weka and skips tokenization entirely.

Why: submitting a CPU-only job to Beaker to do this would require dependency job orchestration. Local pre-caching takes ~5 minutes and is simpler.

During this caching run, mason strips arguments that shouldn't affect the cache key or would cause errors locally:

| Stripped arg | Has value? | Why stripped |
|---|---|---|
| `--with_tracking` | no | W&B not needed for caching |
| `--checkpoint_state_freq` | yes | irrelevant to caching |
| `--checkpoint_state_dir` | yes | irrelevant to caching |
| `--gs_checkpoint_state_dir` | yes | irrelevant to caching |

**Skip caching with `--no_auto_dataset_cache`.** This is required when running from macOS (vllm cannot be imported locally) and is also useful when you know the cache is already warm. Always include this flag in training scripts that call mason, since mason is sometimes invoked on machines without vllm installed.

**GS bucket models**: If `--model_name_or_path` starts with `gs://`, mason downloads the tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `config.json`) to a local directory under `auto_output_dir_path/<whoami>/tokenizer_<md5hash>/` before the cache run, then temporarily substitutes the local path. This avoids needing GCS credentials inside the caching subprocess.

See [Dataset Cache Internals](#dataset-cache-internals) for details on how the cache hit/miss decision is made.

### 5. Checkpoint State Dir Auto-Override

For resumable jobs (`grpo_fast.py`), mason automatically sets `--checkpoint_state_dir` to:

```
<auto_checkpoint_state_dir>/<whoami>/<unix_timestamp>_<random_int>/
```

Default base: `/weka/oe-adapt-default/allennlp/deletable_checkpoint_states`

**Exception**: if your command already specifies `--checkpoint_state_dir` pointing to a `/weka/` path, mason leaves it alone. This is how you resume from a specific checkpoint — pass the existing `/weka/...` checkpoint dir explicitly and mason won't overwrite it.

### 6. Output Dir Auto-Override

For Weka clusters, if `--output_dir` is not already pointing to a `/weka/` path, mason appends:

```
--output_dir <auto_output_dir_path>/<whoami>/
```

Default base: `/weka/oe-adapt-default/allennlp/deletable_checkpoint`

This is needed because auto-evaluation (launched via `--try_launch_beaker_eval_jobs_on_weka`) requires the output directory to be on Weka so the eval job can access checkpoints. If `auto_output_dir_path` is set to `""`, mason instead requires that you've explicitly disabled auto-evaluation or set a Weka output dir yourself — it raises an error otherwise.

### 7. Multi-node Accelerate Rewriting

For jobs with `--num_nodes > 1` using accelerate, mason rewrites `--num_processes N` into:

```
--num_processes N*num_nodes
--num_machines num_nodes
--machine_rank $BEAKER_REPLICA_RANK
--main_process_ip $BEAKER_LEADER_REPLICA_HOSTNAME
--main_process_port 29400
```

The Beaker env vars (`$BEAKER_REPLICA_RANK`, `$BEAKER_LEADER_REPLICA_HOSTNAME`) are injected at runtime by Beaker's replica orchestration. Your training script sets `--num_processes` per node; mason scales it to the total across all nodes automatically.

### 8. Working Directory Prefix

Unless `--pure_docker_mode` is set, mason prepends `cd <your current working directory> && ` to the command. This ensures the job runs from the same directory you launched from, which matters for relative paths in scripts and configs.

In `--pure_docker_mode`, this prefix is omitted — the container's default working directory is used instead.

---

## What Gets Injected Into the Container

### Environment Variables

Mason sets env vars in this priority order (later entries win if they conflict):

1. **Defaults** (unless overridden by `--env`):

    | Variable | Value | Why |
    |---|---|---|
    | `VLLM_DISABLE_COMPILE_CACHE` | `1` | Torch compile caching is broken; disabling it prevents stale cache errors |
    | `RAY_CGRAPH_get_timeout` | `300` | Prevents Ray compiled-graph timeouts on slow inter-node communication |
    | `NCCL_DEBUG` | `ERROR` | Suppresses noisy NCCL info/warning logs |
    | `VLLM_LOGGING_LEVEL` | `WARNING` | Suppresses noisy vLLM logs |
    | `VLLM_ALLOW_INSECURE_SERIALIZATION` | `1` | Required for vLLM to serialize/deserialize model weights |

2. **Your `--env` overrides** (these replace defaults of the same name).

3. **Secrets** from `--secret NAME=VALUE` (looked up as Beaker secrets, not plaintext values).

4. **API key secrets** — mason checks Beaker's secret store for each of the following keys and injects them if found. It checks `<whoami>_SECRET` first (user-namespaced), falling back to bare `SECRET`:

    `HF_TOKEN`, `WANDB_API_KEY`, `BEAKER_TOKEN`, `OPENAI_API_KEY`, `AZURE_API_KEY`, `AZURE_API_BASE`, `ANTHROPIC_API_KEY`, `SLACK_WEBHOOK_URL`

5. **`PATH`** — your local machine's `$PATH` is injected so the container uses the same Python/conda environment. **Not injected in `--pure_docker_mode`** — the image's own PATH is used instead. This is the key behavioral difference of pure docker mode.

6. **Weka-specific vars** (only when all target clusters are Weka clusters: jupiter, saturn, titan, neptune, ceres, triton, rhea, prometheus):

    | Variable | Value |
    |---|---|
    | `HF_HOME` | `/weka/oe-adapt-default/allennlp/.cache/huggingface` |
    | `HF_DATASETS_CACHE` | `/weka/oe-adapt-default/allennlp/.cache/huggingface` |
    | `HF_HUB_CACHE` | `/weka/oe-adapt-default/allennlp/.cache/hub` |
    | `CHECKPOINT_OUTPUT_DIR` | `/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/<wandb_id>` |

    For multi-node Weka jobs, additionally:

    | Variable | Value | Why |
    |---|---|---|
    | `NCCL_SOCKET_IFNAME` | `ib` | Route NCCL over InfiniBand, not ethernet |
    | `NCCL_IB_HCA` | `^=mlx5_bond_0` | Exclude a specific HCA that causes issues |

7. **Resumability vars** (only for resumable jobs — currently only `grpo_fast.py`):

    | Variable | Value |
    |---|---|
    | `WANDB_RUN_ID` | A random 8-char base-36 ID generated once at mason startup |
    | `WANDB_RESUME` | `allow` |

    The W&B run ID is generated **at import time** — a single call to mason produces one ID shared across all tasks and all retries. This means if Beaker retries your job, W&B resumes the exact same run rather than starting a new one.

### Mounts

- **Weka**: if all clusters are Weka clusters, mounts `/weka/oe-adapt-default` and `/weka/oe-training-default`.
- **Docker socket**: if `--mount_docker_socket`, mounts the host's `/var/run/docker.sock` (needed for Podman/Docker sandbox setups like tmax).
- **Extra datasets**: any `--beaker_datasets mount_path:dataset_id` entries.

---

## Resumability

Only `grpo_fast.py` is resumable. Mason detects this by checking if any command contains a path from `OPEN_INSTRUCT_RESUMABLES`. If it does and `--non_resumable` is not set, mason:

1. Sets `WANDB_RUN_ID` and `WANDB_RESUME=allow` so W&B continues the same run on retry.
2. Auto-generates `--checkpoint_state_dir` (unless you provide a `/weka/` path) so the trainer saves resumable state to Weka.

If you're running a non-resumable script, mason logs a warning that the job won't be resumable but proceeds anyway.

---

## Multi-node Jobs

For `--num_nodes > 1`:

- Mason warns you if any target cluster is not in the interconnect cluster list (`jupiter`, `ceres`, `titan`) and asks for confirmation — cross-rack multi-node without InfiniBand is unreliable.
- The Beaker task spec gets `leader_selection=True`, `propagate_failure=True`, `propagate_preemption=True` — if one replica dies or is preempted, all others are killed too.
- Host networking is enabled by default (required for NCCL). Disable with `--no-host-networking`, but this will break multi-node.

---

## `--pure_docker_mode`

Use this for clusters without NFS (e.g. jupiter2) or when you want the image to be fully self-contained. Effects:

- No `cd <cwd> &&` prefix on the command.
- No host `PATH` injected — the container uses its own environment.
- Weka volumes are still mounted if the cluster is a Weka cluster; this flag only affects PATH and the cwd prefix.

---

## External Users

If mason finds no `~/.beaker/config.yml` and no `BEAKER_TOKEN` env var, it treats you as an external user: it prints the full resolved command but does not submit anything to Beaker. This is useful for seeing exactly what command would run inside the container.

---

## Dataset Cache Internals

The cache system lives in `open_instruct/dataset_transformation.py`. It is fully content-addressed — cache hits and misses are determined entirely by a hash of the inputs, with no manual cache management needed under normal circumstances.

### The Cache Key

`compute_config_hash()` produces a 10-character SHA-256 hash from four inputs:

1. **`DATASET_CACHE_VERSION`** — currently `"v9"`. Bumped manually in the code when transformation logic changes significantly enough to invalidate all existing caches globally.
2. **Dataset configs** — for each dataset in the mixer: name, split, transform functions, sample count/fraction, seed, and all other non-`None` `DatasetConfig` fields.
3. **Tokenizer config** — all non-`None` fields of `TokenizerConfig`.
4. **Chat template hash** — a separate SHA-256 of the serialized chat template string, since the template determines how messages are tokenized.

Any change to any of these produces a different hash → cache miss → full re-tokenization.

### Two Cache Backends

Controlled by `--dataset_cache_mode` in the training command (default: `"local"`):

**`local`** — saves/loads from a directory on disk structured as `<cache_dir>/<hash>/`

- On Beaker: hardcoded to `/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache/<hash>/`
- Locally (during mason pre-caching): defaults to `local_dataset_cache/<hash>/` relative to cwd
- Hit check: `os.path.exists(cache_path)`
- On a hit, loads with `Dataset.load_from_disk()` and also restores per-dataset statistics from `dataset_statistics.json` if present

**`hf`** — pushes/pulls from a private HuggingFace dataset repo `<hf_entity>/dataset-mix-cached`, using the config hash as the git revision

- Hit check: `revision_exists(repo_name, config_hash, repo_type="dataset")`
- On a miss, tokenizes, pushes to Hub with the hash as revision, then re-downloads to populate the local HF cache
- Also pushes a model card to the revision with the full config for auditability

### The Full Flow

```
mason (local machine)
  └─ runs: python grpo_fast.py [args] --cache_dataset_only
       └─ compute_config_hash(dataset_configs, tokenizer_config)  →  hash = "a3f9b2c1d0"
            └─ LocalDatasetTransformationCache.load_or_transform_dataset()
                 ├─ cache exists at /weka/.../deletable_open_instruct_dataset_cache/a3f9b2c1d0/
                 │    └─ load from disk, return immediately  ✅
                 └─ cache missing
                      └─ tokenize all datasets → save to disk  🔄

Beaker job (container)
  └─ runs: python grpo_fast.py [args]   (no --cache_dataset_only)
       └─ compute_config_hash(...)  →  same hash "a3f9b2c1d0"
            └─ cache exists at /weka/...  →  load from disk, skip tokenization  ✅
```

### What Triggers a Cache Miss

| Change | Effect |
|---|---|
| Different dataset name, split, or sample count | New hash → re-tokenize |
| Different transform function or its args | New hash → re-tokenize |
| Different `dataset_config_seed` | New hash → re-tokenize |
| Different model (different tokenizer or chat template) | New hash → re-tokenize |
| `DATASET_CACHE_VERSION` bumped in code | All existing hashes invalid → re-tokenize everything |

### Arguments That Control Caching Behavior

These are **training command args** (after `--`), not mason args:

| Argument | Default | Effect |
|---|---|---|
| `--dataset_cache_mode` | `local` | `local` = disk on Weka; `hf` = HuggingFace Hub |
| `--dataset_local_cache_dir` | `local_dataset_cache` | Base directory for local cache. Overridden to the shared Weka path on Beaker automatically. |
| `--dataset_skip_cache` | off | Skip both reading from and writing to cache — always re-tokenizes, result is not saved. Useful for debugging transform logic without polluting the cache. |
| `--dataset_config_hash` | `None` | Manually supply a hash to load a specific cached version, bypassing hash computation. Useful for loading a known-good cache when you've changed args that would otherwise produce a miss. |

To **force re-tokenization and update the cache**, delete the cache directory on Weka:

```bash
rm -rf /weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache/<hash>/
```

Then re-run mason (without `--no_auto_dataset_cache`) — it will re-tokenize and write a fresh cache at the same path.

---

## Quick Arg Reference

| Argument | Default | Notes |
|---|---|---|
| `--cluster` | *(required)* | Space-separated; multiple clusters = job can land on any |
| `--budget` | *(required)* | e.g. `ai2/oe-adapt` |
| `--gpus` | `0` | Per node |
| `--num_nodes` | `1` | |
| `--shared_memory` | `10.24gb` | |
| `--image` | `ai2/cuda11.8-cudnn8-dev-ubuntu20.04` | |
| `--priority` | `normal` | `low` / `normal` / `high` / `urgent` |
| `--preemptible` | off | |
| `--pure_docker_mode` | off | See above |
| `--mount_docker_socket` | off | Needed for Podman sandboxes |
| `--workspace` | user default | |
| `--timeout` | none | e.g. `2h30m` |
| `--max_retries` | `0` | Beaker-level task retries |
| `--non_resumable` | off | Force-disable resume even for grpo_fast.py |
| `--no_auto_dataset_cache` | off | Skip local pre-caching; required on macOS |
| `--auto_output_dir_path` | `/weka/.../deletable_checkpoint` | Set to `""` to disable output_dir override |
| `--auto_checkpoint_state_dir` | `/weka/.../deletable_checkpoint_states` | |
| `--beaker_datasets` | `[]` | Format: `mount_path:beaker_id` |
| `--env NAME=VALUE` | `[]` | Repeatable; overrides defaults |
| `--secret NAME=VALUE` | `[]` | Repeatable; looked up as Beaker secrets |
| `--hostname` | `None` | Pin to specific hosts instead of cluster |
| `--no-host-networking` | off | Breaks multi-node |

---

## Overriding Args Without Editing Scripts

Use `scripts/update_command_args.py` to patch any arg (mason or training) on the fly:

```bash
python scripts/update_command_args.py scripts/train/tulu3/grpo_fast_8b.sh \
    --cluster ai2/saturn \
    --priority normal \
    --image myuser/my_image \
    --non_stop_penalty False | uv run bash
```

This prints the modified script to stdout and pipes it to bash, leaving the source file unchanged.
