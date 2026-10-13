# Upstream MILES images and shared TOML

This experimental path builds an unmodified checkout of
[`radixark/miles`](https://github.com/radixark/miles) on its own upstream binary
environment. Select FSDP, Megatron, or torchtitan instead of the OLMo-core trainer.
The existing `--miles` workflow and its private runtime pins remain independent.

The checked-in lock is a **candidate**. Matching dependency fingerprints and
passing CPU tests do not establish GPU training correctness. Qualify a short
rollout/train/weight-update/save/resume lifecycle on the intended hardware before
long training. No backend or cluster has yet been qualified for this new wrapper.

## Configuration compatibility

The adapter accepts a shared subset of the existing TOML format. It requires an
explicit `trainer.backend = "fsdp"`, `"megatron"`, or `"torchtitan"`. It does not
compile through the Core RunSpec or inherit its defaults.

| Existing setting | Upstream meaning |
| --- | --- |
| `model.source`, `model.format = "hf"` | Local/shared HF directory passed as `hf_checkpoint` |
| `output.root` | Checkpoints saved in `<root>/checkpoints` |
| `trainer.gpus`, `trainer.trainer_num_nodes` | Actor GPU and node counts |
| `inference.gpus`, `inference.rollout_gpus_per_node` | Rollout GPUs and physical allocation size |
| `inference.placement_mode` | `colocated` or `disaggregated` |
| `training.num_rollouts` | `num_rollout` |
| `inference.samples_per_prompt`, `inference.max_response_length` | `n_samples_per_prompt`, `rollout_max_response_len` |
| `optimizer.learning_rate` | `lr` |
| `trainer.activation_recompute` | `gradient_checkpointing` for FSDP/torchtitan; use native recompute settings for Megatron |
| `data.prompt_data`, `data.eval_prompt_data` | Already prepared data paths; eval values alternate names and paths |
| `[miles]` | Native flags accepted by the selected upstream backend |

Native option names also work in `[training]`, `[trainer]`, `[inference]`,
`[optimizer]`, `[async]`, and `[tracking]`. Use underscore spellings and typed
TOML values. Conflicting aliases, unknown options, wrong types, and invalid choices
raise errors; fields are never silently dropped. New upstream flags become
available when the image and its exported parser schemas are refreshed together.

`[core]`, Core compiler caches, managed judges, records/selection, background
evaluation, model conversion, and `data.tasks` preparation are not supported by
this adapter. Existing Core-heavy run files therefore need edits; they are not
drop-in compatible. Prepare data separately and use upstream reward functions.
Megatron needs its native architecture arguments and converted checkpoint paths
in `miles.load`/`miles.ref_load`; see the
[upstream backend guide](https://miles.radixark.com/docs/user-guide/training-backends).
The [Megatron candidate example](../../configs/miles-upstream/megatron.toml)
uses Qwen3-0.6B architecture settings from the locked upstream recipe. Convert
weights inside the runtime before launching, using upstream's
`tools/convert_hf_to_torch_dist.py` with the arguments produced by
`python miles/utils/external_utils/model_args_utils.py qwen3-0.6B`.

The adapter validates option syntax against schemas captured from the actual
image, not Open Instruct's Core option snapshot. Upstream still performs its
own backend/model/combination validation when training starts. An accepted flag
does not guarantee identical semantics across trainers. Dynamic plugin-specific
flags absent from the exported schemas need native upstream commands.

Relative model, checkpoint, and dataset paths resolve against the TOML file's
directory on the submitting host. Use absolute worker-visible paths for Beaker.
`model.source` is a directory, not a Hugging Face repository ID. Native YAML
config overrides are rejected so they cannot replace validated TOML values.

## Build and refresh

The lock records an immutable upstream image digest, a public MILES commit, the
platform, and upstream's dependency-build fingerprint. The refresh operation
checks that the selected source and image have matching dependency inputs before
writing the lock. A mismatch requires building upstream's Docker recipe first;
updating only MILES sources is insufficient in that case.

```bash
# Optional: select a newer candidate, then review the lock diff.
python -m scripts.miles_upstream.build_image refresh --ref main

# Inspect build arguments without Docker or network access.
python -m scripts.miles_upstream.build_image build --dry-run

# Vanilla upstream checkout and dependencies; no TOML adapter.
python -m scripts.miles_upstream.build_image build --target runtime

# Add the small adapter and export each backend's option schema (CPU build host).
python -m scripts.miles_upstream.build_image build \
  --target application --output-dir runs/upstream-image

```

Both targets clone directly from the public upstream repository without GitHub
credentials. The `runtime` target can be used interactively or as a downstream
Docker base; the `application` target adds only the TOML adapter and CPU validation
helpers, not the Core application or trainer. `/root/miles` contains the checkout.
`/opt/miles-upstream` contains the lock, package inventory, and dependency commits.

Build identity includes the lock, Dockerfile, and adapter sources. Application
builds export `image.json` and
`schemas/{fsdp,megatron,torchtitan}.json` to the output directory.
Distribute these files with the image. Rebuild after source or
adapter changes; do not `git pull` inside a validated image.

Parser imports load TransformerEngine and need NVIDIA driver libraries. The
build uses the upstream image's bundled CUDA compatibility library only while
capturing schemas on the CPU. This does not change the workers' library search
path or qualify the image's CUDA kernels. The schemas are baked into the image
and exported without starting a container.

After committing changes, publish an application image with
`build --publish-workspace ai2/YOUR_USERNAME`. The receipt then includes its
immutable Beaker image ID. Publication and training submission are separate.

## Plan and launch a single-node run

```bash
cp configs/miles-upstream/fsdp.toml runs/upstream-fsdp.toml
# Edit model/data/output paths and mounts.
python -m scripts.miles_upstream.cli plan runs/upstream-fsdp.toml \
  --schema-dir runs/upstream-image/schemas
python -m scripts.miles_upstream.cli validate runs/upstream-fsdp.toml \
  --schema-dir runs/upstream-image/schemas

# Render the actual allocation without submitting it.
python -m scripts.miles_upstream.cli run runs/upstream-fsdp.toml \
  --schema-dir runs/upstream-image/schemas --image IMMUTABLE_IMAGE_ID --dry-run

# Commit changes before submitting.
./scripts/train/build_image_and_launch.sh --miles-upstream runs/upstream-fsdp.toml \
  --schema-dir runs/upstream-image/schemas --image IMMUTABLE_IMAGE_ID
```

The launcher supports one node with up to eight GPUs, one actor pool, and one
rollout pool. Disaggregated allocations must equal trainer plus rollout GPUs;
colocated allocations equal trainer GPUs and omit `inference.gpus`.
Set `inference.rollout_gpus_per_node` to the allocation size. Extra critic/eval
pools, external engines, Kubernetes, and multi-node orchestration require native
launches. The wrapper starts a local Ray cluster and executes upstream `train.py`
or `train_async.py`, preserving native arguments.

Defaults are Holmes, high priority, unallocated scheduling (`min_runtime = 0`),
a 30-minute timeout, and no automatic retry. Automatic retry is opt-in and requires
`miles.load` to equal the save directory, a positive `save_interval`, and persistent
WEKA storage. Verify upstream resume behavior with a short interruption test.
Jupiter needs a separate CUDA/driver qualification; this wrapper does not assume
the Core image's compatibility-library workaround applies.

`launch.secrets` contains Beaker secret names. Mount every required WEKA path.
Logs and the frozen native plan go to Beaker results; checkpoints go to the
configured save directory. Image metadata and an in-container identity check
reject schemas from a different application build.

## Validation before qualification

Run the CPU contracts and lint checks:

```bash
uv run pytest tests/miles_upstream open_instruct/test_miles_options.py open_instruct/test_miles_runtime_sources.py
uv run ruff check scripts/miles_upstream tests/miles_upstream open_instruct/miles/configuration/options.py
uv run ruff format --check scripts/miles_upstream tests/miles_upstream open_instruct/miles/configuration/options.py
```

Then exercise FSDP and Megatron on a small supported dense model. Verify finite
losses and gradients, multiple optimizer updates, rollout weight refresh, a durable
checkpoint, and successful resume. Megatron also needs conversion validation.
Measure startup/compilation, sustained throughput and memory, and checkpoint/eval
overhead separately. Keep BF16, efficient attention, batching and data-loading
settings consistent with the selected upstream recipe. Record image IDs, configs,
and Beaker runs before changing the lock's qualification status.
