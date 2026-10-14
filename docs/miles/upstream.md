# Upstream MILES images and shared TOML

This experimental path builds an unmodified checkout of
[`radixark/miles`](https://github.com/radixark/miles) on its own upstream binary
environment. Select FSDP, Megatron, or torchtitan instead of the OLMo-core trainer.
The existing `--miles` workflow and its private runtime pins remain independent.

The checked-in lock is a **candidate**. Matching dependency fingerprints and
passing CPU tests do not establish GPU training correctness. Qualify a short
rollout/train/weight-update/save/resume lifecycle on the intended hardware before
long training. No backend or cluster has yet been qualified for this new wrapper.
The Qwen3-0.6B FSDP and Megatron smoke tests below have passed rollout, training,
weight refresh, and checkpoint saving on Holmes. Resume, parallel training across
multiple GPUs, and torchtitan remain unverified.

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
Megatron needs its native architecture arguments. At the pinned upstream
revision, `miles.ref_load` can point to an HF directory for native initialization
through mbridge, or to an already converted Megatron checkpoint. Set `miles.load`
to the run's checkpoint directory for subsequent resume; see the
[upstream backend guide](https://miles.radixark.com/docs/user-guide/training-backends).
The [Megatron smoke example](../../configs/miles-upstream/megatron.toml)
uses Qwen3-0.6B architecture settings from the locked upstream recipe and
initializes directly from HF weights. A separate conversion job is optional.
To prepare a Megatron release checkpoint instead, convert weights inside the
runtime using upstream's
`tools/convert_hf_to_torch_dist.py` with the arguments produced by
`python miles/utils/external_utils/model_args_utils.py qwen3-0.6B`.
Conversion requires a GPU. The direct-HF smoke recipe uses one Megatron trainer
GPU and one SGLang GPU, BF16, FlashAttention, dynamic batches capped at 4096
tokens, four samples per prompt, and four GRPO updates. It limits SGLang's KV
cache to 32768 tokens and disables prefill graph capture to keep this small
validation bounded; decode graphs remain enabled. Initial weight equality
checking compares the Megatron-to-SGLang transfer against the HF-loaded model.
The four-update GPU validation below exercised this direct-HF path successfully.

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
checkpoint, and successful resume. For Megatron, validate HF initialization and
weight transfer; also validate standalone conversion if using converted inputs.
Measure startup/compilation, sustained throughput and memory, and checkpoint/eval
overhead separately. Keep BF16, efficient attention, batching and data-loading
settings consistent with the selected upstream recipe. Record image IDs, configs,
and Beaker runs before changing the lock's qualification status.

### Qwen3-0.6B FSDP smoke test (2026-10-13)

[The four-update run](https://beaker.org/ex/01M4W4CMXVGB6S2SXT1QXB7XST)
exited successfully on one Holmes node with two B300 GPUs, high priority,
`min_runtime = 0`, no automatic retry, and a 30-minute timeout. One GPU trained
with upstream FSDP and the other ran SGLang. The
[FSDP example](../../configs/miles-upstream/fsdp.toml) records the exercised
settings with placeholder WEKA paths.

The application image was `01M4W3VJ926WHT53SDGE69CBBA`, build
`12ea4ee6e13e06dfc982007fa5f4cc68c462b4e6978347794aafc38556238022`, with
MILES `bef2b237c42f1d6aef30b11df5d332f5ad228338`. Inputs were
`Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca` and the first
128 rows of `zhuzilin/gsm8k`'s `train.parquet` at
`0cbd9f31d91ac21a7613dcbc7fef992adac459ae`, written as JSONL with `messages`
and `label` fields. Preparation ran separately on Saturn with no protected
minimum runtime; a lightweight Python container avoided pulling the training
image just to download inputs.

The run produced 128 responses across four batches (eight prompts, four samples
each, up to 512 response tokens, Qwen thinking disabled). Saved samples have
finite rollout log probabilities and both zero and one rewards in every batch.
All four optimizer updates had finite losses and nonzero gradient norms
(1.72–2.46). Initial weight synchronization passed upstream's exact tensor
comparison, and subsequent rollouts used weight versions 2, 3, and 4 without
mixing versions. FSDP saved model, optimizer, scheduler, RNG, and progress
checkpoints at updates 2 and 4. This is a lifecycle smoke test, not evidence of
improved model quality or multi-GPU FSDP sharding correctness.
An independent [CPU inspection](https://beaker.org/ex/01M4W5K9RBW8NYGDA3V2R95CPM)
confirmed nonempty model, optimizer, and scheduler shards, their metadata,
the RNG files, and the expected `iteration`/`next_rollout_id` values in both
checkpoints. It did not deserialize and restore their training state.

The [restart check](https://beaker.org/ex/01M4W5A4NSGV9NN7W1VXDGXJGB) was canceled
after five minutes in the queue: scheduler events reported no Holmes node with
two available slots. It never started. **Checkpoint loading and continued
training remain unverified**, as do preemption recovery and torchtitan.
The lock remains a candidate. At this upstream revision, FSDP leaves the
`global_step` and `micro_step` metadata counters at zero; resume position uses
`iteration` and `next_rollout_id`. Do not use those unused counters as evidence
that optimizer updates did or did not run.

Measured costs for this small run:

| Phase | Observed time |
| --- | --- |
| Cold Beaker image pull | 7 min 39 s |
| Process start to first rollout | About 5 min 58 s, including imports, model setup, graph capture, and weight checks |
| SGLang prefill / decode graph capture | 81.4 s / 4.3 s |
| First training step, including initial compilation | 43.1 s |
| Three subsequent training steps | 1.57–1.69 s each; actor training alone 5.8k–7.6k tokens/s |
| Checkpoint writes | About 16.2 s and 12.1 s |

The trainer reported about 22.5 GiB used before its post-step cache clear and
12.8 GiB afterward; these are sampled readings, not measured peak memory.
SGLang's 0.4 static-memory fraction reserved about 105.5 GiB of KV cache on a
B300, far more than this test used. For small smoke runs, consider
`sglang_max_total_tokens = "32768"` and
`sglang_disable_prefill_cuda_graph = true` to limit cache reservation and startup
work. These settings retain decode CUDA graphs. The tiny sample of warm steps
is not a sustained-throughput benchmark. No evaluation pass was configured.

### Qwen3-0.6B Megatron smoke test (2026-10-13)

[The Megatron run](https://beaker.org/ex/01M4W7DE75JDVZ41RD3F8Q35TT) exited
successfully on one Holmes node with two B300 GPUs: one Megatron trainer and one
SGLang rollout worker. It used high priority, `min_runtime = 0`, no automatic
retry, and a 30-minute timeout. The image, MILES revision, model, and dataset
were identical to the FSDP run above; Megatron-Core was
`fd15ee20a4f03b03680529baf4b8f7eeed64df1d`. The
[Megatron example](../../configs/miles-upstream/megatron.toml) records the exercised
settings with placeholder WEKA paths. No upstream source changes were needed.

Native mbridge initialization loaded the HF model directly into Megatron.
The initial Megatron-to-SGLang transfer passed exact equality checking for all
226 tensors. Four GRPO optimizer updates completed with finite losses and
nonzero gradient norms (2.01–2.82). The 128 saved responses had finite rollout
log probabilities and both zero and one rewards in every batch. Rollouts used
weight versions 1 through 4 without mixing versions. Megatron reported successful
`torch_dist` checkpoint saves after updates 2 and 4, including distributed
optimizer state. Its directories are `iter_0000001` and `iter_0000003` because
this backend passes the zero-based rollout ID to the checkpoint writer.
The optional independent file inspection was canceled before execution on
[Saturn](https://beaker.org/ex/01M4W8KEF73WC9FTT65RD5NW5F) and then
[Jupiter](https://beaker.org/ex/01M4W8NWKBPC06EMVQNXQQ3XXS): scheduler events
reported unavailable slots, with higher queued work also blocking Jupiter.
Checkpoint-save evidence for this run is therefore the trainer's completion logs;
the files have not been independently inspected or restored.

Measured costs for this small run:

| Phase | Observed time |
| --- | --- |
| Cold Beaker image pull | 9 min 31 s |
| Process start to first rollout | About 6 min 1 s, including imports, model setup, graph capture, and weight checks |
| SGLang decode graph capture | 68.6 s; prefill graph capture disabled |
| First training step, including initial compilation | 79.1 s |
| Three subsequent training steps | 0.68–1.03 s each; actor training alone 11.8k–15.3k tokens/s |
| Checkpoint writes | About 21.8 s and 14.8 s |

The trainer reported 19.2–24.2 GiB used before cache clearing and about 15.5 GiB
afterward; these are sampled readings, not measured peak memory. SGLang reserved
3.5 GiB of KV cache with the 32768-token cap. The run took about 8 min 28 s after
the container started. The three warm steps are too small a sample for a
sustained-throughput benchmark, and no evaluation pass was configured.
Checkpoint restoration, preemption recovery, standalone conversion, and
Megatron tensor/pipeline/data parallelism across multiple trainer GPUs were not
tested. The runtime lock remains a candidate.
