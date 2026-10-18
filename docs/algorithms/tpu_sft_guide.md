# Running SFT on TPU: a known-good starting point

Olmo 3.5 small (the 4T EMO base) on Dolci-Think with H045's recipe, on **one TPU v4 `4x4x8` slice (128 chips)** of
the `gke-tpu-v4` cluster, through MaxText + Tunix.

Measured cost and outcome: 11,768 steps of 16 × 65,536 tokens at a median 3.94 s/step, 13.5 h wall clock from submit
to the last step. The loss matches H045's GPU runs (parity windows in [open-instruct#1933](https://github.com/allenai/open-instruct/issues/1933)). The GPU recipe takes 5 h 54 m on
2×8 H100, so the TPU path is slower per run. Its advantage is capacity: four of these runs fit at once without
touching Beaker.

The recipe is `scripts/tpu/maxtext/k8s/olmo35-sft-4x4x8/`.
Its `kustomization.yaml` header records why each setting is what it is; read it before editing. Olmo 3 7B has its
own overlay, `k8s/instruct-sft-4x4x8/`; its base `k8s/olmo3-sft/` targets a `2x2x4` slice, which no longer exists. `scripts/tpu/maxtext/README.md` lists the MaxText patches.

## Example runs

| Run | What it is |
|---|---|
| `olmo35-sft-h045-tpu-full-r2` | This recipe, verbatim: the TPU replica of the ledger's H045 anchor (arm `anchor-emo-olmo35-tpu`). 0 restarts, final CE 0.9245. HF export: `/weka/oe-adapt-default/abhishekr/tpu-posttrain/hf/olmo35-sft-h045-tpu-full-r2-step11768`. |
| H067, H072 (olmo-post-training-ledger) | The same recipe with other data: arms, seeds, resumes and their checks are recorded in those hypothesis files and in [open-instruct#1976](https://github.com/allenai/open-instruct/issues/1976) and [#1977](https://github.com/allenai/open-instruct/issues/1977). |

## Before you start

- **IAM on the `ai2-olmo` GCP project**, e.g. `roles/container.developer`.
- **A network the cluster accepts.** [allenai/olmo-gcp-infra's README](https://github.com/allenai/olmo-gcp-infra#connecting-to-gke-tpu-v4)
  lists them: the office IPs, its Tailscale exit nodes, and Beaker sessions on `ai2/hammond`. From anywhere else, use the
  DNS endpoint (`--dns-endpoint`). Then:

  ```bash
  gcloud container clusters get-credentials gke-tpu-v4 --zone us-central2-b --project ai2-olmo
  kubectl -n olmo get workloads
  ```

  `kubectl` needs `gke-gcloud-auth-plugin` (`gcloud components install gke-gcloud-auth-plugin`, or a binary in `~/.local/bin`).
- **gcloud logins expire about daily, and kubectl then fails quietly.** Watchers go blind without an error. Agents
  use the service account `tpu-posttrain-agent@ai2-olmo.iam.gserviceaccount.com`: export
  `CLOUDSDK_CORE_ACCOUNT` to it in every process, background watchers included. Move
  `~/.kube/gke_gcloud_auth_plugin_cache` aside when switching identity, or kubectl keeps the cached one. People log in
  with their own account.
- **Where things live.** Everything is in `us-central2`. Data, tokenizers and checkpoints are in `gs://ai2-olmo-dev/post-training/`,
  and the nodes mount it at `/mnt/post-training` through gcsfuse. Images are in
  `us-docker.pkg.dev/ai2-olmo/olmo/maxtext-posttrain`.

## What the cluster is

`gke-tpu-v4` holds 2,048 reserved v4 chips at 4 per host: four `4x4x8` slices (128 chips), one `8x8x8` (512) and one
`8x8x16` (1,024). A slice runs one job at a time. Kueue admits jobs from four queues in one cohort:

| Queue | Who | Preempted by |
|---|---|---|
| `hero` | the main pretraining run; owns all the quota | nothing |
| `dev` | tests | hero |
| `post-training`, `data-adaptation` | us; share what is left | hero, dev |

**Use `post-training`.** Its jobs run on borrowed quota and are preempted whenever hero or dev needs the slice, so
checkpoint often enough to lose little. Never hold a slice in a way that delays hero.

The recipe needs exactly a `4x4x8`: its mesh puts one 64K row on 8 chips, so 128 chips hold its 16 rows. The `8x8x8`
runs the same mesh at 64 rows per step, which is a different recipe (H078's 4x batch). Running the 16-row recipe on a
big slice needs `olmo35-cp-ep`, which splits each row 32 ways and changes the MoE load-balance term. Measured meshes are
in [Changing the recipe](#changing-the-recipe).

## Cost: use the `4x4x8`

Same recipe and tokens per run (12.3B), from TensorBoard `perf/*` on finished runs (2026-10-15 to 10-18):

| slice | rows/step | median s/step | tokens/s/chip | TFLOP/s/chip | steps over 3x median | wall, step 1 to end | chip-hours |
|---|---|---|---|---|---|---|---|
| `4x4x8` (A0', H076, H082; 6 runs) | 16 | 3.90 | 2,100 | 10.5 | 2-6 per run | 13.2-13.4 h | ~1,700 |
| `8x8x8` (H078, 2 runs) | 64 | 4.27 | 1,920 | 9.6 | 322-384 (11-13%) | 5.8-6.3 h | ~3,000-3,200 |

Startup before step 1 (pods, restore, an 11-12 min compile) adds 20-35 min on either slice, paid again at every
resume.

- The `8x8x8` finishes 2.2x sooner but costs 1.8-1.9x the chip-hours. Its steady-state rate per chip is only 9% lower:
  the rest is stalls, where one step in eight takes 20-35 s instead of 4.3 s (2.1-2.5 h per run). The stall sits
  inside the device step (batch handed over to loss read back on host 0), not in data loading or checkpointing; the
  cause is not known. Without it the `8x8x8` would cost ~1.1x the chip-hours at 3.5x the speed, so fix it before
  using a big slice again.
- Both slices run at ~4% of v4's 275 TFLOP/s bf16 peak by MaxText's own FLOP count. No mesh, remat or per-device
  batch sweep has been run (see [Changing the recipe](#changing-the-recipe)).
- `jax` `memory_stats()` peaks (0.9-2.3 GiB) leave out XLA temporaries: the same `8x8x8` config needed 32.17 GB at
  compile with `ragged_buffer_factor=4.0`. Read HBM headroom from the compile error or xprof, not from those lines.

## 1. Image

```
us-docker.pkg.dev/ai2-olmo/olmo/maxtext-posttrain:51f8cc4b-p907d86f3-rfb46e215e
```

This tag is r2's image (`51f8cc4b-p907d86f3`, built from `image-olmo35/`) plus `image-olmo35-rf/`. That patch adds a
fix for resuming from a checkpoint (without it, restore runs out of HBM) and batch-hash logging. Training numerics are
unchanged: a replay produced bit-identical CE and 9,952 of 9,952 equal batch hashes. Use it for every new run. The
Olmo 3 image is built by `image/build.sh <maxtext commit>`.

To build a new image, run `build.sh` from any host with `gcloud` (a Beaker session works). Cloud Build carries the context base64-encoded inside the build
config. Builds use Cloud Build's default service account. Set `BUILD_SERVICE_ACCOUNT` to a full
`projects/<project>/serviceAccounts/<email>` resource only when the caller is authorized to use that account.

## 2. Base checkpoint

The loader reads an Orbax parameter tree (`…/0/items`) from GCS.

- **Olmo 3.5 (OLMo-core distcp).** This runs in two stages, on CPU:

  ```bash
  python scripts/tpu/maxtext/olmo35/dump_olmocore_masters.py <step dir> <masters dir>              # needs olmo_core
  python scripts/tpu/maxtext/olmo35/convert_olmocore_to_maxtext.py <step dir> <masters dir> <out>   # needs maxtext
  gcloud storage cp -r <out>/0 gs://ai2-olmo-dev/post-training/checkpoints/<name>/
  ```

  It keeps OLMo-core's padded 100,352 vocabulary, whose padded rows were trained. The 4T EMO base is already at
  `gs://ai2-olmo-dev/post-training/checkpoints/olmo35-small-4t-emo-step5961/0/items`.
- **Olmo 3 (HF).** Use MaxText's converter with `model_name=olmo3-7b`. `olmo3-7b-pt` has no YaRN, so it is wrong for
  the 1025 base.

  ```bash
  JAX_PLATFORMS=cpu python -m maxtext.checkpoint_conversion.to_maxtext model_name=olmo3-7b base_output_directory=<out> \
    scan_layers=True use_multimodal=false hardware=cpu skip_jax_distributed_system=true \
    checkpoint_storage_use_zarr3=1 checkpoint_storage_use_ocdbt=1 --hf_model_path=allenai/Olmo-3-1025-7B --save_dtype=bfloat16
  ```

**Check any new conversion numerically** against the source model before training on it. Key and shape checks pass on
broken conversions.

## 3. Data and tokenizer

MaxText streams parquet files. Its shuffle only permutes files and mixes a small buffer, and hub shards are grouped by
source, so shuffle globally first. `olmo35/build_full_dolci_think.py` (Olmo 3.5) and `data/build_full_dolci.py`
(Olmo 3) write N shuffled shards from one dataset revision and use the standard HF cache settings. Pin the input with
`--revision <dataset commit>`, or use `--data-files '<local parquet glob>'` for local files; the output directory must be
new or empty. Upload the shards to `gs://ai2-olmo-dev/post-training/data/<name>/`. Ledger experiments shard each
arm and seed into 512 files with `experiments/h067/shard.py` in olmo-post-training-ledger.

The tokenizer is a directory on the bucket. For Olmo 3.5, `olmo35/tok_oi35.py` saves H045's tokenizer and template
(`allenai/dolma2-tokenizer-olmo35` @8b971706, `olmo35`). For Olmo 3, save
`PreTrainedTokenizerFast.from_pretrained("allenai/olmo-3-tokenizer-instruct-dev")` (see Traps).

## 4. Launch

Copy `k8s/olmo35-sft-4x4x8/` to a new directory. Then rename the JobSet and both ConfigMaps (in `jobset.yaml` and
`kustomization.yaml`) to a new unique name, and edit the env block:

| Variable | Meaning |
|---|---|
| `RUN_NAME`, `OUTPUT_DIR` | Checkpoints go to `$OUTPUT_DIR/$RUN_NAME/checkpoints/<step>/`. |
| `LOAD_PARAMETERS_PATH`, `TOKENIZER_PATH`, `TRAIN_FILES` | Step 2 and step 3 outputs (`/mnt/post-training/...` = `gs://ai2-olmo-dev/post-training/...`). |
| `TOTAL_DEVICES`, `GLOBAL_BATCH`, `PER_DEVICE_BATCH` | 128 chips × 0.125 = 16 sequences per step, with no gradient accumulation. The launcher derives accumulation and fails if the numbers don't divide. |
| `SEQ_LEN` | 65,536. Tokens per step = `GLOBAL_BATCH × SEQ_LEN`. |
| `STEPS`, `LR_SCHEDULE_STEPS` | Steps trained, and the horizon the LR decays over (warmup `WARMUP_FRAC` of it). Set `STEPS` below the horizon to run the start of a longer recipe, e.g. for a loss-parity check. |
| `NUM_EPOCH` | Passes the stream holds. `STEPS` decides what is trained; a pass shorter than `STEPS` repeats data. |
| `EXTRA_ARGS` | MaxText flags. The mesh is `custom_mesh_and_rule=olmo35-ep-as-cp ici_fsdp_parallelism=16 ici_expert_parallelism=8`: batch over FSDP, and the expert axis doubles as context parallelism outside the MoE. `checkpoint_period` and `max_num_checkpoints_to_keep` set what is saved. |
| `images: newTag` | The image tag from step 1. |

```bash
kubectl --context gke_ai2-olmo_us-central2-b_gke-tpu-v4 apply -k <your overlay dir>
kubectl -n olmo get workloads              # ADMITTED True = it has a slice
kubectl -n olmo get jobset <name>          # TERMINALSTATE, RESTARTS, SUSPENDED (a hero preemption suspends it)
kubectl -n olmo get pods -l jobset.sigs.k8s.io/jobset-name=<name>
```

To stop a run, deactivate its Kueue workload:
`kubectl -n olmo patch $(kubectl -n olmo get workloads -o name | grep "/jobset-<name>-") --type merge -p '{"spec":{"active":false}}'`.
Setting the JobSet's `spec.suspend` does nothing while Kueue has it admitted: Kueue resets it, and a watcher that
relied on it left a run training on the `8x8x8` for 4.5 h after its stop rule fired.

Startup takes ~35 min: the 32 pods start a few minutes after admission, then the model compiles and the first step runs. `k8s/watch_jobset.sh <jobset> <run> [max minutes]`
saves every pod's log when the JobSet ends. Pod logs vanish once the JobSet is deleted
or restarted, so save them before either.

## 5. Read the loss

Read the step loss only from the `LOSS-PARTS step=N ce=… moe_lb=… z=… tokens=…` line, which one pod logs. Each pod's
`completed step: N, … loss:` line is that pod's own value: on one step the 32 pods ranged 1.28–1.68. Compare `ce=`
with H045's `train/CE loss`, never Tunix's `loss`, which adds the MoE load-balancing and z-losses.

A healthy Olmo 3.5 run: `REQUIRED_RBF` (the MoE buffer factor that would have dropped no token) stays below the
configured `ragged_buffer_factor=2.5`. r2's maximum was 2.24.

`REQUIRED_RBF` is logged every step by JAX process 0 only, and that is often not pod `slice-0-0`: find the pod whose log
has it. Kubelet keeps about 700 steps of each pod's log, and the agent service account cannot read Cloud Logging. So
if a rule needs every step's value, stream that pod with `kubectl logs -f` from step 1. A watcher polling
`kubectl logs` every 5 minutes lost 14 steps at one rotation (H078 bb4-lr1e4-s0), and the run had to be relaunched.

## 6. Checkpoints, preemption and resume

- `enable_checkpointing=true` is required even if you want no checkpoints. Without it, `jax.distributed` is never
  initialised on multi-host TPU and Tunix crashes.
- Tunix always saves the final step. An Olmo 3.5 checkpoint with optimizer state is ~137 GB. Agents delete nothing on
  GCS; ask the run's owner.
- After a preemption, JobSet restarts the run from the last checkpoint, on the same overlay. A resume replays the data
  pipeline to the restored step at ~1.4 s per skipped batch, which is hours late in a run. Resume only on the
  restore-fix image (step 1).

## 7. Export to HF and evaluate

```bash
gcloud storage cp -r gs://ai2-olmo-dev/post-training/runs/<run>/checkpoints/<step>/model_params <local dir>
python scripts/tpu/maxtext/olmo35/export_maxtext_to_hf.py <local dir>/model_params <reference HF dir> <out>
```

The reference directory supplies tensor names, `config.json`, modeling code, tokenizer and chat template. Use an HF
export of the same model with the matching tokenizer and chat template, and choose `<out>` under your own output
location. One existing reference export is H045's:
`/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-4t/runs/h045-anchor-emo-olmo35-s34521/hf_step11768`.
Its `config.json` is the HF model config, unlike the run directory's OLMo-core training config. The exporter is
bit-exact on all 23,441 tensors. For Olmo 3, use `export_hf.sh`, which also takes the base model's `config.json`,
because MaxText's own is wrong.

Evaluate the HF export with olmo-eval on Beaker GPUs, as for a GPU run (see [the GPU guide](sft_running_guide.md)).
HF Olmo 3.5 needs `fla`, so every pass runs on GPU. Where exports and checkpoints go, and what to keep, is the ledger's
`storage` skill (olmo-post-training-ledger `skills/storage/SKILL.md`).

## Traps

Each of these trained silently on wrong data or a wrong model before it was found.

- **Dropped EOS targets.** Olmo tokenizers have `bos == eos`, and MaxText's target shift ignored BOS targets, so SFT
  never learned to stop while the loss looked fine. Fixed in the image. Check P(EOS) after an answer on any new pipeline.
- **YaRN.** Olmo 3 applies YaRN to full-attention layers only. Unpatched MaxText also applies it to sliding-window
  layers, which costs +0.1 nats/token at the start. Fixed in the image.
- **Vocab padding.** 100,278 isn't divisible by tensor parallelism. Zero-padding to 100,352 is wrong (+3.2 nats/token).
  Replicate the vocab axis instead: `logical_axis_rules=[['vocab',[]],['activation_vocab',[]]]`.
- **Olmo 3 tokenizer.** `allenai/olmo-3-tokenizer-instruct-dev` loads as `GPT2Tokenizer` under transformers 5 and drops
  the pre-tokenizer. Train with a `PreTrainedTokenizerFast` save.
- **KDA miscompile on v4.** Slicing a reshape inside the sub-block inverse dropped a column offset on TPU only (CPU was
  correct). Fixed in maxtext 9f55e7b6 (in the image). Kernel rewrites need a TPU-vs-exact-scan check, not only a CPU one.
- **MoE buffer.** Fully dropless runs out of HBM at 64K. `ragged_buffer_factor=2.5` drops nothing as long as
  `REQUIRED_RBF` stays below it.
- **Restore OOM.** r2's image replicates the expert Adam moments on restore and runs out of HBM. Use the
  restore-fix image.
- **Data order depends on slice shape.** Batches are packed per data host, so a different host count gives a
  different, still valid, data order. Keep every arm of a comparison, and its control, on the same slice shape.
- **Zero steps, exit 0.** `train_sft` exits 0 with no steps when the data iterator fails. The launcher's step guard
  catches it; keep that guard.
- **KDA all-gathers on a new mesh.** Under `olmo35-cp-ep` on the `8x8x8`, the KDA chunk scan's inputs and states were
  sharded on the chunk axis, so XLA all-gathered them inside every scan iteration: 97.6% of a 475 s step. Logical
  constraints in `_delta_rule_chunked_subblock` (abhishekraok/maxtext c5f1251d) bring the step to 3.33 s. CPU runs never
  show in-loop gathers, so profile one TPU step with xprof before trusting a new mesh.
- **Compile-time HBM.** At 64 rows on the `8x8x8`, `ragged_buffer_factor=4.0` failed to compile (HLO temporaries
  32.17 GB vs 31.75 GB of HBM); 3.0 compiles, and the runs peaked at 2.39-2.84.
- **GCS egress quota.** Two big slices loading the base checkpoint at once hit `google_egress_bandwidth` (us-central2)
  and failed at startup. Stagger big-slice launches.
- **The control host is a single point of failure.** Watchers and log tailers die with the host they run on (a Beaker
  session lost its node on 2026-10-16). Keep any record a rule depends on in a source that survives it.
- **hostNetwork.** The image's libtpu forms a multi-host slice only on the host network. Keep `hostNetwork: true` and
  `dnsPolicy: ClusterFirstWithHostNet`.
- **`pgrep -f` / `pkill -f` inside `bash -c`** match their own command line: loops never end and kills hit the caller.
  Use PIDs.

## Changing the recipe

- **Other slice sizes and meshes.** `olmo35-ep-as-cp` puts one 64K row on 8 chips (KDA's 8 heads). Measured on Olmo 3.5
  small with 65,536-token rows:

  | slice | mesh | rows/step | s/step | tokens/s/chip | status |
  |---|---|---|---|---|---|
  | `4x4x8` | `olmo35-ep-as-cp`, fsdp 16 x expert 8 | 16 | 3.94 | 2,080 | r2 and A0'; the recipe |
  | `8x8x8` | `olmo35-ep-as-cp`, fsdp 64 x expert 8 | 64 | 4.28 | 1,910 | H078's 4x batch; `ragged_buffer_factor=3.0`; stalls, see [Cost](#cost-use-the-4x4x8) |
  | `8x8x8` | `olmo35-cp-ep` + c5f1251d | 16 | 3.33 | 620 | runs, but numerics differ from the `4x4x8` (#1933) |
  | `8x8x16` | `olmo35-cp-ep`, fsdp 64 x context 2 x expert 8 | 64 | - | - | never reached step 1 (egress quota, then a TPU core halt) |

  Each mesh was picked to keep r2's per-device layout, not tuned. No sweep of the parallelism split, remat policy or
  per-device batch has been run, so these are known-good points, not optima. Any other slice or mesh needs its own
  same-slice control.
- **Comparing against a GPU result.** TPU and GPU runs of the same recipe differ in data order and numerics. Compare
  TPU arms with a TPU control on the same slice shape.
