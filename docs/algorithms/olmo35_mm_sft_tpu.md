# Olmo 3.5 multimodal SFT on TPU

The multimodal SFT recipe of Olmo 3.5 tiny (B2e base, OLMo-core's `olmo35_mm_sft`) in MaxText on **one TPU v4
`4x4x8` slice (128 chips)** of `gke-tpu-v4`. What the recipe is, its data and its GPU path are in
[Olmo 3.5 multimodal SFT](olmo35_mm_sft.md). This page covers only the TPU path. It assumes the cluster basics of
[Running SFT on TPU](tpu_sft_guide.md): access, the `post-training` queue, stopping a run through Kueue,
`watch_jobset.sh`, and `LOSS-PARTS`.

Measured on the reference run (ledger H271: H231's GPU arm `4n-emo256k` repeated on TPU, data seed 34521):

| | |
|---|---|
| Steps | 10,962 of 32 × 65,536 tokens, all completed, 0 restarts |
| Speed | ~10.3 s/step (31.7 h, ~4,060 chip-hours). The E4 flags measure 9.35 s/step (about 28.5 h); see Speed |
| Training CE vs the GPU run | within ±0.002 in every 200-step window |
| Step-1 CE | TPU 1.0915, GPU 1.073. Not a model difference: see Gates |
| MoE buffer | `REQUIRED_RBF` ≤ 2.46 after the first step, under the 3.0 buffer. The first step read 4.42 (see Traps) |
| Evals | the GPU run's own pipeline and cells, read against H231's two GPU seeds (Results) |

Parity verdict (H271, descriptive and post hoc): the TPU run matches the GPU run within the GPU seed spread.
At s10962, 47 tasks have scores from both GPU seeds and the TPU. |TPU − mean of the two GPU seeds| is at most the
|seed Δ| on 33 of 47 tasks. The mean |TPU − Ḡ| is 3.31, against a mean |seed Δ| of 6.83. The one flagged task is
`bfcl_parallel`, measured on exports with the stop-token bug (Results).

## Files

All under `scripts/tpu/maxtext/`:

| Path | What |
|---|---|
| `k8s/olmo35-mm-sft-4x4x8/` | The run: JobSet, launcher `train_mm_sft.sh` (MaxText's `pre_train` trainer, not `train_sft`), and the settings of the reference run. `kustomization.yaml` explains each setting. |
| `k8s/olmo35-mm-sft-4x4x8-gate/` | The pre-run gate: 20 steps with the EMo pool fixed at 512 and batch hashes on, and no checkpoints. |
| `image-olmo35-mm/` | The image: MaxText from jason718/maxtext at a pinned commit, plus CPU torch and OLMo-core `fd9bba1ba`. The OLMo-core data loader runs in-process. |
| `olmo35-mm/convert_b2e.sh` | OLMo-core DCP → MaxText start checkpoint, with `<think>` rows seeded, then a bitwise check. |
| `olmo35-mm/export_to_olmocore.sh` | A TPU checkpoint → OLMo-core DCP for the GPU export and evals, with a bitwise round-trip check. |
| `olmo35-mm/check_gates.py` | Reads the gate run's logs against the GPU reference. |

## How it matches the GPU run

- **Same data, same order.** MaxText's `dataset_type=olmo_mm` builds OLMo-core's `MixtureDataLoader` and
  `MultimodalCollator` from the GPU run's own `config.json`, inside every TPU host process. With
  `OLMO_MM_LOADER_WORLD=32`, each of the 32 trained rows of a step is the batch of one of the GPU run's 32 ranks. The
  loader's order depends only on its seed, rank and world size, so each step trains the GPU run's rows in the GPU run's
  order. The loader keeps no state in the checkpoint, so a resume re-reads every earlier batch: 0.1–1.4 s per batch
  per rank.
- **Same weights.** The start checkpoint is B2e step 22240, converted from its fp32 masters. Rows 100256 (`<think>`)
  and 100266 (`</think>`) are seeded at conversion, as OLMo-core seeds them before step 1. The trainer must not seed
  again (`olmo_seed_token_rows=[]`).
- **Same recipe.** `configs/olmo/olmo35_tiny_mm_sft_64k.yml` in the image sets:
  - LR 2e-4 (LM), 4e-4 (connector) and 2e-5 (ViT); 3% warmup, then linear to 0; wd 0.
  - Clipping at 1.0 per scheduler group; nothing frozen.
  - EMo causal pools uniform in [16, 512]; router aux losses divided by loss tokens.
  - Mesh `olmo35-ep-as-cp`: fsdp 16 × expert 8, per-device batch 0.125, gradient accumulation 2.

## Run it

1. **Image.** `image-olmo35-mm/build.sh` builds `run` by default (`217c8458…-olmocore-fd9bba1ba`, already
   published, so the script only prints it). Use `e4` for the faster flags and `gates` for the gate.
2. **Start checkpoint.** The reference run's start checkpoint is
   `gs://ai2-olmo-dev/post-training/checkpoints/olmo35-mm-b2e-emocausal-step22240-thinkseed/0/items`; reuse it for
   this base. For another OLMo-core base, convert it on a CPU box with a few hundred GB of RAM: the conversion holds several fp32
   copies of the 12.9B values, ~52 GB each. `PY` needs MaxText at the
   pin, OLMo-core `fd9bba1ba`, torch, `jax[cpu]` and tensorstore; the training image has all of them.

   ```bash
   PY=... scripts/tpu/maxtext/olmo35-mm/convert_b2e.sh <OLMo-core step dir> <new local dir>
   gcloud storage cp -r <new local dir> gs://ai2-olmo-dev/post-training/checkpoints/<new name>
   ```

   The script checks the converter's own round trip (0.0) and then a separate bitwise check. Both must pass: the
   reference conversion was bitwise equal on all 829 tensors (12.92B values).
3. **Data.** The pods mount only the folders of the recipe's inputs from `gs://midtraining-data/weka/<path without
   /weka/>`, each at its `/weka/...` path, so the paths in the OLMo-core config resolve unchanged:
   - the C12 numpy cache (147 GB);
   - the h182 replay (13 GB);
   - the stage2_v1 image corpora under `mm-olmo` (~4.6 M objects, ~0.9 TB);
   - the tokenizer snapshot in the c11 HF cache.

   A new data config needs its folders mirrored first. Check each mirrored folder's object count and bytes against
   Weka. Do not rely on the rsync exit code: the first pass missed files in three folders.
4. **Gate** (next section). Then copy `k8s/olmo35-mm-sft-4x4x8/`, give the JobSet and the ConfigMaps new names, set
   `RUN_NAME`, and apply it with `kubectl apply -k`. The reference gate logged step 1 about 17 minutes after launch.
   That time includes each host building its loader's packing index (~2.5 min).
5. **Read the loss** from `LOSS-PARTS ce=` only, as for text SFT. Compare it with the GPU run's `train/CE loss` in
   200-step windows.
6. **Evaluate.** `olmo35-mm/export_to_olmocore.sh <…/checkpoints/S/items> <dcp dir>/step<S+1> <OLMo-core config.json>`
   writes an OLMo-core DCP whose round trip is checked bitwise. Then export and evaluate it exactly as a GPU checkpoint
   ([Olmo 3.5 multimodal SFT](olmo35_mm_sft.md)). MaxText's checkpoint dir `S` holds the state after `S+1` updates:
   dir 5500 is step 5501, and the final dir 10961 is step 10962. The run keeps two checkpoints (each ~145 GB with
   optimizer state), so `checkpoints/5500` is deleted when 6500 saves. Convert a mid-run checkpoint within ~1,000
   steps of its save.

## Gates (run before every new TPU run)

These two gates are agreed with the owner for every future TPU multimodal SFT run (H271's brief). Both compare against
one OLMo-core run of the same recipe with the same 32 ranks, launched with
`--recipe.batch_hash_dir=<dir> --recipe.step1_reference=true` and stopped after step 20. It writes `rank{R}.jsonl`
(OLMo-core `BatchHashCallback`) and `step1_eval.json`, the eval-mode CE of step 1's batch. In eval mode every
document routes over the eval pool of 512, which is every expert.

1. **Step-1 CE within 1e-3, with the EMo pool fixed at 512 on both sides.** EMo causal routing samples each
   document's expert pool in [16, 512] from the trainer's RNG, and OLMo-core and MaxText use different generators.
   That is why the reference run's step 1 read 1.0915 on TPU against 1.073 on GPU. With the pool fixed, a CPU fp32
   forward matched OLMo-core to 3.7e-5 in CE (one image plus a chat, 638 tokens). The gate overlay sets
   `emo_min_document_expert_pool=emo_max_document_expert_pool=512`.
2. **Batch hashes for steps 1–20.** The gate image's MaxText (`olmo_mm_batch_hash_steps=20`) logs one
   `MM-BATCH-HASH {"step", "rank", "hashes", "index"}` line per loader rank and step. It hashes the raw OLMo-core
   collator batch with OLMo-core's own definition: sha256 of `input_ids`, `labels`, `loss_masks`, `example_ids`,
   `document_ids`, `position_ids`, `image_crop_counts` and `pooled_patches_idx`, plus `all`. Each record therefore
   compares one to one with the GPU's `rank{R}.jsonl`. The reference run predates this logging, so it has no hash
   receipt. Its batch order matched by construction only.

```bash
kubectl --context gke_ai2-olmo_us-central2-b_gke-tpu-v4 apply -k <your copy of k8s/olmo35-mm-sft-4x4x8-gate>
scripts/tpu/maxtext/k8s/watch_jobset.sh olmo35-mm-sft-gate <RUN_NAME> 120
python scripts/tpu/maxtext/olmo35-mm/check_gates.py --tpu-logs <saved pod logs> \
  --gpu-hashes <batch_hash_dir> --gpu-step1 <batch_hash_dir>/step1_eval.json
```

`check_gates.py` prints `GATES-PASS` only if both gates pass. It also fails the CE gate when the TPU's first step
dropped routed MoE tokens: a step that drops tokens cannot be compared with OLMo-core's dropless forward (see Traps).

## Speed

All rows are 130-step checks on the `4x4x8` in the same data order. The table is from allenai/olmo-gcp-infra#74.

| image, flags | s/step, image steps 50–129 (median) | steps 2–129 | CE vs the reference run | est. HBM |
|---|---:|---:|---|---:|
| `217c8458`, the reference run's flags | 10.73 | 10.41 | reference | |
| **E4**: `ed83783e` + the flags in the overlay's comment | **9.35** | 9.17 | steps 1–40 up to 3.3e-3 apart (reduction order), 41–129 up to 3.8e-4 | 20.8 GB |
| E2: `ed83783e` + 4 flags | 9.85 | 9.71 | steps 1–40 equal | 26.6 GB |

The first full attempt ran at 16.9 s/step, before balanced ViT crops and loader prefetch. Use E4 for new runs. Use
E2 for bitwise continuity with an existing run's early steps.

## Results (H271)

Each TPU checkpoint was converted to OLMo-core and checked bitwise. It then went through the GPU run's HF export and
olmo-eval cells. TPU s5501 is compared with GPU s5481, and TPU s10962 with GPU s10962. At 500 items, the binomial SE of
a difference is about 2–3 points.

- **Vision.** All 13 tasks are within 2 SE at both reads; at s10962 the mean difference is −0.6. Against the GPU seed
  pair, none is flagged at either read. Worth watching: InfoVQA is −4.8 and −4.4 at the two reads (about 1.5 SE each,
  same sign).
- **Text, s10962.** IFEval −1.5, GPQA diamond/main −3.0/−1.3, RULER 64K/128K +2.4/+1.3. None is flagged against the
  seed pair.
- **BFCL.** Every multimodal export of that time listed only eos 100257 in `generation_config.json`, so vLLM did not
  stop at `<|im_end|>`. With both exports fixed to `[100265, 100257]`, BFCL is 57.8 on TPU and 58.9 on GPU. The BFCL
  seed Δ of the two GPU seeds is −14.9.

## Traps

- **The first step can drop MoE tokens.** The reference run's first step logged `REQUIRED_RBF step=0 max=4.42`
  against `ragged_buffer_factor=3.0`, so it dropped routed tokens. Every later step stayed ≤ 2.46. `REQUIRED_RBF` and
  `MOE-DROPPED` count steps from 0, and `LOSS-PARTS` counts from 1. The gate logs `MOE-DROPPED`. If step 1 drops, raise
  the gate's buffer and check that it still fits in HBM.
- `base.yml` defaults to `freeze_vision_encoder_params: true`. The mm-SFT config overrides it.
- B2e needs `emo_enabled=true emo_causal_pool=true` with the pool uniform in [16, 512]
  (`emo_document_expert_pool_choices=[]`). The 256K midtrain config's [32, 64, 256] choices are different.
- Never set `--xla_tpu_enable_async_ragged_all_to_all`; the launcher refuses it.
- Keep `OLMO_ORBAX_BARRIER_TIMEOUT=3600`. Hosts that build the loader reach Orbax's checkpoint-load barrier minutes
  late.
- The launcher sets `ulimit -c 0`. A core dump (~75 GB) would get the pod evicted for ephemeral storage, which hides
  the real error.
- The image needs torchvision. Without it, OLMo-core's loader skips every image example as a data error and trains
  text only. MaxText's adapter raises at startup instead.
- An Orbax restore of a training checkpoint on a CPU box crawls and gets OOM-killed. The conversion scripts read
  params with tensorstore instead. Run one such job per box at a time.
- Check `generation_config.json` of every multimodal `hf_text` export for `eos_token_id: [100265, 100257]` before any
  tool-call eval.

## Code

- MaxText (jason718/maxtext):
  - `jasonr-olmo35-mmsft` 2b8446d3: olmo35-sft, cross-CP EMo, `OLMO_MM_LOADER_WORLD`, the mm-SFT config and the
    adapter fixes.
  - `-perf` 217c8458: the reference run.
  - `-perf2` ed83783e: E4.
  - `-gates` e3543ef2: batch hashes. This branch is not pushed yet.
- OLMo-core: allenai/OLMo-core@fd9bba1ba3f12721ae1ff3493f0b266366debec6 (#942).
- Original overlays and image build: allenai/olmo-gcp-infra#74.
- The ledger record and eval pipeline: allenai/olmo-post-training-ledger#1586 (H271,
  `experiments/mmsft/tpu/`).
