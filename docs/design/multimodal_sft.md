# Design: Multimodal SFT (Molmo stage 2) in open-instruct

**Status:** V1 implemented on the `vision` branch (foundation #1834, training path #1856, parity below)
**Branch:** `vision` (open-instruct), depends on `vision` (OLMo-core)
**Scope:** V1 = stage-2 training parity with OLMo-core's `src/scripts/train/Molmo2-Stage2.py`; HF export and in-loop eval deferred.

## 1. Motivation and goal

Molmo training has two stages: stage 1 (caption pre-training of the vision connector on PixMo-Cap) and stage 2 (multimodal multitask SFT over a mixture of demo, academic VQA, pointing, and text-only NLP data). Today, stage 2 lives in mm_olmo / the OLMo-core `vision` branch, while Olmo text post-training SFT lives in open-instruct. This design merges them: **stage-2 multimodal SFT runs inside open-instruct as a single stage whose `nlp` mixture group is the real Olmo post-training SFT mix**, produced by open-instruct's own dataset tooling.

Guiding constraints:

- **Thin wrapper.** Everything heavy is imported from the OLMo-core `vision` branch. open-instruct contributes argument dataclasses, mixture assembly, and launch scripts.
- **OLMo-core native, end to end.** No DeepSpeed; FSDP/HSDP via OLMo-core. The trainer, train module, checkpointer, and callbacks are all OLMo-core's own — open-instruct adds a callback only where OLMo-core has no equivalent.
- **A generic mixture layer.** Multimodal is one *source type*, not the shape of the system. Mixture sources plug in through a small protocol + registry, so future SFT data (more text corpora, new modalities, tool trajectories) is one adapter each, with no entry-point changes.
- **Both backbones.** Qwen3 (the existing Molmo2-4B/8B presets) works on day one; Olmo 3 enablement is a parallel OLMo-core workstream specified in §7.

## 2. What already exists (and is reused unmodified)

OLMo-core `main` (already pinned by open-instruct) contains `olmo_core.nn.vision`: `MultimodalLM` / `MultimodalLMConfig` with `molmo2_4B()` / `molmo2_8B()` presets — SigLIP2-SO400M/14-378 ViT (truncated to 25 blocks, features from layers (24, 18) concatenated), 2×2 attention-mean-query pooling with padding mask, SwiGLU MLP projector, Qwen3 LM, and `SplitVocabEmbedding` (base vocab 151936 + 128 inputs-only extra tokens).

The OLMo-core `vision` branch (HEAD `cb17582` at time of writing) adds everything else stage 2 needs — all of it a faithful port of mm_olmo's `train_multitask_model.py` machinery:

| Piece | Location (OLMo-core) |
|---|---|
| Multimodal train module: FSDP wrap of lm + vision + connector, `freeze_params` fnmatch, per-group LR via `OptimGroupOverride` + `PerGroupScheduler`, weighted CE with global-batch divisor, `response_logits_only`, vision/connector compile + activation checkpointing | `src/olmo_core/train/train_module/transformer/multimodal_train_module.py` |
| Data package: `MultimodalCollator`, `MultimodalDataLoader`, `MixtureDataLoader` (weighted multinomial multi-source), 2D token+crop knapsack packing, threaded prefetch | `src/olmo_core/data/multimodal/` |
| Image preprocessing (multi-crop tiling, max_crops=8, 378px/patch14, 2×2 pooling) | `src/olmo_core/nn/vision/molmo2_image_processor.py` |
| Datasets: PixMo (cap/AMA/points/clocks/…), ~30 academic VQA sets, multi-image, text-only Tulu4 | `src/olmo_core/data/multimodal/{pixmo_*,academic*,multi_image_datasets,tulu}.py` |
| The stage-2 mixture (demo 0.25 / academic 0.418 / pointing 0.166 / nlp 0.166) + validation ladder | `src/olmo_core/data/multimodal/mixtures/image_only_v9.py` |
| HF interop: load `allenai/Molmo2-4B/8B`, export back to HF | `src/olmo_core/nn/vision/molmo2_loader.py` |
| Reference training scripts | `src/scripts/train/Molmo2-Stage1.py`, `Molmo2-Stage2.py` |

open-instruct has zero multimodal code today. Its text SFT path (pre-tokenized `.npy` memmaps + `NumpyPackedFSLDataset`) is not reusable for images, but two precedents shape this design: `HFDataLoader(DataLoaderBase)` in `open_instruct/data_loader.py` (HF-dataset-backed OLMo-core data loader) and `DPOTrainModule` in `open_instruct/olmo_core_train_modules.py` (train-module subclassing — which, notably, we do **not** need here).

### 2.1 The validated Olmo 3 text SFT baseline

`scripts/train/debug/oc_sft_olmo3_7b_1node.sh` (Abhishek Rao, #1811/#1819) is the **verified end-to-end Olmo-3-7B text SFT pipeline** — one epoch over Dolci-Instruct-SFT, 1723 steps on 1×8 H100, ~9.5 h wall / ~77 GPU-hours (includes a restart from the step-400 checkpoint), CE 1.195 → 0.639. Beaker `01KZHT32T30M2VHCWKRJS1G9P7`; full analysis in the wandb report "Olmo 3 7B SFT — pipeline validation and eval results".

Eval movement (identical prompts and tokenizer, so the delta isolates the weights): **IFBench 18.3 → 25.0 (+6.7, +36% relative)** and **GSM8K 74.3 → 76.1 (+1.8, no regression)** at the final checkpoint; all five checkpoints beat base on IFBench. For reference, the released Olmo-3-7B-Instruct-SFT scores 27.4 (2.4 above this run — within noise, different mixture, single epoch) and the released Instruct (SFT→DPO→RLVR) 32.3. This was an infra check, not a recipe reproduction, but it establishes: the pipeline trains stably and metrics move.

It defines the text half of the merged stage:

- **Data:** `--mixer_list allenai/Dolci-Instruct-SFT 1.0` — the Olmo 3 post-training SFT mixture the bridge should default to for Olmo 3 backbones (not the older tulu mixes).
- **Tokenizer/template:** `allenai/olmo-3-tokenizer-instruct-dev` with `--chat_template_name olmo123` — deliberately unregistered so it **falls through to the tokenizer's own built-in template**, which is what the released Olmo 3 Instruct models used (fallback tracked in #1805). `add_bos` stays off (asserted for `olmo*` templates).
- **Cache-key discipline:** tokenizer config, mixer, transform fns, `max_seq_length`, and seed are hashed into the numpy cache key; the tokenize (CPU) and train jobs must pass byte-identical values. The bridge inherits the same rule (§5.5).
- **Config lore this design inherits:** `--ephemeral_save_interval -1 --keep_last_n_checkpoints -1` (olmo-core deleting a ~100 GB checkpoint tree on weka overruns a soft timeout and killed a healthy run at step 401 — fixed by #1810); `selected_modules` AC rather than budget mode; text runs use `attn_implementation flash_2`.
- **YaRN RoPE is load-bearing, not a length knob:** three earlier runs collapsed to CE ≈ 8 at 8e-5, 2e-5, and 5e-6 alike. The cause was a missing `--rope_scaling_factor 8` — Olmo-3-1025-7B's weights are YaRN-trained, and without the flag the 8 full-attention layers run plain RoPE against YaRN-trained weights. With the flag, 8e-5 trained stably from cold for all 1723 steps. Consequence for this design: **any config that loads Olmo-3-1025-7B-lineage weights must apply the matching YaRN scaling, regardless of training sequence length** (§7).
- **Context parallelism is off the table for this lineage:** `--cp_strategy ulysses` is incompatible with `--rope_scaling_factor` (ulysses splits across attention heads and never shards RoPE buffers along the sequence, which YaRN requires), and ring strategies need `ring-flash-attn`, absent from the image. The multimodal train module doesn't support CP anyway, so the designs agree.
- **Template sizing:** `olmo123` resolves to the tokenizer's own chat template, which prepends a function-calling system block to every conversation lacking a system message — ~840 tokens/row rather than the ~643 measured under the `olmo_thinker` templates (the epoch was 1.81 B tokens, ~30% longer than a 643-based estimate). The bridge inherits this cost per text example (§5.5).
- Related fixes already on main: #1809 (olmo-core → HF converter, previously broken twice over), #1812 (olmo-core runs record their Beaker experiment in wandb).

## 3. Architecture

```
open_instruct/olmo_core_mixture_finetune.py   (entry point)
        │
        ├── MultimodalLMConfig.build()  +  molmo2_loader (HF init)      [OLMo-core]
        ├── MultimodalTransformerTrainModuleConfig.build(model)        [OLMo-core, unmodified]
        ├── MixtureDataLoader ◄── sft_mixture.build_mixture() ◄── SOURCE_REGISTRY
        │        + MultimodalCollator        ├── "molmo": image_only_v9 sources  [OLMo-core]
        │                                    ├── "open_instruct_sft": text adapter [open-instruct]
        │                                    └── future source types: one factory each
        └── TrainerConfig + OLMo-core native callbacks (config_saver, gc, gpu_monitor, wandb, checkpointer, beaker)
```

**No train-module subclass, and no callback wrappers.** `DPOTrainModule` exists because DPO needed a different loss. Stage-2 SFT needs nothing `MultimodalTransformerTrainModuleConfig` doesn't already expose (per-group LR/scheduler, freezing, `response_logits_only`, weighted CE, compile/AC flags). Trainer-side, the run mirrors `Molmo2-Stage2.py`'s OLMo-core-native setup: `TrainerConfig` + `CheckpointerConfig` and OLMo-core's own callbacks (`ConfigSaverCallback`, `GarbageCollectorCallback`, `GPUMemoryMonitorCallback`, `WandBCallback`, and the checkpointer). One exception, found in the first smoke run: OLMo-core's `beaker` callback requires `beaker-gantry` (its attach path imports `olmo_core.launch.beaker`, whose module top imports gantry), which open-instruct's image does not ship — so the Beaker-description role uses open-instruct's beaker-py-2.x `BeakerCallbackV2` instead. `PerfCallback` is **not** used by default — add them only if the team wants open-instruct's Beaker-description/MFU conventions on these runs, as an additive opt-in. If a future need appears (e.g. logging per-group LRs), prefer an OLMo-core callback over any subclass.

### 3.1 New files

| File | Purpose |
|---|---|
| `open_instruct/olmo_core_multimodal_utils.py` | Dataclasses + builder functions. **All** `olmo_core.nn.vision` / `olmo_core.data.multimodal` imports live here and in the entry point, inside functions (lazy) — both to keep the text SFT/DPO paths importable if the pin ever moves, and because `data/multimodal/paths.py` freezes `MOLMO_DATA_DIR` at import time. |
| `open_instruct/olmo_core_mixture_finetune.py` | Entry point, mirrors `olmo_core_finetune.py`'s structure. |
| `open_instruct/sft_mixture.py` | The generic mixture layer: `MixtureSource` protocol, `SourceSpec`, `SOURCE_REGISTRY`, `build_mixture` (§4). Deliberately not multimodal-named — any SFT source type registers here. |
| `open_instruct/sft_text_dataset.py` | The first source adapter: `OpenInstructTextDataset` + `OpenInstructTextDatasetConfig` (§5). |
| `scripts/train/debug/mm_sft.sh` | 1-GPU Beaker smoke: `--mixture debug`, 10 steps, compile off. |
| `scripts/train/vision/molmo2_stage2.sh` | Production 8-GPU stage-2 parity run. |
| `open_instruct/test_olmo_core_multimodal_utils.py`, `open_instruct/test_olmo_core_mixture_finetune_gpu.py`, `open_instruct/test_sft_mixture.py`, `open_instruct/test_sft_text_dataset.py` | Tests (§8). |

### 3.2 Configuration

Reused from `olmo_core_utils.py`: `ExperimentConfig`, `LoggingConfig`, `CheckpointConfig`, `setup_distributed_env`, `is_hf_checkpoint`. **Not** reused: `ModelConfig` / `TrainingConfig` (rope-scaling, context-parallel, and budget-AC fields are text-path traps), and `build_base_callbacks` / `build_checkpointer_callback` (this path uses OLMo-core's callbacks directly — §3.3 step 7). Three new dataclasses, aggregated into `MultimodalSFTArguments` and parsed with `utils.ArgumentParserPlus`:

- **`MultimodalModelConfig`** — `base_hf_model_id` (default `allenai/Molmo2-4B`; source of the HF config → `MultimodalLMConfig` via `molmo2_config_from_hf_config`, and of the tokenizer), `model_name_or_path` (None ⇒ init weights from `base_hf_model_id` via the molmo2 loader; an OLMo-core checkpoint dir ⇒ trainer `load_path` init from a stage-1 run), `model_preset` (a `MultimodalLMConfig` classmethod name — the hook through which the future `molmo3_7B` preset arrives; open-instruct just resolves `getattr`), `residual_dropout=0.1`, `tokenizer_name_or_path`, `trust_remote_code=True`.
- **`MultimodalTrainingConfig`** — Stage2-parity defaults: `max_seq_length=16384`, `global_batch_instances=128`, `rank_microbatch_instances=2`, `learning_rate=1e-5` (LM) / `connector_lr=5e-6` / `vision_lr=5e-6`, `warmup_steps=200`, cosine with `alpha_f=0.1`, `weight_decay=0.0`, betas (0.9, 0.95), `max_grad_norm=1.0`, `z_loss_multiplier=1e-4`, `response_logits_only=True`, `freeze_params` glob list, compile flags (model/vision/connector), vision/connector AC, `ac_block_interval=2`, `dp_shard_degree` (None ⇒ FSDP single-node, HSDP with shard=gpus-per-node multi-node, mirroring `olmo_core_finetune.py`).
- **`MixtureConfig`** (generic — lives in `sft_mixture.py`, not multimodal-named) — `mixture` (a named preset: the keys of `image_only_v9.VALIDATION_MIXTURES` — `debug`, `demo`, `pointing`, `academic`, `multi-image`, …, `image-only-v9` — each expanding to a list of `SourceSpec`s), `sources` (a JSON list of `SourceSpec` entries that replace or extend preset groups: `{"group": "nlp", "rate": 0.166, "type": "open_instruct_sft", "args": {...}}` — the general mechanism), convenience shorthands `nlp_source: "tulu4" | "open_instruct"` and `nlp_rate` (sugar for the common override), `mixer_list`/`mixer_list_splits` (args for the `open_instruct_sft` adapter), packing knobs (`pack_sequences=True`, `pack_max_crops=125`, `est_tokens_per_example=1500`, `prefetch_workers=4`), `max_crops=8`, `p_high_res`. `MOLMO_DATA_DIR` is deliberately **not** a CLI field: `olmo_core.data.multimodal.paths` freezes it at import time and repo convention keeps imports at the top of the file, so it is a launch-environment variable (`mason.py --env MOLMO_DATA_DIR=...`).

Not included: `dataset_transformation.TokenizerConfig` at the top level — the run's tokenizer is the HF Molmo2/Qwen3 tokenizer (`trust_remote_code`), and image-data tokenization goes through the vision branch's `qwen3_layout`. The bridge owns its own `TokenizerConfig` internally.

### 3.3 Entry-point flow

1. Fail fast if the `MOLMO_DATA_DIR` root (a launch env var, §3.2) is not a directory; `setup_distributed_env(seed)`.
2. `AutoTokenizer.from_pretrained(tokenizer_name_or_path or base_hf_model_id, trust_remote_code=True)`.
3. `setup_multimodal_model(cfg)` — weights are loaded **before** the train module wraps the model, following Stage2's order (there is no `reload_hf_checkpoint_after_parallelization` analogue on this path):
   - preset or HF-derived `MultimodalLMConfig`, `config.lm.block.dropout = residual_dropout`, `build(init_device="meta")`;
   - HF init: `AutoModelForImageTextToText` → `reinit_rope_buffers` → `molmo2_hf_state_dict_to_multimodal_lm` → `load_state_dict(strict=False)` → `retie_word_embeddings`;
   - stage-1 init: `to_empty(device)` + `retie_word_embeddings` (required pre-FSDP), loading deferred to the trainer via `load_path`.
4. `build_multimodal_train_module_config(...)` — a direct transcription of Stage2's optimizer/scheduler/parallelism block (AdamW with `OptimGroupOverride` for `connector.*` and `vision.*`, `PerGroupScheduler`, `rank_microbatch_size = instances × seq_len`, bf16 FSDP/HSDP with fp32 reduce, selected-blocks AC) — then `.build(model)`.
5. `MultimodalCollatorConfig(pad_token_id=<from tokenizer, assert non-None>, label_ignore_index=-100, pad_sequence_length=max_seq_length).build()`.
6. Build the bridge dataset if `nlp_source == "open_instruct"`; `build_mixture(...)` (§4); `MixtureDataLoader` with `global_batch_size = global_batch_instances × max_seq_length`, packing knobs, seed.
7. Callbacks: OLMo-core's own set, mirroring `Molmo2-Stage2.py` — `ConfigSaverCallback`, `GarbageCollectorCallback`, `GPUMemoryMonitorCallback`, `WandBCallback`, and the checkpointer callback — plus open-instruct's `BeakerCallbackV2` for Beaker description updates (OLMo-core's `beaker` callback needs `beaker-gantry`, absent from the image; §3). `PerfCallback` remains an opt-in addition. Checkpointer with `save_async=False` for V1 (Stage2 does the same; async save with the multimodal module is unproven). Checkpoint-retention defaults follow the validated text recipe's lore (§2.1): production scripts pass `--ephemeral_save_interval -1 --keep_last_n_checkpoints -1` — olmo-core deleting a ~100 GB checkpoint tree on weka overruns a 30 s timeout and kills the job.
8. `TrainerConfig(save_folder=output_dir, max_duration=Duration.steps(max_train_steps), ...)`. Init/resume cases:
   - **HF init**: weights already loaded; `load_strategy=if_available` so preempted runs resume from `save_folder`.
   - **Stage-1 init**: `load_path=model_name_or_path`, `load_trainer_state=False`, `load_optim_state=False`. The trainer prefers a checkpoint in `save_folder` over `load_path` when one exists, giving preemption resume for free (verify on the pinned SHA — risk §9).
9. `trainer.fit()`; teardown.

## 4. Mixture assembly and the nlp override

`build_mixture(tokenizer, mixture_cfg, training_cfg, seed, text_dataset=None)`:

The mixture layer is generic (`sft_mixture.py`); multimodal is one source type among several:

- **`MixtureSource` protocol** — what every source must satisfy: map-style `__len__` / `__getitem__(i) → dict[str, np.ndarray]` in the example schema of §5.3, index-deterministic. This is exactly what `MixtureDataLoader` consumes, so the protocol is a documentation of the OLMo-core contract, not a new abstraction layer.
- **`SourceSpec`** — a declarative entry: `{group, rate, type, name, args}`.
- **`SOURCE_REGISTRY: dict[str, SourceFactory]`** — `type` → factory. V1 registers two: `"molmo"` (resolves any `image_only_v9` dataset name through the vision branch's builders) and `"open_instruct_sft"` (the text adapter, §5). A future data type — another text corpus, a new modality, tool trajectories — is one new factory registration; the entry point and loader wiring don't change.

`build_mixture(tokenizer, mixture_cfg, training_cfg, seed)` then:

1. Expand the named preset (`--mixture`) into `SourceSpec`s — `image-only-v9` expands to `IMAGE_ONLY_V9_SUBMIXTURES` (demo 0.25 / image_academic 0.418 / image_pointing 0.166 / nlp 0.166) with all sources typed `"molmo"`.
2. Apply `sources` overrides (and the `nlp_source`/`nlp_rate` shorthands, which desugar to a `SourceSpec` replacing the nlp group with an `"open_instruct_sft"` entry).
3. **Prune groups to the resolved sources before computing lengths** — `compute_flat_mixture_weights` only reads lengths for sources present in the groups, and dataset lengths are what force lazy weka datasets to build. This keeps `--mixture debug` cheap.
4. Instantiate each spec through `SOURCE_REGISTRY`, `compute_flat_mixture_weights(groups, lengths)` → flat `(name, weight)` list; hand `datasets`, `weights`, `dataset_names` to `MixtureDataLoader`.

This requires **zero OLMo-core changes**: `Molmo2-Stage2.py`'s `_append_extra_sft_sources` already proves external sources can be spliced into the mixture from the launch script layer.

CLI examples:

```bash
# smoke: weka tulu dump + two academic sets
--mixture debug

# the merged single stage: full image mixture, nlp group = open-instruct post-training mix
--mixture image-only-v9 --nlp_source open_instruct \
  --mixer_list allenai/tulu-3-sft-olmo-2-mixture 1.0 --nlp_rate 0.166

# Olmo 3 backbone (once §7 lands): text half = the validated Dolci recipe (§2.1)
--model_preset molmo3_7B --tokenizer_name_or_path allenai/olmo-3-tokenizer-instruct-dev \
  --mixture image-only-v9 --nlp_source open_instruct \
  --mixer_list allenai/Dolci-Instruct-SFT 1.0 --nlp_rate 0.166
```

Deferred: inline JSON mixture specs; Stage2's `mmfinereason_rate`/`finevision_rate` extras (default 0 upstream).

## 5. The first source adapter: `OpenInstructTextDataset`

The text bridge is the first (and V1's only new) implementation of the `MixtureSource` protocol; its schema mapping (§5.3) doubles as the contract every future adapter must satisfy.

### 5.1 Interface (verified against the vision branch)

`MixtureDataLoader` requires only a map-style source: `__len__` and `__getitem__(int) → dict[str, np.ndarray]`, **index-deterministic across calls** (epoch shuffling is done by the loader's `reshuffle(epoch)`; resume replays previously consumed refs; no epoch is ever passed to the dataset). `len()` must agree across ranks — guaranteed by the existing rank-0-builds-cache + barrier pattern (`olmo_core_utils.load_dataset_distributed`).

Text examples participate in the 2D-knapsack packs for free: the packer keys on `len(input_ids)` and `len(images)`; a zero-crop example consumes no image budget and mixes with image examples in one pack (explicitly supported upstream; only the legacy next-fit path refuses to mix).

### 5.2 Tokenization decision: reuse open-instruct token ids directly

The bridge consumes open-instruct's pre-tokenized rows (`sft_tulu_tokenize_and_truncate_v1` + `sft_tulu_filter_v1`) **as-is** and does not re-encode through the vision branch's layout modules. Rationale:

1. This is the point of the merge: the text half must behave exactly like Olmo text post-training SFT — open-instruct's chat template, its tools-aware assistant-span label derivation, its mixer and content-addressed caching. Re-encoding via the vision layout would silently reintroduce mm_olmo's text dialect (system turn folded into user text, loss only on the final EOS) and drop tools support.
2. Vocab safety: text ids are always below the VLM's base vocab (`SplitVocabEmbedding` base block), hence valid LM-head targets. The bridge asserts `ids.max() < base_vocab_size` per example.
3. Chat-template consistency is a **per-backbone run invariant enforced at config time**, not by re-tokenizing: the entry point pins (tokenizer artifact, open-instruct `chat_template_name`, image chat layout) as one bundle — Qwen3 layout + Molmo2 tokenizer for Molmo2 backbones; `allenai/olmo-3-tokenizer-instruct-dev` + `chat_template_name=olmo123` (the tokenizer-native template fallback, §2.1) + `olmo3_layout` for the Olmo 3 backbone (§7). `qwen3_layout` itself calls `tokenizer.apply_chat_template`, so image and text data already share the template source when they share the tokenizer.

Residual dialect deltas vs mm_olmo's tulu4 source (per-turn `<|im_end|>` supervision; system rendered as a real system turn) are deliberate and should be flagged in parity comparisons.

### 5.3 Schema mapping

Open-instruct rows are unshifted (`labels[i]` is the target *at* position i, −100 = masked); the vision schema is **already next-token-shifted** (`labels[i] = input_ids[i+1]`, loss masks shifted to match; the train module applies no additional shift).

| bridge output | construction |
|---|---|
| `input_ids` | `oi_ids` unchanged (int64) |
| `labels` | `labels[i] = oi_ids[i+1]` for `i < n−1`; last = eos id (don't-care, mask 0) |
| `loss_masks` | `m[i] = (oi_labels[i+1] != −100)`, `m[n−1] = 0`; scaled by `w = 2/sqrt(m.sum())` (`root_tokens` weighting, parity with the tulu source) and an optional scalar `message_weight` (default None). `root_subsegments` skipped — a no-op for single-branch text. float32 |
| `position_ids` | `arange(n)` |
| `token_type_ids` | `zeros(n)` (1 marks image-structural tokens) |
| `subsegment_ids` | omitted (packer fills zeros) |
| `images` | `zeros((0, 729, 588), float32)` — trailing dims must be the real `(N_PATCHES_SQ, PATCH_DIM)`; the collator and packer read `shape[1]/[2]` off empty arrays |
| `pooled_patches_idx` | `full((0, 4), −1, int64)` |
| `attention_mask` (oi) | dropped (always all-ones upstream) |

Constructor surface: `OpenInstructTextDatasetConfig(mixer_list, mixer_list_splits, max_seq_length, transform_fn=(sft_tulu_tokenize_and_truncate_v1, sft_tulu_filter_v1), loss_token_weighting="root_tokens", message_weight=None, cache args) .build(tc: TokenizerConfig, base_vocab_size) → OpenInstructTextDataset`. `build()` uses `get_cached_dataset_tulu_with_statistics` via the rank-0-first pattern, then `dataset.set_format("numpy", columns=["input_ids", "labels"])`.

### 5.4 Prerequisite fix and guards

- **`keep_in_memory` plumb-through (required):** `LocalDatasetTransformationCache` currently does `load_from_disk(cache_path, keep_in_memory=True)`, which would materialize the full tokenized text mix in every rank's process (multiple GB × 8 local ranks). Thread a `keep_in_memory: bool = True` parameter through `get_cached_dataset_tulu_with_statistics` → the cache, and pass `False` from the bridge. Arrow mmap is read-thread-safe, so the loader's prefetch threads are fine.
- Startup assert that the bridge's tokenizer and the multimodal pipeline's tokenizer are the same artifact (probe-encode a fixed string, compare ids).
- Pass the mixture sequence length (16384) as the bridge's `max_seq_length` so open-instruct truncation effectively never fires; the collator tail-truncates as a last resort, which is safe for text (no image block to orphan).
- Set `epoch_instances` explicitly on the loader for step-bounded runs — the default epoch size is the sum of source lengths, which a large text mix inflates.

### 5.5 Alignment with the validated Olmo 3 SFT recipe

For Olmo 3 backbones, the bridge's defaults follow `oc_sft_olmo3_7b_1node.sh` (§2.1) exactly: `mixer_list = allenai/Dolci-Instruct-SFT 1.0`, tokenizer `allenai/olmo-3-tokenizer-instruct-dev`, `chat_template_name=olmo123`, `add_bos=False`, and the same `local_cache_dir` (`/weka/oe-adapt-default/allennlp/numpy_sft_cache`'s dataset-transformation sibling). Because the tokenization cache key hashes (tokenizer config, mixer, transform fns, max_seq_length, seed), the bridge's tokenized text should be produced by the **same arguments the validated text-only pipeline uses** wherever they overlap — this makes the merged stage's text half byte-comparable to the text-only baseline, and is verified by the tokenization-parity test in §8. One deliberate difference: the bridge tokenizes at the multimodal sequence length (16384), not the text pipeline's 32768, so the cache entries are distinct by construction.

Sizing: under `olmo123`, every conversation without a system message gets the tokenizer template's function-calling system block — ~840 tokens/row vs ~643 under `olmo_thinker` (§2.1). Budget the nlp group's token share and `epoch_instances` with the ~840 figure, and note the block competes with image tokens inside the 16384 pack budget.

- **mason.py:** the new entry point is deliberately **not** added to `OPEN_INSTRUCT_COMMANDS`. mason's auto-dataset-cache would re-run the command locally with `--cache_dataset_only` (an argument this path doesn't have — there is no numpy pre-tokenization stage) and rewrite `--output_dir`. Consequences: launch scripts set the wandb entity and an explicit weka `--output_dir` themselves, and keep `--no_auto_dataset_cache` for intent.
- **Weka-only in V1:** the image datasets live at `MOLMO_DATA_DIR` (default `/weka/oe-training-default/mm-olmo`). mason auto-mounts both weka buckets when all `--cluster` values are weka clusters (e.g. `ai2/jupiter`). The entry point fail-fasts with a clear message if `MOLMO_DATA_DIR` is not a directory. Non-weka clusters (e.g. augusta) are unsupported in V1.
- Env vars in scripts: `OLMO2_FLEX_ATTN=1` (FlexAttention path for the multimodal masks), `VIT_CROP_MICROBATCH=16`.
- Dependencies: repoint `pyproject.toml`'s `ai2-olmo-core` git rev to the vision-branch SHA and `uv lock` (PR 0). open-instruct's `datasets>=4` pin is compatible (the vision branch carries `dataset_compat` for datasets 5.x); `transformers` must support `AutoModelForImageTextToText` and Molmo2 remote code.
- Debug script sketch (`scripts/train/debug/mm_sft.sh`): 1 GPU on `ai2/jupiter`, `--mixture debug --max_train_steps 10 --global_batch_instances 1 --rank_microbatch_instances 1`, compile off, HF init from `allenai/Molmo2-4B`. Production script (`scripts/train/vision/molmo2_stage2.sh`): 8 GPU, full parity defaults, `--mixture image-only-v9`, optional `--model_name_or_path <stage-1 weka ckpt>`, resumable. Both launched via `./scripts/train/build_image_and_launch.sh <script>`.

## 7. Olmo 3 backbone workstream (OLMo-core `vision` branch)

The Molmo2 presets are Qwen3-only in three places: hardcoded special-token ids (`molmo2_tokens.py`, base 151936), the `_molmo2_like` factory (`image_patch_token_id` hardcode, Qwen3 `TransformerConfig`s), and the `qwen3_layout` chat template imported directly by `message_sequence.py`. Enablement changes (all upstream, in the vision branch):

1. **`molmo2_tokens.py`:** a frozen `SpecialTokenIds` dataclass with `from_base(base_vocab)` (mm_olmo's offset order: im_start+0, im_end+1, im_patch+2, im_col+3, low_res_im_start+4, `<|image|>`+5) and `from_tokenizer(tokenizer)`. `MOLMO2_TOKENS = from_base(151_936)`; `OLMO3_TOKENS = from_base(100_352)` (the dolma2 padded vocab size — the 74 padding slots above dolma2's 100278 are too few for the 128-token block, so extras sit above 100352, exactly the mm_olmo OLMo-backbone precedent). Keep old constants as aliases.
2. **`multimodal.py`:** `_molmo2_like` gains an `image_patch_token_id` parameter; add a `molmo3_7B()` preset — `TransformerConfig.olmo3_7B(vocab_size=100_352, n_extra_vocab=128)`, `connector_mlp_hidden_size=11_008`, and an **attention-backend override** (olmo3 defaults to flash_2; the multimodal bidirectional/subsegment masks need the torch or flex backends).
3. **New `data/multimodal/olmo3_layout.py`** with the same function API as `qwen3_layout.py`, rendering **the `allenai/olmo-3-tokenizer-instruct-dev` tokenizer's built-in template** — the one the released Olmo 3 Instruct models used and the one the validated text pipeline trains with via the `olmo123` fallback (§2.1) — not the `olmo` template registered in `dataset_transformation.CHAT_TEMPLATES`. Two real deltas from qwen3 force a separate module: the default system turn (header extraction must locate the last `<|im_start|>user\n`, not the first `<|im_end|>`), and the final assistant turn closing with the eos token rather than `<|im_end|>`. No BOS. Add a golden-token test that renders both through `tokenizer.apply_chat_template` and asserts identity with the text pipeline's rendering.
4. **Layout threading:** `message_sequence.py::encode_sft_example` and `sequence_builder` default to qwen3; add a `layout` knob (name or module) plus tokenizer-derived `image_token_ids` threaded through the dataset config dataclasses. Mechanical but the widest-touch change.
5. **Stage-1 generalization:** `Molmo2-Stage1.py::_init_weights_from_scratch` takes a `model_type` so `convert_state_from_hf` can load olmo-family LMs (`allenai/Olmo-3-1025-7B`); the SigLIP2 ViT init is backbone-agnostic already.
6. `SplitVocabEmbedding` and `TransformerConfig.n_extra_vocab` need **no** changes — already fully parameterized.

**The critical dependency: no Olmo 3 stage-1 checkpoint exists.** Stage-2-only SFT on an Olmo 3 backbone would start from a random connector and random extra-token embeddings — not the Molmo recipe, and expected to underperform badly. Plan of record: land the changes above, run stage 1 (caption pre-training) with the `molmo3_7B` preset in OLMo-core, and have open-instruct stage 2 consume that checkpoint via `--model_name_or_path`. A stage-2-only connector warm-up (freeze LM, high connector LR) is a fallback experiment, not the plan.

Open items: the tokenizer artifact — extend `allenai/olmo-3-tokenizer-instruct-dev` (the artifact the validated text pipeline uses, §2.1) with padding fillers to 100352 plus the EXTRA_TOKENS block, and decide publish-as-HF-repo vs build-on-the-fly; the bridge and image pipeline must share the extended artifact, and its base ids must remain byte-identical to the unextended one so the text cache stays valid. Sliding-window attention (3 of 4 olmo3 layers) × bidirectional image-token masks is untested — needs a correctness test, with `sliding_window=None` in the preset as the fallback (diverges from text Olmo 3; flag to modeling owners). RoPE scaling: **a hard requirement, not a length knob** — Olmo-3-1025-7B's weights are YaRN-trained, and loading them without the matching scaling collapses training (CE ≈ 8 at every LR tried; §2.1). The `molmo3_7B` preset must bake in the same YaRN config as the base checkpoint (`with_rope_scaling`, factor 8), even though multimodal stage 2 trains at 16384. This also rules out ulysses context parallelism for the lineage (incompatible with scaled RoPE) — consistent with the multimodal train module, which has no CP path. Attention backend: the text recipe uses flash_2, which the multimodal masks cannot use — the merged run's LM half runs on torch/flex SDPA instead; numerically benign but worth noting in parity comparisons.

open-instruct's only coupling to all of this: a small backbone map (preset name, layout name, tokenizer artifact, open-instruct `chat_template_name`) and pass-through of `--model_preset` / `--tokenizer_name_or_path`.

## 8. Testing and verification

**CPU unit tests** (no weka, no GPU):
- Mixture math: group rewrite (`nlp_source`/`nlp_rate`), weight normalization vs hand-computed sqrt-size values, `open_instruct_text` substitution, subset pruning.
- Train-module config: group LRs land on `connector.*`/`vision.*`, scheduler names align, `rank_microbatch_size = instances × seq`, HSDP iff multi-node.
- Collator passthrough with synthetic zero-crop + image examples (fixed-length right-pad, dummy crop for all-text batches, float loss_masks preserved).
- Bridge: schema/dtype conformance, shift correctness against a hand-built conversation, root-tokens weighting, max-id assert, packer + collator round-trip with one synthetic image example.
- Text-half tokenization parity: a fixed conversation tokenized through the bridge's config (Olmo 3 bundle, §5.5) must produce ids identical to the validated text pipeline's tokenization (`oc_sft_olmo3_7b_1node.sh` arguments), modulo the sequence-length difference.
- Arg parsing round-trip; unknown `--mixture` raises listing valid keys.

**GPU tests** (`*_gpu.py`, via the standard GPU-test Beaker path):
- Tiny `MultimodalLMConfig` (2 layers, tiny ViT): one `train_batch` through `MultimodalTransformerTrainModule` with a packed text+image batch; finite loss; `freeze_params=["vision.*"]` leaves vision grads absent.
- Checkpoint save/load round-trip including `MixtureDataLoader` state: the first batch after restore equals the uninterrupted run (packing/resume determinism).

**Beaker validation:**
1. `mm_sft.sh` 10-step smoke green (HF init) + a stage-1 `--model_name_or_path` smoke.
2. **Parity run (done):** 50 steps, single-source mixture `pixmo_count_train`, seeds 6198/50189, this entry point vs upstream `Molmo2-Stage2.py` on the same 2×H100 — step-1 CE identical to 4 decimals (0.4734), max |diff| 0.0032 / mean 0.0011 across common steps (accumulated bf16 nondeterminism). Note: strict per-step parity requires a single-source mixture — upstream renormalizes subset weights over the full group membership while `build_mixture` prunes groups first, so multi-source subsets like `debug` have identical within-group ratios but different group-vs-group weights (the full `image-only-v9` mixture is identical on both sides).

## 9. Risks

| Risk | Mitigation |
|---|---|
| OLMo-core `vision` branch is a moving target (13 ahead / 4 behind main at time of writing); rebases could invalidate the pinned SHA | Pin by SHA; ask OLMo-core owners to tag it; repoint deliberately with a parity re-run |
| Weka-only data — no CI coverage of real datasets | Every dataset-touching test skips without `MOLMO_DATA_DIR`; smoke coverage via Beaker debug runs |
| RAM: `keep_in_memory=True` cache loads × 8 ranks | §5.4 plumb-through is a hard prerequisite (PR A1) |
| Packing determinism on mid-epoch resume | Covered by the GPU round-trip test; fallback: coarse resume (restart epoch) + upstream issue |
| `load_path` vs `save_folder` precedence for stage-1 init + preemption | Verify on the pinned SHA before relying on it |
| Missing `retie_word_embeddings` in the `to_empty` branch = silent corruption | Encapsulated in `setup_multimodal_model`; unit-tested on meta/cpu |
| Bridge emitting ids ≥ base vocab as targets (`SplitVocabEmbedding` extra block is inputs-only) | Per-example assert |
| `qwen3_layout` hardwired in upstream datasets | Upstream layout knob is an explicit Olmo 3-workstream requirement (§7.4) |
| datasets 5.x compat shim (`dataset_compat`) is process-global | Bridge avoids `load_from_disk_compat`; keep `datasets < 6` behaviorally |

Findings from the first smoke runs (attempts 1–8, all root-caused): OLMo-core's `beaker` callback needs `beaker-gantry` (§3); the step-0 pre-train checkpoint force-allocates full fp32 Adam states via torch DCP's `_init_optim_state` and OOMs HF-init runs (skipped via `pre_train_checkpoint=False`); FlexAttention in eager mode (compile off) OOMs at seq 16384 — the smoke must run compiled like Stage2 production; and Molmo2-4B at Stage2-parity settings on 2×H100 peaks at ~81.1 GiB on the image-heavy rank — over jupiter H100s' 81,090 MiB but under some other hosts' 81,559 MiB — so the smoke runs `VIT_CROP_MICROBATCH=8` + per-block LM AC for headroom. Upstream `Molmo2-Stage2.py`'s documented 1-GPU smoke recipe hits the same pre-train-checkpoint and eager-flex issues and appears to be untested; reported upstream as OLMo-core #846 (step-0 checkpoint OOM), #847 (eager-flex OOM), #848 (compile_vision stride assert), #849 (system-turn template), #850 (BeakerCallback gantry dependency).

Findings from the first merged-mixture run (image sources 0.1 / Dolci 0.9 via the `open_instruct_sft` adapter, 5 steps green): (1) the Molmo2 tokenizer's built-in chat template rejects system turns ("roles must alternate user/assistant") — real SFT mixes need `--text_chat_template_name olmo` (near-identical ChatML on the Qwen vocab, system-capable) or mm_olmo-style system flattening; (2) `compile_vision`/`compile_connector` hit an inductor saved-tensor stride assertion in backward when crop counts swing hard between batches (a text-heavy mixture alternates zero-crop and image packs) — vision and connector run eager until fixed upstream; the compile-for-memory requirement only concerns the LM's FlexAttention.

Deferred (post-V1): HF export (`multimodal_lm_state_dict_to_hf` wiring into a convert script — note the text-model converter `scripts/train/convert_olmo_core_to_hf.py` was fixed by #1809; the multimodal variant still needs its own key mappings), in-loop eval (the `VALIDATION_MIXTURES` bisect ladder), multi-node HSDP validation, inline mixture specs, non-weka data roots.

## 10. Milestones

open-instruct (`vision` branch):

1. **PR 0** — branch + repoint `ai2-olmo-core` rev + `uv lock` + guarded import smoke.
2. **PR A1** — `keep_in_memory` plumb-through in `dataset_transformation.py` + test.
3. **PR 1** — `olmo_core_multimodal_utils.py` (model/train-module dataclasses + builders) and `sft_mixture.py` (protocol, `SourceSpec`, `SOURCE_REGISTRY`, `build_mixture`) + CPU tests.
4. **PR 2** — entry point + `mm_sft.sh`; evidence: 10-step Beaker smoke (HF init) + stage-1-init smoke.
5. **PR A2** — `sft_text_dataset.py` adapter + tests, including the §5.5/§8 tokenization-parity check against the validated Olmo 3 recipe.
6. **PR 3** — adapter integration (`SourceSpec` overrides, `--nlp_source open_instruct` sugar), contract test against the `MixtureSource` protocol; Olmo 3 bundle defaults (Dolci-Instruct-SFT + `olmo-3-tokenizer-instruct-dev` + `olmo123`).
7. **PR 4** — `scripts/train/vision/molmo2_stage2.sh`, GPU tests, 50-step parity report, docs + CHANGELOG.

OLMo-core (`vision` branch, parallel):

1. **PR B1** — `SpecialTokenIds` + preset parameterization + `molmo3_7B` preset + unit tests.
2. **PR B2** — `olmo3_layout.py` + layout threading + golden-token tests against the open-instruct `olmo` template render.
3. **PR B3** — stage-1 from-scratch init generalization + SWA×multimodal-mask correctness test.
4. Olmo 3 stage-1 training run (compute, not code) → the checkpoint open-instruct stage 2 consumes.
