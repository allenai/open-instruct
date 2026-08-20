# Multimodal (VLM) SFT

`open_instruct/finetune.py` supports supervised finetuning of vision-language models. Multimodal
mode is **detected from the checkpoint**, not requested with a flag: if the model's `model_type` has
a plugin registered in `open_instruct/mm_plugin.py`, the multimodal tokenization chain, collator and
model class are selected automatically.

Supported today: `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`, `qwen3_vl_moe`, `qwen3_5`, `qwen3_5_moe`.

Qwen3.5 needs three extra things. `mm_token_type_ids` is required to place 3-D M-RoPE positions, and
the collator builds it automatically. Its stock chat template is not prefix-stable — it emits
`<think>` only for the assistant turn after the last user message — so pass
`--chat_template_name qwen3_5_nothink`. And its `head_dim` of 256 is more than FlashAttention-4 can
serve on SM100/SM110, so pass `--attn_implementation sdpa` on B300.

## Data format

One JSON object per row, with an `images` column and an `<image>` placeholder in the message text:

```json
{
  "messages": [
    {"role": "user", "content": "<image>Click on the search box."},
    {"role": "assistant", "content": "{\"action\": {\"name\": \"click\", \"x\": 369, \"y\": 78}}"}
  ],
  "images": ["screenshots/page_001.png"]
}
```

The number of `<image>` placeholders must equal the length of `images`, or the row is rejected with
an error naming the row's roles. Multiple images per conversation and images in any turn are
supported. Relative paths are resolved against `--media_dir`; absolute paths are used as-is.

## Launching

```bash
./scripts/train/build_image_and_launch.sh scripts/train/debug/finetune_multimodal.sh
```

Key flags:

| Flag | Description | Default |
|---|---|---|
| `--media_dir` | Root that relative image paths resolve against | `None` |
| `--image_max_pixels` | Per-image pixel cap applied before the model's image processor | `768*768` |
| `--image_min_pixels` | Per-image pixel floor | `32*32` |
| `--freeze_vision_tower` | Freeze the vision encoder | `True` |
| `--freeze_multi_modal_projector` | Freeze the vision→text projector | `False` |
| `--freeze_language_model` | Freeze the language model | `False` |

`--image_max_pixels` controls the cost/quality tradeoff: it sets how many image tokens each image
becomes. It is part of the dataset cache key, because changing it changes `input_ids`.

## How it works

The design keeps open-instruct's existing text machinery untouched and adds two pieces around it.

1. **Placeholder expansion before tokenization** (`open_instruct/mm_plugin.py`). Assistant label
   spans are derived from character offsets over the rendered chat template
   (`_tokenize_tulu_sft_with_assistant_labels`), which is text-only. So before the chat template
   runs, each `<image>` in the message *text* is replaced by exactly as many real image tokens as
   the vision tower will emit — for Qwen, `prod(image_grid_thw) // merge_size**2` tokens wrapped in
   `<|vision_start|>`/`<|vision_end|>`. After that the conversation is plain text again and offset
   based labeling works unchanged. Image tokens are always masked out of the loss, since they sit in
   the user turn.

2. **Pixels are never cached** (`open_instruct/mm_collator.py`). The tokenized dataset carries image
   paths; `MultiModalDataCollator` runs the image processor once per batch and merges
   `pixel_values` / `image_grid_thw` into the batch, cast to the model's compute dtype.

The final checkpoint gets the processor saved alongside the tokenizer, so it can be served without
pointing at the base model for image-processing config. On transformers 5 that lands in
`processor_config.json` (which embeds the image and video processor configs) plus
`chat_template.jinja`, rather than the older `preprocessor_config.json`.

Adding a model family is a subclass implementing `image_token_counts` and `format_image_tokens`,
plus entries in `MM_PLUGIN_REGISTRY` and `COMPOSITE_MODULES`. Only add a family you have actually
tested end to end: a token count that is off by one trains without error and silently degrades the
model.

## Things worth knowing

- **Over-length rows are dropped, not truncated.** Truncating a multimodal sequence can cut through
  the middle of an image-token block, which fails deep in the model forward. The transform chain is
  `sft_tulu_tokenize_mm_v1` → `sft_tulu_filter_v1` → `sft_mm_max_length_filter_v1`; the last one
  drops rows longer than `--max_seq_length`. Watch how many rows this removes when tuning
  `--image_max_pixels`.
- **All-text batches get a dummy image.** Under ZeRO-3, a rank whose batch never touches the vision
  tower does not participate in that parameter's collective and the job hangs. The collator appends a
  64×64 white image to such batches, with `attention_mask=0` and `labels=-100` so it affects neither
  attention nor the loss. It contributes no gradient by design; its purpose is to keep every rank's
  forward pass structurally identical.
- **A frozen vision tower is not activation-checkpointed.** ZeRO-3 releases frozen parameters after
  the forward pass and never re-gathers them, but gradient checkpointing needs them again to
  recompute — torch then raises `CheckpointError: Recomputed values ... have different metadata`
  with shape-`[0]` tensors. So with `--freeze_vision_tower` and `--gradient_checkpointing`, the
  vision tower's checkpointing flag is turned off (the language model keeps it). This costs nothing:
  a frozen tower whose inputs are pixels has no backward pass to trade compute for.
- **LoRA target modules are filtered.** Qwen's vision blocks have their own `gate_proj`/`up_proj`/
  `down_proj`, so the default target list would train the vision encoder by accident. Targets are
  expanded to full module names and vision-side matches removed.
- **`--freeze_vision_tower` does not freeze the projector.** Qwen nests the projector inside
  `visual`, so the frozen key list names the tower's sub-modules (`visual.patch_embed`,
  `visual.blocks`) rather than `visual` itself.
- **Image tokens count as tokens** in the throughput and token-accounting metrics.

## Not supported

These raise an explicit error rather than failing subtly:

- `--packing` — `TensorDataCollatorWithFlattening` is text-only, and packed multimodal batches need
  per-sub-sequence vision slicing and a packed mrope path.
- `--sequence_parallel_size > 1` — splitting the sequence across ranks breaks the image-token to
  vision-feature correspondence.
- `--dataset_cache_mode hf` — a media-bearing dataset should not be pushed to the Hub.
- QLoRA and liger-kernel.

`position_ids` are deliberately not computed in the collator: the Qwen VLM forward derives its 3-D
mrope positions internally when they are absent. Any future work on packing or sequence parallelism
will have to build them explicitly.

## Checkpoint integrity

Two failure modes here are silent, so both are now checked automatically.

**Weights that do not load.** Under DeepSpeed ZeRO-3, some composite VLM checkpoints (Qwen3.5) load
with every `model.language_model.*` key missing: the vision tower loads, the language model is
randomly initialised, and training proceeds from loss ≈ ln(vocab) on a curve that descends fast
enough to look healthy. `finetune.py` now reads `output_loading_info` and refuses to start when
weights are missing; `--allow_missing_checkpoint_keys` overrides it. **Use ZeRO-2 for Qwen3.5.**

**Corrupted tensor names on save.** `save_pretrained(save_original_format=True)` — the default —
rewrites keys into a legacy layout for backwards compatibility. For Qwen3.5 that mapping is wrong:
it emits `model.language_model.language_model.language_model.layers.0…` and folds the vision tower
under the language-model prefix, producing a checkpoint nothing can load. `save_with_accelerate`
detects a repeated path component and re-saves in the model's native layout. Qwen2.5-VL is
unaffected and its output is unchanged.
