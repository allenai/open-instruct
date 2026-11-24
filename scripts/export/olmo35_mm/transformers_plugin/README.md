> Copied into open-instruct from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 `ladders/olmoe3/transformers_plugin/README.md`; see
> [../README.md](../README.md) for how open-instruct uses it. Install paths below that name scaling-ladders map to
> `scripts/export/olmo35_mm/{transformers_plugin,vllm_plugin}` here.

# OLMoE3 Transformers plugin

This package registers the OLMo Core `olmo3moe` Hugging Face implementation
used by the OLMoE3 KDA + LatentMoE ladder and provides a checkpoint
conversion CLI.

```python
from olmoe3_transformers_plugin import register

register()
```

Convert an OLMo Core checkpoint:

```bash
uv run python -m olmoe3_transformers_plugin.convert_checkpoint \
  -i /path/to/step1234 \
  -o /path/to/hf-output \
  --tokenizer allenai/dolma2-tokenizer \
  --tokenizer-revision 5292e5d6c0f40b67cc765fe41bec991cf4345b5c
```

The exported config records KDA, NoPE attention, LatentMoE, routed-expert, and
global-load-balancing metadata. The current ladder's evaluation pool contains
all routed experts, making inference equivalent to ordinary global top-k
routing. Other evaluation-pool sizes fail closed during export.

New exports also set `moe_use_core_numerics=true`. HF then reproduces the
source no-EP expert inference path: packed SwiGLU with one activation rounding,
followed by TransformerEngine index-map combination in routing-slot order. That
combiner rounds its probabilities and each fused multiply-add to the activation
dtype; it is not an FP32 reduction. Existing HF configs without this field keep
the original HF expert implementation. This is the normal exported model's
forward behavior, not a validation-only override.

KDA recurrent caching is not implemented. Use `use_cache=False`.

## Multimodal checkpoints

A checkpoint whose `model` config is an olmo-core `MultimodalLMConfig` (the
OLMoE3 language model plus a vision tower and connector, written by
`MultimodalOLMoDDPTrainModule`) converts with the same command into one
whole-model directory. The language model goes through the text-only path
above unchanged, so its tensors and config are byte for byte what a text-only
export of the same LM produces; the layout then becomes:

- `config.json`: `architectures: ["Olmo3MoeVisionForConditionalGeneration"]`,
  `model_type: "olmo3moe_vision"`, `text_config` (the text export's
  `Olmo3MoeConfig` dict, `auto_map` included), `vision_config` and
  `connector_config` (the olmo-core `VisionEncoderConfig` /
  `VisionConnectorConfig` dicts), `image_patch_token_id`, `vit_layers`,
  `olmo_core_config` (the checkpoint's full `config.json`) and the top-level
  token ids / dtype of the text export.
- `model.safetensors`: `language_model.<text export names>`,
  `vision_tower.<olmo-core VisionTransformer names>` and
  `connector.<olmo-core VisionConnector names>`, in the text export's dtype.
- The text team's `modeling_olmo3moe.py` / `configuration_olmo3moe.py` /
  `moe_numerics.py` unchanged, plus `configuration_olmo3moe_vision.py` /
  `modeling_olmo3moe_vision.py` (copied from
  `olmoe3_transformers_plugin/hf_vision/`): an HF wrapper whose
  `.language_model` is `Olmo3MoeForCausalLM` and whose `vision_tower` /
  `connector` modules hold the vision weights. Its forward runs the text model;
  image inputs raise `NotImplementedError`.
- Tokenizer files as the text export.

The LM is validated twice with the strict 1e-4 logit check: by the text path on
its own export, and again on the wrapper's text forward from the final layout.
Source checkpoints hold no `model.*` entries: trainable parameters live as
`module.<name>.main` FP32 masters, frozen parameters as `frozen_model.<name>`
and persistent buffers as `model_buffer.<name>` (names `lm.*`, `vision.*`,
`connector.*`), resolved as olmo-core's `export_multimodal_olmoddp.py` does.

Install the pinned `olmo-core` submodule at
`b5db31dbdfed2228ed67ada7b0187be6cb5b4a37` before installing this package. It includes
[OLMo-core #872](https://github.com/allenai/OLMo-core/pull/872) plus the candidate
parameter-alignment and Torch 2.13 shared-expert stream fixes used in the
[B300 qualification campaign](../qualification/2026-09-25-b300.json). The
implementation stays in olmo-core; this package only registers its Auto classes
and exposes the conversion entry point. No model code is vendored here.
For the final architecture this includes independent per-head Q/K gains and
scalable-softmax weights, expanded-value KDA, latent projections, and router
normalization/scale restoration.

The CLI validates by default. `--skip-validation` explicitly bypasses numerical
forward validation, **not** the exact state-tensor roundtrip check. Conversion
correctness and inference-runtime qualification are different gates; see the
sibling vLLM README before interpreting fast benchmark results.

For SFT checkpoints, copy and verify the intended tokenizer/chat template and
generation configuration from the actual training recipe. This generic PT
converter does not guess a reasoning template or sampling settings.

`--tokenizer` is required by this plugin and wins over any tokenizer directory
beside the native checkpoint. For SFT, pass the **saved training tokenizer**,
not the PT example above: preserving token IDs does not establish that the chat
template or numeric splitting matches training. The companion core converter
requires a serialized `tokenizer.json`, preserves its backend, and checks an
independent `AutoTokenizer` save/reload. These checks are never disabled by
`--skip-validation`; a mismatch fails the export. `tokenizer-export-audit.json`
records the source, revision, backend hash, and reload result. Use the updated
companion core PR; older core revisions do not have this contract.

The package deliberately depends on core **without its training FLA extra**:
inference must use a separately pinned vLLM/Torch environment, not silently
upgrade Torch to satisfy the latest training stack.

## Tests

The dedicated `olmoe3-plugins` CI job installs that exact core revision with
Torch 2.13 and runs the plugin tests in `ladders/olmoe3/tests` (`test_plugins.py`,
`test_generation.py`, `test_kda_decode.py`). They cover Auto-class
registration, normalization arithmetic, and HF save/reload of latent hybrid
models with EMO enabled and disabled. GPU generation tests require a separate
pinned vLLM environment and `OLMOE3_RUN_GPU_TESTS=1`. The root test run skips
these plugin tests, since its environment lacks the vLLM plugin and pinned core.
