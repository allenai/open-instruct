> Copied into open-instruct from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 `ladders/olmoe3/vllm_plugin/README.md`; see
> [../README.md](../README.md) for how open-instruct uses it. Install paths below that name scaling-ladders map to
> `scripts/export/olmo35_mm/{transformers_plugin,vllm_plugin}` here.

# OLMoE3 vLLM plugin

Registers `Olmo3MoeForCausalLM` (and, language-model-only, the multimodal export's
`Olmo3MoeVisionForConditionalGeneration`) with vLLM and serves converted OLMoE3
KDA + full-width EMo checkpoints. Install this package and the sibling
`olmoe3-transformers-plugin` into the vLLM environment. The transformers plugin is
not a declared dependency, because it isn't published on PyPI, so install both from
the same scaling-ladders ref.

The first implementation supports tensor, pipeline, and expert parallel size 1.
It uses vLLM's KDA four-state cache (Q/K/V convolution state plus FP32 recurrent
state) and paged KV cache for full-attention layers.

## Supported vLLM versions

The plugin supports vLLM 0.19.1.x and 0.26.x. Other versions inside the broad
package-resolution interval are rejected at import time; the interval exists
because Python package metadata cannot express two disjoint version bands.

| Surface | vLLM 0.19.1.x | vLLM 0.26.x |
|---|---|---|
| KDA layer | `layers.kda.KimiDeltaAttention` | `layers.mamba.gdn.kimi_gdn_linear_attn.KimiGatedDeltaNetAttention` |
| KDA kernels | `layers.fla.ops.kda` | `third_party.flash_linear_attention.ops.kda` |
| KDA construction | Explicit layer index, hidden size, cache/model configs | HF config, `VllmConfig`, prefix |
| Convolution cache | Four separately allocated Q/K/V/recurrent states | Packed convolution state plus recurrent state |
| Expert mapping | `FusedMoE.make_expert_params_mapping` | Module-level `fused_moe_make_expert_params_mapping` |

Both adapters use the chunk KDA kernel for decode. This avoids the rectangular
K/V output truncation in the fused recurrent path and preserves recurrent state
as `[value_heads, value_dim, key_dim]`.

```bash
uv pip install -e ladders/olmoe3/transformers_plugin \
  -e ladders/olmoe3/vllm_plugin
vllm serve /path/to/converted/hf/checkpoint --tensor-parallel-size 1
```

## Multimodal exports, language model only

The converter writes a multimodal OLMoDDP checkpoint as one whole-model directory
(`model_type: olmo3moe_vision`, architecture `Olmo3MoeVisionForConditionalGeneration`,
tensors `language_model.*` / `vision_tower.*` / `connector.*`). The plugin registers
that architecture too, but serves only its language model: pass
`language_model_only=True` (olmo-eval: `provider.kwargs.language_model_only=true`),
and the executor builds exactly the `Olmo3MoeForCausalLM` of `text_config`, loads the
`language_model.*` tensors through the text loader, and counts and skips the
`vision_tower.*` / `connector.*` tensors (any other name is an error, and vLLM's strict
loader check still proves every language-model parameter was initialized). Without
the flag the executor raises `NotImplementedError`: vision inference is not
implemented. The export's `config.json` must be readable without `trust_remote_code`,
which the sibling transformers plugin provides by registering `olmo3moe_vision`.

## Numerical qualification

The exported `moe_use_core_numerics` setting selects the HF reference expert
implementation. vLLM continues to use its optimized fused experts, whose
low-precision activation and reduction order can differ from that reference;
its KDA prefill/decode kernels also have separate rounding behavior. Strict
core-to-HF conversion validates the exported HF model. It does not establish
HF-to-vLLM parity, which needs its own prompt, logit, and generation checks on
the exact runtime and checkpoint.
