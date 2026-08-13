# Qwen 3.5 MoE SFT with OLMo-core

This is the operator guide for SFT of `Qwen/Qwen3.5-35B-A3B-Base` through
Open Instruct's OLMo-core trainer. The validated branches are
`jacobm/olmoe3-post-training` in both `open-instruct` and `OLMo-core`.

## Fixed choices

- Model config and weights: `Qwen/Qwen3.5-35B-A3B-Base`
- Tokenizer: `Qwen/Qwen3.5-35B-A3B`
- Training template: `olmo_thinker`
- BOS: none
- EOS: `<|im_end|>` (`248046`)
- PAD: `<|endoftext|>` (`248044`)
- Padded model vocabulary: `248320`
- HF export context cap: `65536`
- HF generation EOS: `[248046, 248044]`, matching the official Qwen 3.5 generation config
- Beaker image: `01KZVZ95WZKYDRS83W95P16DR2`

Use the non-base tokenizer deliberately. The base model supplies the architecture and
weights; the public chat model supplies complete chat/generation metadata. The exporter
copies the locally packaged `olmo_thinker` template rather than the Qwen chat template.

## OpenThoughts Agent lengths

Lengths include the complete `olmo_thinker` rendering with the Qwen 3.5 tokenizer.

| Dataset | Rows | Mean | Max | Rows over 32K | Rows over 65,536 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `OpenThoughts-Agent-SFT-1K` | 1,000 | 17,064 | 61,810 | 51 | 0 |
| `OpenThoughts-Agent-SFT-10K` | 10,000 | 17,024 | 62,498 | 514 | 0 |
| `OpenThoughts-Agent-SFT-100K` | 94,334 | 16,876 | 63,822 | 4,848 | 0 |

Use `65536` to retain every example. Detailed JSON statistics live under
`datasets/OpenThoughts-Agent-SFT-*/qwen3.5-35b-a3b-olmo_thinker/`.

The datasets expose canonical `conversations` records. The transformation layer adds a
`messages` column only when one is absent; existing `messages` datasets are unchanged.

## Tokenization

From the Open Instruct checkout:

```bash
BEAKER_IMAGE=01KZVZ95WZKYDRS83W95P16DR2
DATASET_SIZE=1K bash scripts/train/qwen/qwen35_openthoughts_agent_tokenize.sh "$BEAKER_IMAGE"
DATASET_SIZE=10K bash scripts/train/qwen/qwen35_openthoughts_agent_tokenize.sh "$BEAKER_IMAGE"
DATASET_SIZE=100K bash scripts/train/qwen/qwen35_openthoughts_agent_tokenize.sh "$BEAKER_IMAGE"
```

The default outputs are
`datasets/OpenThoughts-Agent-SFT-${DATASET_SIZE}/qwen3.5-35b-a3b-olmo_thinker-65536`.
Tokenization is CPU-bound, so the launcher requests one GPU only to get a schedulable
Beaker worker. Override `BEAKER_CLUSTER`, `BEAKER_WORKSPACE`, or `NUM_GPUS` if needed.

## Import the base model

```bash
bash scripts/train/qwen/qwen35_35b_a3b_convert_hf_to_olmo.sh "$BEAKER_IMAGE"
bash scripts/train/qwen/qwen35_35b_a3b_verify_olmo_core_conversion.sh "$BEAKER_IMAGE"
```

The default OLMo checkpoint is
`checkpoints/qwen3.5-35b-a3b-base-olmo`. The import maps the hybrid schedule of 30
GatedDeltaNet layers and 10 full-attention layers, full-attention Q gates, packed linear
QKV/conv tensors, routed experts, and shared experts.

The validated base conversion maps all 693 HF parameters to 783 OLMo tensors. Exporting
it back to HF is bit-identical to the source model. Direct HF versus OLMo execution has
expected BF16/kernel drift while preserving the selected token: cosine similarity
`0.999288`, mean absolute logit difference `0.0589`, and top-1 agreement `1.0`.

## Training smoke

```bash
NUM_GPUS=4 EP_DEGREE=4 \
MAX_SEQ_LENGTH=4096 MAX_TRAIN_STEPS=2 COMPILE_MODEL=false \
bash scripts/train/qwen/qwen35_35b_a3b_olmo_core_sft_smoke.sh "$BEAKER_IMAGE"
```

The launcher defaults to `ai2/OLMo-3-moe-experiments`, urgent priority, on Holmes.
The validated minimum is EP4 on four B300s. EP2 runs out of memory while constructing
FP32 optimizer state.

Keep these activation-checkpoint settings:

```text
--moe_recompute_each_block false
--moe_checkpoint_block_internals true
```

Whole-block recomputation fails because routed expert token counts can differ between
the original forward and recomputation. Internal MoE checkpointing completed dry-run,
forward, backward, clipping, and two optimizer steps with finite losses and gradients.

For 65,536-token training, use Ulysses context parallelism degree 2 with EP8:

```bash
PROJECT_ROOT=/weka/oe-adapt-default/jacobm/olmoe3/post-training
DATASET_PATH="$PROJECT_ROOT/datasets/OpenThoughts-Agent-SFT-1K/qwen3.5-35b-a3b-olmo_thinker-65536" \
OUTPUT_DIR="$PROJECT_ROOT/checkpoints/qwen35-openthoughts-1k-65k-smoke" \
MAX_SEQ_LENGTH=65536 MAX_TRAIN_STEPS=1 \
NUM_NODES=1 NUM_GPUS=8 EP_DEGREE=8 \
CP_DEGREE=2 CP_STRATEGY=ulysses \
COMPILE_MODEL=true ACTIVATION_MEMORY_BUDGET=0.3 \
CHECKPOINTING_ENABLED=false \
bash scripts/train/qwen/qwen35_35b_a3b_olmo_core_sft_smoke.sh "$BEAKER_IMAGE"
```

Qwen 3.5's GatedDeltaNet layers do not support ring context parallelism. Ulysses is
supported by both GatedDeltaNet and FlashAttention 4. Degree 2 also divides the two KV
heads in every full-attention layer; higher CP degrees do not. The validated topology is
`(dp=4, cp=2)` for dense parameters and `(ep_dp=1, ep_mp=8)` for experts.

The one-node CP2/EP8 smoke completed the compiled forward/backward dry run and one real
optimizer step at 65,536 tokens. It used 208.2 GiB active GPU memory, with finite grad
norm `1.930`, CE loss `0.2757`, and PPL `1.317`. The corresponding no-CP EP8 run ran out
of memory in grouped expert projection. EP16 on two nodes is therefore unnecessary for
this sequence length.

Disabling checkpoint writes avoids writing roughly 388 GB of model plus FP32 optimizer
state for a one-step test. Enable final checkpointing for a real run. If a future data or
batch configuration needs more memory, try `ACTIVATION_CHECKPOINTING_MODE=selected_modules`
before raising EP: budget checkpointing cannot see through opaque GatedDeltaNet kernels,
while selected-module checkpointing can recompute those mixers explicitly.

## Export after SFT

```bash
CHECKPOINT_PATH=/absolute/path/to/stepN \
TOKENIZER_NAME=/absolute/path/to/tokenized-dataset/tokenizer \
OUTPUT_PATH=/absolute/path/to/hf-export \
bash scripts/train/qwen/qwen35_35b_a3b_convert_olmo_to_hf.sh "$BEAKER_IMAGE"

CHECKPOINT_PATH=/absolute/path/to/stepN \
TOKENIZER_NAME=/absolute/path/to/tokenized-dataset/tokenizer \
OUTPUT_PATH=/absolute/path/to/hf-export \
VERIFY_ONLY=true \
bash scripts/train/qwen/qwen35_35b_a3b_convert_olmo_to_hf.sh "$BEAKER_IMAGE"
```

The exporter accepts both plain converted checkpoints and OLMo DDP training checkpoints.
Verification requires exact equality for every mapped tensor and validates model token
IDs, official generation token IDs, the context cap, and the chat template. A successful
run writes `weight-verification.json` into the HF export.

## Validated artifacts

- Base OLMo checkpoint: `checkpoints/qwen3.5-35b-a3b-base-olmo`
- Base exact HF round trip: `checkpoints/qwen3.5-35b-a3b-base-olmo-roundtrip-hf`
- Two-step SFT checkpoint: `checkpoints/qwen3.5-35b-a3b-dolci-sft-smoke-ep4/step2`
- Two-step SFT HF export: `checkpoints/qwen3.5-35b-a3b-dolci-sft-smoke-ep4-step2-hf`
- 65K CP2/EP8 smoke: Beaker experiment `01KZW8CRCXP4DKRD8E5P3E7QJT`

The two-step SFT export was an exact match across all 693 HF parameters and retained the
byte-identical `olmo_thinker` template, EOS/PAD metadata, official generation EOS list,
and the 65,536-token cap.
