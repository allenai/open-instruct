# Exporting Olmo 3.5 multimodal SFT checkpoints

Everything a multimodal (vision + Olmo 3.5 MoE LM) SFT checkpoint needs between "an OLMo-core DCP step dir" and
"something the evals can load", without scaling-ladders. The interface is one directory in, three out:

```
STEP_DIR  (OLMo-core DCP: model_and_optim/ + config.json; a GPU run's save, or a TPU run converted to OLMo-core)
   ├─> OUT/olmo-core-export/   model.safetensors + olmo_core_config.json   vision evals (olmo-eval olmo_core_vlm)
   ├─> OUT/export/hf_text/     HF Olmo3MoeForCausalLM, LM only             text evals (vLLM / HF)
   └─> OUT/export/hf_mm/       HF Olmo3MoeVisionForConditionalGeneration   optional whole-model container
```

| Output | Made by | Read by |
|---|---|---|
| `olmo-core-export` | OLMo-core's `src/scripts/export_multimodal_olmoddp.py --verify` (bitwise against the train module's own loader) | olmo-eval's `olmo_core_vlm` provider: every image-QA eval. It runs OLMo-core's own `MultimodalLM`, so it needs OLMo-core at the multimodal SFT code (allenai/OLMo-core#942) |
| `hf_text` | [`scripts/train/debug/convert_moe_checkpoint_to_hf.py`](../../train/debug/convert_moe_checkpoint_to_hf.py) `--consolidated-export` (or `-i STEP_DIR -c STEP_DIR/config.json`): the `lm.*` weights, through the same converter the text team uses | olmo-eval text cells on vLLM (with the OLMoE3 vLLM plugin), HF `generate` |
| `hf_mm` | `python -m olmoe3_transformers_plugin.convert_checkpoint` from [`transformers_plugin/`](transformers_plugin/) | HF `AutoModel` with `trust_remote_code` (the text forward only; image inputs raise `NotImplementedError`); vLLM with [`vllm_plugin/`](vllm_plugin/) and `language_model_only=True`. Not used by the H231/H271 evals |

No HF or vLLM path in this directory runs the vision tower: the whole-model HF wrapper holds the vision and connector
weights under their OLMo-core names but implements only the text forward, and the vLLM executor serves the language
model only. Vision inference is `olmo-core-export` + OLMo-core.

## Stop tokens

Both HF exports write `generation_config.json` `eos_token_id: [100265, 100257]` (`<|im_end|>`, `<|endoftext|>`), as the
text SFT converter already did. The scaling-ladders converter took `eos_token_id` from the tokenizer config
(`100257` only), so vLLM, which stops on `generation_config` ids, ran past every `<|im_end|>`; on BFCL the model
repeated tool calls to the token budget (H271; the "stopfix" re-exports). `--eos-token-ids 100257` restores the base /
mid-training behaviour. `smoke_hf_text.py` fails an `hf_text` without the chat stop list.

## Run it

On Beaker (1 GPU, the OLMo-core training image; what [`scripts/eval/olmo35_mm/run_eval.sh`](../../eval/olmo35_mm/run_eval.sh)
submits as its `export` stage):

```bash
OLMO_CORE_DIR=/weka/.../OLMo-core  # allenai/OLMo-core#942 checkout, visible to the job
bash scripts/export/olmo35_mm/export_all_job.sh <run>/step10962 /weka/olmo-3p5-post-checkpoints/<user>/mmsft-eval/<label>
# EXPORT_HF_MM=1 also writes export/hf_mm; EXPORT_DEVICE=cpu skips the logit validation
```

It prints `MMSFT-VISION-EXPORT-OK` once `olmo-core-export` is verified and `TOKENIZER-CHECK OK` once `hf_text` passed the
smoke; the eval launcher waits on those markers. `emo_check_job.sh STEP_DIR OUT_DIR` then compares the HF text export's
logits with OLMo-core's for EMo-routed runs: the exports drop the `emo` router block, which is only exact when the eval
expert pool is all experts (`EMO-LOGIT-CHECK PASS`; `SKIP` for runs without EMo).

Language model only, by hand (CPU is fine with `--skip-validation`; strict validation needs a GPU and can fail on KDA
at this scale with a correct conversion, see the converter's docstring):

```bash
python scripts/train/debug/convert_moe_checkpoint_to_hf.py --consolidated-export OUT/olmo-core-export \
    -o OUT/export/hf_text -t <promoted olmo35 tokenizer> [--skip-validation --device cpu]
```

For the Olmo 3.5 SFT runs the tokenizer is the one the SFT data was tokenized with (think tags promoted to 100256 /
100266), e.g. the C12 numpy cache's `tokenizer/` directory (the default of `export_all_job.sh`).

### TPU checkpoints

A MaxText checkpoint is first converted to an OLMo-core DCP (`model.<name>` entries + the GPU run's `config.json`) by
the TPU conversion tooling (MaxText's `convert_olmo35_multimodal_checkpoint.py --mode maxtext_to_olmo_core`, with its
bitwise round-trip check). Everything here then treats it like a GPU save; `convert_moe_checkpoint_to_hf.py` reads
both the OLMoDDP layout (`module.lm.<p>.main`, `frozen_model.lm.<p>`) and the `model.lm.<p>` layout. MaxText's
checkpoint dir `S` holds the state after `S + 1` updates, so label it `s<S+1>` (`checkpoints/10961` -> `s10962`).

## The plugins

[`transformers_plugin/`](transformers_plugin/) (`olmoe3-transformers-plugin`) and [`vllm_plugin/`](vllm_plugin/)
(`olmoe3-vllm-plugin`) are copies of allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993
`ladders/olmoe3/{transformers_plugin,vllm_plugin}` (branch `jasonr/mixed-midtraining`; the multimodal parts are not on
scaling-ladders `main`). Every file names its source. Package and module names are unchanged, so they install in
place of the scaling-ladders ones. The one change: `convert_checkpoint.py --eos-token-ids` (default the chat stop
list). The text-model HF code itself (`Olmo3MoeForCausalLM`) is OLMo-core's (`olmo_core.nn.moe.v2.hf`).

```bash
pip install -e scripts/export/olmo35_mm/transformers_plugin --no-deps -e scripts/export/olmo35_mm/vllm_plugin
# or, from a pushed ref:
#   olmoe3-vllm-plugin @ git+https://github.com/allenai/open-instruct.git@<sha>#subdirectory=scripts/export/olmo35_mm/vllm_plugin
```

They are separate distributions on purpose: the vLLM plugin registers through the `vllm.general_plugins` entry point,
which must not load in every open-instruct vLLM process, and `register()` imports OLMo-core's MoE v2 HF classes.

SFT evals never use scaling-ladders. `scripts/eval/olmo35_mm/launch_text_cells.py` rewrites every recorded text cell to
install these copies from GitHub at an open-instruct commit, drops the BFCL cells' scaling-ladders archive steps, and
refuses a spec that still mentions scaling-ladders. Note for comparisons: H182/H231/H271's text cells were scored
with older plugin trees (a `scaling-ladders-hero` checkout for most cells; scaling-ladders `3542d99` for BFCL, whose
vLLM executor predates the one copied here), so numbers from those runs and from these cells are only comparable
after one checkpoint is scored both ways.

## Tests

```bash
# needs OLMo-core with nn.vision and nn.moe.v2.hf (#942); skipped otherwise, e.g. in CI
pytest tests/test_olmo35_mm_export.py tests/test_olmo35_mm_vllm_plugin.py
```

`test_olmo35_mm_export.py` is scaling-ladders' `test_multimodal_conversion.py` (tiny multimodal checkpoint ->
whole-model export == text export of its LM, wrapper loads with and without remote code, every parameter kind loads)
plus: the chat stop list in both exports, and `convert_moe_checkpoint_to_hf.py`'s language-model path from an OLMoDDP
DCP, an OLMo-core `model.*` DCP (TPU-converted layout) and a consolidated export, each byte-identical to the text
export. `test_olmo35_mm_vllm_plugin.py` is scaling-ladders' `test_vllm_multimodal.py`; its last test needs vLLM too.

## Pins

- OLMo-core: allenai/OLMo-core@7074cf408a157b71c5c74fbc749c4a757e047367 (#942 head, with #964) for `export_multimodal_olmoddp.py`,
  the multimodal model and the HF text model code.
- Plugins: allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 (copied).
- Export/eval scripts: H271's `/root/mmsft-eval/{export,tpu}` (olmo-post-training-ledger `experiments/mmsft/tpu/`),
  generalized; each file says what it came from.
