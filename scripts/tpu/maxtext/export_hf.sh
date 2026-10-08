#!/bin/bash
# Export a MaxText Olmo 3 checkpoint to HF format for evals (allenai/open-instruct#1933).
#   export_hf.sh <maxtext params dir, e.g. gs://.../checkpoints/<step>/items> <out dir> <base HF model> <tokenizer dir> [generation-config model]
# MaxText's tensors are exact (bit-identical round trip), but its hard-coded Olmo 3 config.json is not:
# eos_token_id falls back to 50279, max_position_embeddings is 8192, and YaRN is written only as
# transformers-5 rope_parameters. So config.json and generation_config.json come from the base model,
# and the tokenizer (with the chat template used in training) from the training tokenizer.
set -euo pipefail
PARAMS=${1:?params dir}; OUT=${2:?out dir}
BASE=${3:?base HF model whose config.json describes the architecture}
# generation_config.json decides the stop tokens at serving time. A base model's config stops only at <|endoftext|>,
# so a chat model exported with it runs past <|im_end|>; pass the matching released chat model (e.g.
# allenai/Olmo-3-7B-Instruct-SFT) as GENCFG for SFT exports.
GENCFG=${5:-$BASE}
TOK=${4:?tokenizer dir the run trained with (a PreTrainedTokenizerFast save of allenai/olmo-3-tokenizer-instruct-dev; see README)}
PY=${PY:-/opt/venvs/maxtext/bin/python}
cd "${MAXTEXT_DIR:-$HOME/repos/maxtext}" # a MaxText checkout at the image's base commit with image/patches/maxtext.patch applied
# Runs trained with tensor parallelism use the vocabulary padded to 100352 (resize_vocab.py); trim it back.
vocab=$($PY - "$PARAMS" <<'PYEOF'
import sys, orbax.checkpoint as ocp
m = ocp.PyTreeCheckpointer().metadata(sys.argv[1])
m = m.item_metadata.tree if hasattr(m, "item_metadata") else m.tree
p = m["params"]["params"] if "params" in m else m  # to_maxtext layout, or Tunix's model_params
e = p["token_embedder"]["embedding"]
e = e["value"] if isinstance(e, dict) else e  # Tunix wraps every leaf as {"value": ...}
print(e.shape[0])
PYEOF
)
if [[ "$vocab" != "100278" ]]; then
  trimmed=$(mktemp -d /opt/scratch/trim.XXXX)
  JAX_PLATFORMS=cpu $PY "$(dirname "$0")/resize_vocab.py" "$PARAMS" "$trimmed" 100278
  PARAMS="$trimmed/0/items"
fi
JAX_PLATFORMS=cpu $PY -m maxtext.checkpoint_conversion.to_huggingface src/maxtext/configs/base.yml \
  model_name=olmo3-7b load_parameters_path="$PARAMS" base_output_directory="$OUT" scan_layers=True \
  use_multimodal=false hardware=cpu skip_jax_distributed_system=true weight_dtype=bfloat16 --hf_model_path="$BASE"
$PY - "$OUT" "$BASE" "$TOK" "$GENCFG" <<'PYEOF'
import json, os, shutil, sys
from huggingface_hub import hf_hub_download
out, base, tok, gencfg = sys.argv[1:]
shutil.copy(hf_hub_download(base, "config.json"), os.path.join(out, "config.json"))
shutil.copy(hf_hub_download(gencfg, "generation_config.json"), os.path.join(out, "generation_config.json"))
for f in os.listdir(tok):
    shutil.copy(os.path.join(tok, f), os.path.join(out, f))
c = json.load(open(os.path.join(out, "config.json")))
assert c["rope_scaling"]["rope_type"] == "yarn" and c["eos_token_id"] == 100257, c
assert "chat_template" in open(os.path.join(out, "tokenizer_config.json")).read() or os.path.exists(os.path.join(out, "chat_template.jinja"))
print("export ok:", out)
PYEOF
