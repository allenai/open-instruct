#!/bin/bash
# Export a MaxText Olmo 3 checkpoint to HF format for evals (allenai/open-instruct#1933).
#   export_hf.sh <maxtext params dir, e.g. gs://.../checkpoints/<step>/items> <out dir> <base HF model> <tokenizer dir> [generation-config model]
# MaxText's tensors are exact (bit-identical round trip), but its hard-coded Olmo 3 config.json is not:
# eos_token_id falls back to 50279, max_position_embeddings is 8192, and YaRN is written only as
# transformers-5 rope_parameters. So config.json and generation_config.json come from the base model,
# and the tokenizer (with the chat template used in training) from the training tokenizer.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARAMS=${1:?params dir}; OUT=${2:?out dir}
BASE=${3:?base HF model whose config.json describes the architecture}
# generation_config.json decides the stop tokens at serving time. A base model's config stops only at <|endoftext|>,
# so a chat model exported with it runs past <|im_end|>; pass the matching released chat model (e.g.
# allenai/Olmo-3-7B-Instruct-SFT) as GENCFG for SFT exports.
GENCFG=${5:-$BASE}
TOK=${4:?tokenizer dir the run trained with (a PreTrainedTokenizerFast save of allenai/olmo-3-tokenizer-instruct-dev; see README)}
PY=${PY:-python3}
PY=$("$PY" -c 'import sys; print(sys.executable)')
: "${MAXTEXT_DIR:?set MAXTEXT_DIR to the checkout pinned in image/maxtext_source}"
# Resolve local arguments before changing to the MaxText checkout. GCS parameter paths remain URIs.
PARAMS=$("$PY" -c 'import os, sys; p = sys.argv[1]; print(p if "://" in p else os.path.abspath(p))' "$PARAMS")
OUT=$("$PY" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$OUT")
TOK=$(cd "$TOK" && pwd)
[[ ! -d "$BASE" ]] || BASE=$(cd "$BASE" && pwd)
[[ ! -d "$GENCFG" ]] || GENCFG=$(cd "$GENCFG" && pwd)
"$PY" - "$OUT" <<'PYEOF'
import os
import sys

out = sys.argv[1]
if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
    raise SystemExit("out must be a new or empty directory; do not mix exports")
PYEOF
export JAX_PLATFORMS=cpu
cd "$MAXTEXT_DIR"
# Runs trained with tensor parallelism use the vocabulary padded to 100352 (resize_vocab.py); trim it back.
vocab=$("$PY" - "$PARAMS" <<'PYEOF'
import sys

from orbax import checkpoint as ocp
m = ocp.PyTreeCheckpointer().metadata(sys.argv[1])
m = m.item_metadata.tree if hasattr(m, "item_metadata") else m.tree
p = m["params"]["params"] if "params" in m else m  # to_maxtext layout, or Tunix's model_params
e = p["token_embedder"]["embedding"]
e = e["value"] if isinstance(e, dict) else e  # Tunix wraps every leaf as {"value": ...}
print(e.shape[0])
PYEOF
)
if [[ "$vocab" != "100278" ]]; then
  trimmed=$(mktemp -d "${TMPDIR:-/tmp}/maxtext-trim.XXXXXX")
  trap 'rm -rf -- "$trimmed"' EXIT
  "$PY" "$SCRIPT_DIR/resize_vocab.py" "$PARAMS" "$trimmed" 100278
  PARAMS="$trimmed/0/items"
fi
"$PY" -m maxtext.checkpoint_conversion.to_huggingface src/maxtext/configs/base.yml \
  model_name=olmo3-7b load_parameters_path="$PARAMS" base_output_directory="$OUT" scan_layers=True \
  use_multimodal=false hardware=cpu skip_jax_distributed_system=true weight_dtype=bfloat16 --hf_model_path="$BASE"
"$PY" - "$OUT" "$BASE" "$TOK" "$GENCFG" <<'PYEOF'
import json
import os
import shutil
import sys

from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

out, base, tok, gencfg = sys.argv[1:]


def model_file(model, name):
    return os.path.join(model, name) if os.path.isdir(model) else hf_hub_download(model, name)


PreTrainedTokenizerFast.from_pretrained(tok).save_pretrained(out)
shutil.copy(model_file(base, "config.json"), os.path.join(out, "config.json"))
shutil.copy(model_file(gencfg, "generation_config.json"), os.path.join(out, "generation_config.json"))
with open(os.path.join(out, "config.json")) as stream:
    c = json.load(stream)
assert c["rope_scaling"]["rope_type"] == "yarn" and c["eos_token_id"] == 100257, c
with open(os.path.join(out, "tokenizer_config.json")) as stream:
    tokenizer_config = json.load(stream)
assert tokenizer_config.get("chat_template") or os.path.exists(os.path.join(out, "chat_template.jinja")) or os.path.isdir(os.path.join(out, "chat_templates"))
print("export ok:", out)
PYEOF
