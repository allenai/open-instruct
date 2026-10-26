#!/bin/bash
# H134 P2: HF step-1 CE of the 14T base (olmo35-small-14t600b-mtlc-emo, converted + exported by h134/convert) on H122's
# c11-s0 and c11-s1 step-1 batches. jupiter H100: cu128 (as H123's refce); holmes B300: cu130 with a CUDA 13 ptxas for
# triton (as H103's run_refce4b_holmes3.sh). Ids, counts and losses only.
set -euo pipefail
R=/weka/oe-eval-default/abhishekr/watch/tpu/h134/refce; B=/weka/oe-eval-default/abhishekr/watch/tpu/h122/p5
HF=/weka/olmo-3p5-post-checkpoints/abhishekr/tpu-posttrain/hf/olmo35-small-14t600b-mtlc-emo-lc5961-base-h134
G=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU $G"
sha256sum $R/refce_h134.py; cut -c1-16 $HF/model.safetensors.sha256
command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl >/dev/null)
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH=$HOME/.local/bin:$PATH
IDX=https://download.pytorch.org/whl/cu128
if echo "$G" | grep -q B300; then
  IDX=https://download.pytorch.org/whl/cu130; P=""
  for c in /usr/local/cuda/bin/ptxas /usr/local/cuda-13*/bin/ptxas; do [ -x "$c" ] && "$c" --help 2>&1 | grep -q sm_103a && { P=$c; break; }; done
  if [ -z "$P" ]; then uv venv -q --python 3.12 /tmp/nvcc13 && uv pip install -q --python /tmp/nvcc13/bin/python "nvidia-cuda-nvcc==13.0.*"; P=$(find /tmp/nvcc13 -name ptxas -type f | head -1); chmod +x "$P"; fi
  echo "PTXAS $P"; export TRITON_PTXAS_PATH=$P
fi
for s in 0 1; do
  [ -f $R/refce_c11-s$s.json ] && continue
  uv run --python 3.12 --index $IDX --index-strategy unsafe-best-match --with torch==2.9.1 --with transformers==4.57.6 --with numpy --with accelerate --with flash-linear-attention \
    python $R/refce_h134.py $B/s$s $HF $R/refce_c11-s$s.json.tmp && mv $R/refce_c11-s$s.json.tmp $R/refce_c11-s$s.json
done
