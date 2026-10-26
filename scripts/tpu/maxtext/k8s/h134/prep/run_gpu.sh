#!/bin/bash
# H134 P2: HF step-1 CE of the 14T base (olmo35-small-14t600b-mtlc-emo, converted + exported by h134/convert) on H122's
# c11-s0 and c11-s1 step-1 batches. jupiter H100 (cu128), as H123's refce. Ids, counts and losses only.
set -euo pipefail
R=/weka/oe-eval-default/abhishekr/watch/tpu/h134/refce; B=/weka/oe-eval-default/abhishekr/watch/tpu/h122/p5
HF=/weka/olmo-3p5-post-checkpoints/abhishekr/tpu-posttrain/hf/olmo35-small-14t600b-mtlc-emo-lc5961-base-h134
nvidia-smi --query-gpu=name,memory.total --format=csv
sha256sum $R/refce_h134.py; cut -c1-16 $HF/model.safetensors.sha256
command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl >/dev/null)
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH=$HOME/.local/bin:$PATH
for s in 0 1; do
  [ -f $R/refce_c11-s$s.json ] && continue
  uv run --python 3.12 --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match --with torch==2.9.1 --with transformers==4.57.6 --with numpy --with accelerate --with flash-linear-attention \
    python $R/refce_h134.py $B/s$s $HF $R/refce_c11-s$s.json
done
