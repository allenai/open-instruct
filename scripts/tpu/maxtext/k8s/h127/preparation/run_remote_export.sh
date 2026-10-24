#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 JAX_PLATFORMS=cpu
uv --no-config pip install --quiet --target /tmp/h127-export-deps 'jax[cpu]==0.11.1' 'orbax-checkpoint==0.12.4' 'numpy==2.2.6' 'safetensors==0.7.0' 'google-crc32c==1.8.0' 'requests==2.32.5'
export PYTHONPATH=/tmp/h127-export-deps:/export-code
ROOT=/weka/olmo-3p5-post-checkpoints/robertb/presentation14t-v1
PARAMS="$ROOT/staging/smoke-r2-step50/model_params"
EXPORT="$ROOT/exports/smoke-r2-step50"
python /transfer/download_checkpoint.py --manifest /transfer/source-manifest.json --credential /secrets/h127-gcs-read.json --destination "$PARAMS"
if [ -e "$EXPORT" ]; then
  echo 'Export destination already exists; refusing to overwrite' >&2
  exit 1
fi
python /export-code/export_bounded.py "$PARAMS" /export-code/hf-reference "$EXPORT" /export-code/atomic-tokenizer
mkdir -p /output
python - <<'PY'
import hashlib
import importlib.metadata
import json
from pathlib import Path
from export_bounded import tensor_hashes
root=Path('/weka/olmo-3p5-post-checkpoints/robertb/presentation14t-v1')
export=root/'exports/smoke-r2-step50'
expected=json.loads(Path('/transfer/expected-tensor-hashes.json').read_text())
actual=tensor_hashes(export/'model.safetensors')
assert len(actual)==23441 and actual==expected, 'Remote CPU export differs from locally validated trained checkpoint'
for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
 assert (export/name).read_bytes()==(Path('/export-code/atomic-tokenizer')/name).read_bytes(), name
receipt=json.loads((export/'export-receipt.json').read_text())
receipt.update(all_23441_tensors_match_local_trained_export=True, registered_tokenizer_identical=True, source_gcs_model_params=json.loads(Path('/transfer/source-manifest.json').read_text())['source_prefix'], packages={k:importlib.metadata.version(k) for k in ('jax','orbax-checkpoint','numpy','torch','safetensors')})
receipt['full_file_matches_local_export']=receipt['model_safetensors_sha256']=='4afba72559271c355506170c0904bbd740d16a4488e12ceadbb950bdbd24ed49'
(export/'remote-export-qualified.json').write_text(json.dumps(receipt,indent=2))
Path('/output/remote-export-qualified.json').write_text(json.dumps(receipt,indent=2))
Path('/output/download-receipt.json').write_bytes((root/'staging/smoke-r2-step50/model_params-download-receipt.json').read_bytes())
print('REMOTE-EXPORT-QUALIFIED',json.dumps(receipt),flush=True)
PY
