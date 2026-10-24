#!/usr/bin/env bash
# The transfer bundle must contain a finalized, generation-pinned source manifest.
set -euo pipefail
ARM="${1:?expected control or uniform}"
STEP="${2:?expected 512 or 2048}"
case "$ARM" in control|uniform) ;; *) exit 2 ;; esac
case "$STEP" in 512|2048) ;; *) exit 2 ;; esac
export H127_EXPORT_ARM="$ARM" H127_EXPORT_STEP="$STEP"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 JAX_PLATFORMS=cpu
uv --no-config pip install --quiet --target /tmp/h127-export-deps 'jax[cpu]==0.11.1' 'orbax-checkpoint==0.12.4' 'numpy==2.2.6' 'safetensors==0.7.0' 'google-crc32c==1.8.0' 'requests==2.32.5'
export PYTHONPATH=/tmp/h127-export-deps:/export-code
ROOT=/weka/olmo-3p5-post-checkpoints/robertb/presentation14t-v1
NAME="$ARM-screen-step$STEP"
PARAMS="$ROOT/staging/$NAME/model_params"
EXPORT="$ROOT/exports/$NAME"
export H127_EXPORT_PATH="$EXPORT" H127_PARAMS_PATH="$PARAMS"
python - <<'PY'
import json
import os
from pathlib import Path
manifest = json.loads(Path('/transfer/source-manifest.json').read_text())
expected = ('gs://ai2-olmo-dev/post-training/runs/robertb/presentation14t-v1/'
            f"robertb-pres14t-v1-{os.environ['H127_EXPORT_ARM']}-screen/checkpoints/"
            f"{os.environ['H127_EXPORT_STEP']}/model_params/")
assert manifest['source_prefix'] == expected
assert manifest['checkpoint_finalized'] is True
assert manifest['step'] == int(os.environ['H127_EXPORT_STEP'])
assert not Path(os.environ['H127_EXPORT_PATH']).exists(), 'Refusing to overwrite an existing export'
PY
python /transfer/download_checkpoint.py --manifest /transfer/source-manifest.json --credential /secrets/h127-gcs-read.json --destination "$PARAMS"
python /export-code/export_bounded.py "$PARAMS" /export-code/hf-reference "$EXPORT" /export-code/tokenizer
mkdir -p /output
python - <<'PY'
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
export = Path(os.environ['H127_EXPORT_PATH'])
params = Path(os.environ['H127_PARAMS_PATH'])
receipt = json.loads((export / 'export-receipt.json').read_text())
assert receipt['tensor_count'] == 23441 and receipt['all_bf16_values_finite']
assert receipt['source_parameters'] == str(params)
for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
    assert (export / name).read_bytes() == (Path('/export-code/tokenizer') / name).read_bytes(), name
source = Path('/transfer/source-manifest.json').read_bytes()
receipt.update(
    arm=os.environ['H127_EXPORT_ARM'], step=int(os.environ['H127_EXPORT_STEP']),
    registered_tokenizer_identical=True,
    source_gcs_model_params=json.loads(source)['source_prefix'],
    source_manifest_sha256=hashlib.sha256(source).hexdigest(),
    packages={k: importlib.metadata.version(k) for k in ('jax', 'orbax-checkpoint', 'numpy', 'torch', 'safetensors')},
    qualification_note='Uses the converter qualified on base and trained-step50 roundtrips. These trained weights are not expected to match the step50 tensor hashes.',
)
for directory in (export, Path('/output')):
    (directory / 'remote-export-qualified.json').write_text(json.dumps(receipt, indent=2))
    (directory / 'source-manifest.json').write_bytes(source)
Path('/output/download-receipt.json').write_bytes(params.with_name('model_params-download-receipt.json').read_bytes())
print('SCIENTIFIC-EXPORT-QUALIFIED', json.dumps(receipt), flush=True)
PY
