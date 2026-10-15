#!/bin/bash
set -euo pipefail
export PATH="/stage/.venv/bin:$PATH"
mkdir -p /output
python - <<'PY'
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import torch

assert torch.cuda.is_available() and torch.cuda.device_count() >= 4
for path, expected in json.loads(os.environ['EXPECTED_SOURCE_HASHES']).items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
versions = {name: importlib.metadata.version(name) for name in
            ['torch', 'deepspeed', 'accelerate', 'transformers', 'liger-kernel']}
provenance = dict(commit=os.environ['GIT_COMMIT'], versions=versions,
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  hashes=json.loads(os.environ['EXPECTED_SOURCE_HASHES']))
Path('/output/provenance.json').write_text(json.dumps(provenance, indent=2))
print('SFT_PARITY_PROVENANCE ' + json.dumps(provenance), flush=True)
PY

run_case() {
    local name="$1"
    shift
    torchrun --standalone --nnodes=1 --nproc-per-node=4 scripts/test/sft_loss_parity.py \
        --output "/output/$name" "$@" 2>&1 | tee "/output/$name.log"
}

# Pin sensitivity to unequal token weighting, rather than a startup failure.
if run_case legacy-control --legacy-control; then
    echo 'ERROR: the old mean-of-microbatch-means control unexpectedly passed'
    exit 1
fi
grep -q 'PARITY_MISMATCH gradient-before-clipping' /output/legacy-control.log
echo 'SFT_PARITY_NEGATIVE_CONTROL_REJECTED'
if run_case zero2-bf16-scale-control --zero-stage 2 --precision bf16 --loss-scale 2; then
    echo 'ERROR: doubled BF16 gradients unexpectedly passed'
    exit 1
fi
grep -q 'PARITY_MISMATCH gradient-before-clipping' /output/zero2-bf16-scale-control.log
echo 'SFT_PARITY_BF16_SCALE_CONTROL_REJECTED'

run_case ddp
run_case ddp-sync --sync-each-batch
run_case zero2 --zero-stage 2
run_case zero3 --zero-stage 3
run_case zero2-bf16 --zero-stage 2 --precision bf16
run_case zero3-bf16 --zero-stage 3 --precision bf16
run_case zero2-sp2 --zero-stage 2 --sp-size 2 --precision bf16
run_case zero3-sp2 --zero-stage 3 --sp-size 2 --precision bf16
run_case zero2-liger --zero-stage 2 --precision bf16 --liger
run_case zero3-liger --zero-stage 3 --precision bf16 --liger
run_case zero2-sp2-liger --zero-stage 2 --sp-size 2 --precision bf16 --liger
run_case zero3-sp2-liger --zero-stage 3 --sp-size 2 --precision bf16 --liger
# Reload into a fresh engine/process group; the saved offset is inside an SP refill.
run_case zero2-resume --zero-stage 2 --precision bf16 --phase save
run_case zero2-resume --zero-stage 2 --precision bf16 --phase resume
run_case zero3-resume --zero-stage 3 --precision bf16 --phase save
run_case zero3-resume --zero-stage 3 --precision bf16 --phase resume
run_case zero2-sp2-resume --zero-stage 2 --sp-size 2 --precision bf16 --phase save
run_case zero2-sp2-resume --zero-stage 2 --sp-size 2 --precision bf16 --phase resume
run_case zero3-sp2-resume --zero-stage 3 --sp-size 2 --precision bf16 --phase save
run_case zero3-sp2-resume --zero-stage 3 --sp-size 2 --precision bf16 --phase resume
run_case zero3-sp2-liger-resume --zero-stage 3 --sp-size 2 --precision bf16 --liger --phase save
run_case zero3-sp2-liger-resume --zero-stage 3 --sp-size 2 --precision bf16 --liger --phase resume
echo 'SFT_LOSS_GPU_PARITY_PASS: 22 positive cases and two rejected negative controls'
