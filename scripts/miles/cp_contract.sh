#!/bin/bash
# Bounded two-GPU Core/MILES gradient-equivalence check; no serving or downloads.
set -euo pipefail
cd /opt/core-rl
root=${1:-/output/cp-contract}
torchrun --standalone --nproc-per-node=2 tests/miles/cp_loss_contract.py
if [[ ! -f "$root/routes.pt" ]]; then
    python tests/miles/ep_contract.py bootstrap "$root" --small-features
    # Capture fixed replay routes with the existing single-rank reference fixture.
    torchrun --standalone --nproc-per-node=1 tests/miles/ep_contract.py run "$root" --mode policy
fi
for averaging in token response; do
    flags=()
    if [[ "$averaging" == token ]]; then flags+=(--token-average); fi
    for cp in 2 1; do
        torchrun --standalone --nproc-per-node=2 tests/miles/cp_contract.py run "$root" --cp "$cp" "${flags[@]}"
        torchrun --standalone --nproc-per-node=2 tests/miles/cp_contract.py run "$root" --cp "$cp" --checkpointing "${flags[@]}"
    done
done
python tests/miles/cp_contract.py compare "$root"
