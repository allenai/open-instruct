#!/usr/bin/env bash
# Copy the weka-hosted uv venv + interpreter to node-local disk, then run the given entry point with it.
# Usage (inside a Beaker job whose cwd is the open-instruct checkout): bash localvenv_run.sh open_instruct/dpo.py ARGS...
# Why: importing torch from the weka venv stalls on Jupiter (page-wait reads); plain copies do not. 2026-09-19.
set -euo pipefail
SRC_VENV=/weka/oe-adapt-default/mingqianz/open-instruct/.venv
SRC_PY=/weka/oe-adapt-default/mingqianz/uv/python/cpython-3.12.13-linux-x86_64-gnu
DST_VENV=/tmp/oi-venv
DST_PY=/tmp/uv-py
t0=$(date +%s)
mkdir -p "$DST_VENV" "$DST_PY"
cp -a "$SRC_PY/." "$DST_PY/"
cp -a "$SRC_VENV/." "$DST_VENV/"
sed -i "s|^home = .*|home = $DST_PY/bin|" "$DST_VENV/pyvenv.cfg"
ln -sfn "$DST_PY/bin/python3.12" "$DST_VENV/bin/python"
ln -sfn python "$DST_VENV/bin/python3"
ln -sfn python "$DST_VENV/bin/python3.12"
echo "LOCAL_VENV_READY in $(( $(date +%s) - t0 ))s: $("$DST_VENV/bin/python" -c 'import sys; print(sys.version.split()[0], sys.executable)')"
exec "$DST_VENV/bin/python" -m torch.distributed.run --nproc_per_node=8 "$@"
