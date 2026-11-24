#!/bin/bash
# stage_items.sh GS_ITEMS LOCAL_ITEMS: copy a MaxText Orbax items dir (zarr v2, non-OCDBT) to local disk without the
# optimizer state's data: every entry except opt_state.* in full (gcloud storage cp, crc32c-checked), and only the
# .zarray metadata of the opt_state.* arrays (orbax's metadata() opens every leaf; the params subtree is then restored
# with partial_restore). Restoring straight from gs:// through tensorstore stalls at <1 MB/s on this box; this runs at
# ~200-400 MB/s (~4 min for the 45 GB of params).
# From allenai/olmo-post-training-ledger@ca639ab0b01ed7cff57a411d73936ecc9f0d8914 experiments/mmsft/tpu/stage_items.sh,
# with the Python (any with gcsfs) taken from PY instead of a fixed venv. Needs a gcloud login with read access.
set -euo pipefail
PY=${PY:-python3}
SRC=${1%/}; DST=${2%/}
[[ -e $DST/.STAGED_OK ]] && { echo "staged already: $DST"; exit 0; }
mkdir -p "$DST"
gcloud storage ls $SRC/ | sed "s#^$SRC/##; s#/\$##" > $DST.all
grep -vE '^opt_state\.' $DST.all > $DST.full
grep -E '^opt_state\.' $DST.all > $DST.meta || true
echo "staging $(wc -l < $DST.full) entries in full + .zarray of $(wc -l < $DST.meta) opt_state arrays: $SRC -> $DST"
sed "s#^#$SRC/#" $DST.full | gcloud storage cp -r -I $DST/ 2>&1 | tail -1
"$PY" - "$SRC" "$DST" "$DST.meta" <<'PY'
import os, sys
from concurrent.futures import ThreadPoolExecutor
import gcsfs
src, dst, lst = sys.argv[1:]
fs = gcsfs.GCSFileSystem()
names = [l.strip() for l in open(lst) if l.strip()]
def get(n):
    os.makedirs(os.path.join(dst, n), exist_ok=True)
    for f in (".zarray", ".zattrs"):
        if f == ".zarray" or fs.exists(f"{src}/{n}/{f}"):
            fs.get(f"{src}/{n}/{f}", os.path.join(dst, n, f))
with ThreadPoolExecutor(64) as ex:
    list(ex.map(get, names))
print(f"fetched metadata of {len(names)} opt_state arrays")
PY
n=$(ls -A $DST | wc -l); [[ $n == $(wc -l < $DST.all) ]] || { echo "staged $n entries, expected $(wc -l < $DST.all)"; exit 1; }
touch $DST/.STAGED_OK; du -sh $DST
