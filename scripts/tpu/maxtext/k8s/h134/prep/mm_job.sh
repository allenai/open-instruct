#!/bin/bash
# H134 B1/B2 Beaker job (tpu-worker fork, 2026-10-26; approved by ledger-manager via tpu-worker). In order:
#  A. read-only checks on B1 and B2: format, keys vs B0, vocab, decay state, sha256 of every file read; stops before
#     any copy if a plan fails or the LR at the saved step is not ~0;
#  B. verified base copies to /weka/olmo-3p5-post-checkpoints/bases/<registry name>/step<N>/ with RECEIPT.json
#     (sha256 per file, source and copy compared); no-clobber, deletes nothing;
#  C. the vision strip (convert_mm.sh stages 0-1b) on each verified copy, into new dirs.
# Results are JSON on WEKA under $R. Prints ids, shapes, counts and hashes only.
set -uo pipefail
J=/weka/oe-eval-default/abhishekr/watch/tpu/h134/mmstrip
CODE=$J/code; R=$J/job-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $R
STAGE=/weka/olmo-3p5-post-checkpoints/abhishekr/tpu-posttrain/maxtext-stage
BASES=/weka/olmo-3p5-post-checkpoints/bases
MID=/weka/olmo-3p5-mid-checkpoints/jasonr/olmo35-multimodal-midtraining/checkpoints
declare -A SRC=([olmo35-small-mm-ckpt1-64k]=$MID/olmoe3-810m-cx2-mm-merged-mt-lc-v6-lm1e4/step20000
                [olmo35-small-mm-ckpt2-256k-amt]=$MID/olmoe3-810m-cx2-mm-hero-mt-lc-256k-lm1e4/step22240)
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/job.log; }
export REF=$J/b0_lm_manifest.json REFCFG=$J/b0_config.json OI=$CODE
dcpdir() { if [ -e "$1/olmo-core/model_and_optim/.metadata" ]; then echo "$1/olmo-core"; elif [ -e "$1/model_and_optim/.metadata" ]; then echo "$1"; else return 1; fi; }
log "start; results $R"
# venv: torch (cpu) + numpy, and OLMo-core at open-instruct's pinned rev so DCP metadata unpickles as on Hammond
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > $R/uv_install.log 2>&1
uv venv -q /tmp/mmv --python 3.12 && uv pip install -q --python /tmp/mmv/bin/python --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0+cpu" \
  && uv pip install -q --python /tmp/mmv/bin/python "numpy==2.3.5" "ai2-olmo-core @ git+https://github.com/allenai/OLMo-core.git@fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d" > $R/venv.log 2>&1 \
  || { log "VENV-FAIL"; exit 9; }
export PY=/tmp/mmv/bin/python
sha256sum $CODE/*.py $CODE/*.sh $REF $REFCFG > $R/code.sha256
fail=0
for name in "${!SRC[@]}"; do
  root=${SRC[$name]}; W=$R/$name; mkdir -p $W
  src=$(dcpdir $root) || { log "[$name] FORMAT: no model_and_optim/.metadata under $root (not an OLMo-core DCP?)"; ls "$root" > $W/ls.txt 2>&1; fail=1; continue; }
  (cd $root && find . -type f -printf '%P\t%s\n' | sort) > $W/files.tsv
  log "[$name] src $root (DCP $src): $(wc -l < $W/files.tsv) files, $(awk -F'\t' '{s+=$2} END {printf "%d", s}' $W/files.tsv) bytes"
  (cd $src && sha256sum config.json model_and_optim/.metadata $( [ -e train/rank0.pt ] && echo train/rank0.pt )) > $W/read_receipt.sha256
  STOP_AFTER_PLAN=1 bash $CODE/convert_mm.sh $name $src $W $STAGE "${LMPREFIX:-}" >> $W/stage0.out 2>&1 || { log "[$name] STAGE0-FAIL: $(grep -h STRIP-FAIL $W/convert_mm.log | tail -1)"; fail=1; }
  $PY - "$src" > $W/decay.json 2> $W/decay.err <<'PY'
import json, os, sys
import torch
src = sys.argv[1]
cfg = json.load(open(os.path.join(src, "config.json")))
tm = cfg.get("train_module", {}); s = tm.get("scheduler") or {}; lr = (tm.get("optim") or {}).get("lr")
dur = (cfg.get("trainer") or {}).get("max_duration") or {}
out = {"scheduler": s, "lr_peak": lr, "max_duration": dur,
       "global_batch_size": (cfg.get("data_loader") or {}).get("global_batch_size"),
       "optim_group_overrides": (tm.get("optim") or {}).get("group_overrides")}
step = None
for f in ("train/rank0.pt",):
    p = os.path.join(src, f)
    if os.path.exists(p):
        st = torch.load(p, map_location="cpu", weights_only=False)
        out["train_state_keys"] = sorted(st) if isinstance(st, dict) else str(type(st))
        if isinstance(st, dict):
            for k in ("global_step", "step"):
                if k in st: step = int(st[k])
            out["global_train_tokens_seen"] = st.get("global_train_tokens_seen")
out["step"] = step
try:
    T = dur["value"] if dur.get("unit") == "steps" else None
    if T is None and dur.get("unit") == "tokens" and out["global_batch_size"]:
        T = dur["value"] // out["global_batch_size"]
    w, af, wmin = s.get("warmup", 0), s.get("alpha_f", 0.0), s.get("warmup_min_lr", 0.0)
    if "LinearWithWarmup" in s.get("_CLASS_", "") and step is not None and T:
        cur = wmin + (lr - wmin) * step / w if step < w else lr - (lr - lr * af) * (step - w) / max(T - w, 1)
        out.update(t_max_steps=T, lr_at_step=cur, lr_frac_of_peak=cur / lr, decayed=cur / lr <= 0.01)
except Exception as e:
    out["lr_compute_error"] = repr(e)
print(json.dumps(out, indent=1, default=str))
PY
  d=$($PY -c "import json;d=json.load(open('$W/decay.json'));print(d.get('step'), d.get('t_max_steps'), d.get('lr_at_step'), d.get('lr_frac_of_peak'), d.get('decayed'))" 2>/dev/null)
  log "[$name] decay: step t_max lr_at_step frac decayed = ${d:-<unreadable, see decay.err>}"
  case "$d" in *False) log "[$name] DECAY-MISMATCH: LR not ~0 at the saved step"; fail=1;; esac
done
[ $fail = 0 ] || { log "PHASE-A-FAIL: stopping before any copy"; exit 3; }
log "PHASE-A-OK"
[ "${STOP_AFTER_CHECKS:-}" = 1 ] && exit 0
for name in "${!SRC[@]}"; do
  src=${SRC[$name]}; step=$(basename $src); D=$BASES/$name/$step; W=$R/$name
  [ -e $D ] && { log "[$name] $D exists; not overwriting"; fail=1; continue; }
  mkdir -p $D.tmp
  log "[$name] copy start -> $D"
  (cd $src && find . -type f -printf '%P\n' | sort) | xargs -P 16 -I{} bash -c '
    s=$(sha256sum "$1/$2" | cut -d" " -f1); mkdir -p "$(dirname "$3/$2")"; cp -n "$1/$2" "$3/$2" || exit 1
    d=$(sha256sum "$3/$2" | cut -d" " -f1); printf "%s\t%s\t%s\n" "$2" "$s" "$d"' _ $src $D.tmp {} > $W/copy.tsv 2> $W/copy.err
  n_bad=$(awk -F'\t' '$2!=$3' $W/copy.tsv | wc -l); n=$(wc -l < $W/copy.tsv); n_src=$(wc -l < $W/files.tsv)
  if [ "$n_bad" = 0 ] && [ "$n" = "$n_src" ]; then
    $PY - $W/copy.tsv $W/files.tsv $src $(dcpdir $src) $D.tmp/RECEIPT.json <<'PY'
import json, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
sizes = dict(l.rstrip("\n").split("\t") for l in open(sys.argv[2]))
json.dump({"source": sys.argv[3], "source_dcp_dir": sys.argv[4], "n_files": len(rows),
           "bytes": sum(int(sizes[r[0]]) for r in rows),
           "files": {r[0]: {"sha256": r[1], "bytes": int(sizes[r[0]])} for r in sorted(rows)}}, open(sys.argv[5], "w"), indent=0)
PY
    mv $D.tmp $D && log "[$name] COPY-OK $D: $n files verified, RECEIPT.json sha $(sha256sum $D/RECEIPT.json | cut -c1-16)"
  else log "[$name] COPY-FAIL: $n/$n_src files, $n_bad sha mismatches; partial copy kept at $D.tmp"; fail=1; fi
done
[ $fail = 0 ] || { log "PHASE-B-FAIL: no strip"; exit 4; }
for name in "${!SRC[@]}"; do
  step=$(basename ${SRC[$name]}); D=$BASES/$name/$step; W=$R/$name
  log "[$name] strip start on $D"
  bash $CODE/convert_mm.sh $name $(dcpdir $D) $W/strip $STAGE "${LMPREFIX:-}" >> $W/strip.out 2>&1 && log "[$name] $(grep -h APPLY-OK $W/strip/convert_mm.log | tail -1)" \
    || { log "[$name] STRIP-FAIL: $(grep -h STRIP-FAIL $W/strip/convert_mm.log | tail -1)"; fail=1; }
done
log "JOB-DONE fail=$fail"; exit $fail
