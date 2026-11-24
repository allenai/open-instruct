#!/bin/bash
# run_eval.sh LABEL STEP_DIR — export and evaluate one Olmo 3.5 multimodal SFT checkpoint on Beaker.
#
# STEP_DIR is an OLMo-core DCP step dir (model_and_optim/ + config.json), as Beaker jobs see it: a GPU run's save
# (e.g. <run>/step10962) or a TPU (MaxText) checkpoint converted to OLMo-core first. LABEL names everything
# downstream (<run>-s<step>; for a TPU run use the OLMo-core-equivalent step, MaxText checkpoint dir + 1). Stages:
#
#   export   [Beaker, 1 GPU] scripts/export/olmo35_mm/export_all_job.sh: verified olmo-core-export (vision evals),
#            then export/hf_text (text evals; generation_config eos [100265, 100257]) + smoke  -> $OUT
#   sanity   [Beaker, 1 GPU] sanity_job.sh: COCO image + 3 questions through olmo_core_vlm (next to SANITY_REF)
#   vision   [Beaker]        olmo-eval scripts/olmo35_mmsft/eval_vision.sh x SUITES:
#                              docs24 = ChartQA DocVQA TextVQA InfoVQA (max_crops=24)
#                              core   = VQAv2 AI2D CountBenchQA RealWorldQA
#                              list   = MMMU MuirBench MMIU BLINK MMMU-Pro          (500 items each, 2 GPUs)
#   emocheck [Beaker, 1 GPU] scripts/export/olmo35_mm/emo_check_job.sh (EMo HF-vs-native logits; SKIP without EMo)
#   text     [Beaker]        launch_text_cells.py: the recorded H182 text cells on hf_text. TEXT_CELLS=full is the
#                            battery below (BFCL on BFCL_CLUSTER/BFCL_WORKSPACE, as the GPU arms ran it); or a list
#                            of cells (e.g. "ruler64k ifeval" for an intermediate checkpoint); none = skip
#
# Resumable: job ids go to $STATE/<stage>; a re-run skips recorded stages. Nothing is deleted or overwritten.
# Every launched job is also logged as "<LABEL> <stage>: <id>" in $JOB_LOG, which compare.py reads.
#
# Env (defaults in brackets; the path defaults are the 2026-11 H231/H271 session's, override them elsewhere):
#   DRY_RUN=1                 print every command and spec, launch nothing
#   STAGES                    [export sanity vision emocheck text]
#   SUITES                    [docs24 core list]
#   TEXT_CELLS                [full]
#   REUSE_EXPORT=1            adopt an existing verified export in $OUT (drop "export" from STAGES)
#   OUT_ROOT / OUT            [/weka/olmo-3p5-post-checkpoints/$SECRET_USER/mmsft-eval] / [$OUT_ROOT/$LABEL]
#   EVAL_HOME                 local state, specs, launch logs [~/.cache/olmo35-mm-eval]; JOB_LOG [$EVAL_HOME/jobs/$LABEL.log]
#   OI_JOB_DIR                this open-instruct checkout as Beaker jobs see it (on Weka) [this checkout, if on /weka]
#   OLMO_CORE_DIR             job-visible allenai/OLMo-core checkout at the multimodal SFT code (#942)
#                             [/weka/oe-training-default/jasonr/home/OLMo-core-wt-sweep-fd9bba1]
#   OLMO_CORE_SHA             OLMo-core the vision jobs install [fd9bba1ba3f12721ae1ff3493f0b266366debec6 = #942]
#   OLMO_EVAL_DIR             local allenai/olmo-eval checkout, branch jasonr/olmo35-mmsft-eval, pushed and clean
#                             (eval_vision.sh runs its HEAD) [/root/olmo-eval-wt-mmsft]; OLMO_EVAL_PY [python]
#   OLMO_EVAL_JOB_DIR         the same olmo-eval as jobs see it (sanity) [/weka/oe-training-default/jasonr/home/olmo-eval-wt-mmsft]
#   LEDGER_DIR                olmo-post-training-ledger checkout (recorded text cell specs) [/root/olmo-post-training-ledger]
#   SECRET_USER               <user>_HF_TOKEN / <user>_WANDB_API_KEY secret prefix [the Beaker account name]
#   WORKSPACE, BUDGET         [ai2/oe-olmo3p5-mt], [ai2/oe-other]
#   PRIORITY                  [urgent]; a refused priority falls back to high, then normal
#   JOB_CLUSTERS, JOB_PREEMPTIBLE  1-GPU jobs (export/sanity/emocheck) [ai2/holmes], [unset = not preemptible]
#   JOB_IMAGE                 [01M28XX5N4KM6PD0G81KC41BDW = akshitab/olmo-core-tch2130cu130-fa4-rma-2026-09-11]
#   TEXT_CLUSTER              [ai2/holmes]; BFCL_CLUSTER [ai2/jupiter]; BFCL_WORKSPACE [ai2/olmo-instruct]
#   PLUGIN_DIR                re-point the text cells' OLMoE3 plugins at a job-visible scripts/export/olmo35_mm
#                             [unset = the recorded scaling-ladders runtime, needed for comparisons with past runs]
#   SANITY_REF                reference export(s) for the sanity side-by-side [none]
set -uo pipefail
LABEL=${1:?usage: run_eval.sh LABEL STEP_DIR}
DCP=${2:?usage: run_eval.sh LABEL STEP_DIR}
DCP=${DCP%/}
DRY=${DRY_RUN:-}
HERE=$(cd "$(dirname "$0")" && pwd)
OI_DIR=$(cd "$HERE/../../.." && pwd)
STAGES=${STAGES:-export sanity vision emocheck text}
SUITES=${SUITES:-docs24 core list}
TEXT_CELLS=${TEXT_CELLS:-full}
REUSE=${REUSE_EXPORT:-}
if [[ -z ${SECRET_USER:-} ]]; then
  SECRET_USER=$(beaker account whoami --format json 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print(d["name"])' 2>/dev/null)
fi
SECRET_USER=${SECRET_USER:-$USER}
OUT_ROOT=${OUT_ROOT:-/weka/olmo-3p5-post-checkpoints/$SECRET_USER/mmsft-eval}
OUT=${OUT:-$OUT_ROOT/$LABEL}
EVAL_HOME=${EVAL_HOME:-$HOME/.cache/olmo35-mm-eval}
STATE=$EVAL_HOME/state/$LABEL; SPECS=$EVAL_HOME/specs/$LABEL; LAUNCH=$EVAL_HOME/launch/$LABEL
JOB_LOG=${JOB_LOG:-$EVAL_HOME/jobs/$LABEL.log}
if [[ -z ${OI_JOB_DIR:-} && $OI_DIR == /weka/* ]]; then OI_JOB_DIR=$OI_DIR; fi
OI_JOB_DIR=${OI_JOB_DIR:-}
export OLMO_CORE_DIR=${OLMO_CORE_DIR:-/weka/oe-training-default/jasonr/home/OLMo-core-wt-sweep-fd9bba1}
OLMO_CORE_SHA=${OLMO_CORE_SHA:-fd9bba1ba3f12721ae1ff3493f0b266366debec6}
OLMO_EVAL_DIR=${OLMO_EVAL_DIR:-/root/olmo-eval-wt-mmsft}; OLMO_EVAL_PY=${OLMO_EVAL_PY:-python}
OLMO_EVAL_JOB_DIR=${OLMO_EVAL_JOB_DIR:-/weka/oe-training-default/jasonr/home/olmo-eval-wt-mmsft}
LEDGER_DIR=${LEDGER_DIR:-/root/olmo-post-training-ledger}
WORKSPACE=${WORKSPACE:-ai2/oe-olmo3p5-mt}; BUDGET=${BUDGET:-ai2/oe-other}
PRIORITY=${PRIORITY:-urgent}
JOB_IMAGE=${JOB_IMAGE:-01M28XX5N4KM6PD0G81KC41BDW}  # akshitab/olmo-core-tch2130cu130-fa4-rma-2026-09-11, the image H231/H271 exported with
TEXT_CLUSTER=${TEXT_CLUSTER:-ai2/holmes}; BFCL_CLUSTER=${BFCL_CLUSTER:-ai2/jupiter}; BFCL_WORKSPACE=${BFCL_WORKSPACE:-ai2/olmo-instruct}
FULL_TEXT="gpqa gpqamain ifeval ifbench ifeval_mt multi_if math-algebra math-counting_and_probability math-geometry math-intermediate_algebra math-number_theory math-prealgebra math-precalculus hmmt26 ruler64k ruler128k rp-aggqa rp-mvmqvt rp-niah-smk"
FULL_BFCL="bfcl-single bfcl-multi-base bfcl-multi-long_context bfcl-multi-miss_func bfcl-multi-miss_param"
case $PRIORITY in urgent) PRIS="urgent high normal" ;; high) PRIS="high normal" ;; *) PRIS=$PRIORITY ;; esac

has() { [[ " $STAGES " == *" $1 "* ]]; }
note() {
  local m; m="$(date -u +%m-%d\ %H:%M) $LABEL $*"
  if [[ -n $DRY ]]; then echo "[note] $m"; else echo "$m" | tee -a "$JOB_LOG"; fi
}
die() { note "FAILED: $*"; exit 1; }
jstate() {
  beaker experiment get "$1" --format json 2>/dev/null | python3 -c "import json,sys
d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d
s=d['jobs'][-1]['status'] if d.get('jobs') else {}
print(('exit%s'%s.get('exitCode')) if s.get('exited') else ('canceled' if s.get('canceled') else ('started' if s.get('started') else 'queued')))"
}
logs_has() { local l; l=$(beaker experiment logs "$1" 2>/dev/null); [[ $l == *"$2"* ]]; }
wait_job() { local st; while true; do st=$(jstate "$1"); case $st in exit* | canceled) echo "$st"; return ;; esac; sleep 120; done; }
recorded() { [[ -f $STATE/$1 ]] && cat "$STATE/$1"; }
record() { [[ -n $DRY ]] || echo "$2" > "$STATE/$1"; }
job_script() { echo "$OI_JOB_DIR/scripts/$1"; }

# A 1-GPU job spec: the training image, the Weka buckets, never preemptible unless JOB_PREEMPTIBLE=1.
mkspec() {  # mkspec NAME PRIORITY COMMAND
  python3 - "$SPECS/$1.json" "$1" "$2" "$3" "$JOB_IMAGE" "${JOB_CLUSTERS:-ai2/holmes}" "${JOB_PREEMPTIBLE:-}" <<'EOF'
import json, sys
path, name, priority, cmd, image, clusters, preemptible = sys.argv[1:]
buckets = ["oe-training-default", "olmo-3p5-post-checkpoints", "olmo-3p5-mid-checkpoints", "olmo-3p5-checkpoints"]
spec = {
    "version": "v2",
    "description": f"olmo35 multimodal SFT eval: {cmd}",
    "tasks": [{
        "name": name[:60],
        "image": {"beaker": image},
        "command": ["/bin/bash", "-c"],
        "arguments": [cmd],
        "datasets": [{"mountPath": f"/weka/{b}", "source": {"weka": b}} for b in buckets],
        "resources": {"gpuCount": 1, "sharedMemory": "64 GiB"},
        "context": {"priority": priority, "preemptible": preemptible == "1"},
        "constraints": {"cluster": clusters.split(",")},
    }],
}
with open(path, "w") as f:
    json.dump(spec, f, indent=1)
EOF
}
submit() {  # submit NAME COMMAND -> experiment id on stdout
  local name=$1 cmd=$2 pri out id
  for pri in $PRIS; do
    mkspec "$name" "$pri" "$cmd"
    echo "+ beaker experiment create $SPECS/$name.json --workspace $WORKSPACE --name $name   # $pri: $cmd" >&2
    [[ -n $DRY ]] && { echo DRYRUN_ID; return; }
    out=$(beaker experiment create "$SPECS/$name.json" --workspace "$WORKSPACE" --name "$name-$(date -u +%H%M)" 2>&1)
    id=$(echo "$out" | grep -oE "01[A-Z0-9]{24}" | head -1)
    [[ -n $id ]] && { echo "$id"; return; }
    echo "$out" | grep -qi priorit || { echo "submit failed: $out" >&2; return 1; }
    echo "priority $pri refused, trying the next lower one" >&2
  done
  return 1
}

mkdir -p "$SPECS"
[[ -n $DRY ]] || mkdir -p "$STATE" "$LAUNCH" "$(dirname "$JOB_LOG")"
if [[ -z $OI_JOB_DIR ]]; then
  [[ -n $DRY ]] || die "set OI_JOB_DIR to this open-instruct checkout as Beaker jobs see it (on Weka); jobs run its scripts"
  OI_JOB_DIR="<OI_JOB_DIR>"
fi
echo "== $LABEL: $DCP -> $OUT (stages: $STAGES; suites: $SUITES; text: $TEXT_CELLS; job scripts from $OI_JOB_DIR)"

# ------------------------------------------------------------------------------------------------ export
EXP=$(recorded export)
if [[ -n $REUSE ]]; then
  has export && die "REUSE_EXPORT=1 with the export stage requested; drop export from STAGES"
  EXP=reused; note "reusing the existing verified export in $OUT (REUSE_EXPORT=1)"
fi
if has export && [[ -z $EXP ]]; then
  EXP=$(submit "mmsft-export-$LABEL" "OLMO_CORE_DIR=$OLMO_CORE_DIR bash $(job_script export/olmo35_mm/export_all_job.sh) $DCP $OUT") \
    || die "export submit failed"
  record export "$EXP"; note "export: $EXP"
fi
vision_export_ok() {  # the verified olmo-core-export exists (MMSFT-VISION-EXPORT-OK in the export job's log)
  [[ -n $DRY || $EXP == reused ]] && return 0
  [[ -n $EXP ]] || return 1
  until logs_has "$EXP" "MMSFT-VISION-EXPORT-OK"; do case $(jstate "$EXP") in exit* | canceled) break ;; esac; sleep 120; done
  logs_has "$EXP" "MMSFT-VISION-EXPORT-OK"
}
text_export_ok() {  # hf_text exists and passed the smoke (TOKENIZER-CHECK OK)
  [[ -n $DRY || $EXP == reused ]] && return 0
  [[ -n $EXP ]] || return 1
  [[ $(wait_job "$EXP") == exit0 ]] && logs_has "$EXP" "TOKENIZER-CHECK OK"
}

# ------------------------------------------------------------------------------------------------ sanity
if has sanity && [[ -z $(recorded sanity) ]]; then
  vision_export_ok || die "export job $EXP failed before the verified olmo-core-export"
  id=$(submit "mmsft-sanity-$LABEL" "OLMO_CORE_DIR=$OLMO_CORE_DIR OLMO_EVAL_JOB_DIR=$OLMO_EVAL_JOB_DIR bash $(job_script eval/olmo35_mm/sanity_job.sh) $OUT/olmo-core-export ${SANITY_REF:-}") \
    || die "sanity submit failed"
  record sanity "$id"; note "sanity: $id"
fi

# ------------------------------------------------------------------------------------------------ vision
if has vision; then
  vision_export_ok || die "export job $EXP failed before the verified olmo-core-export"
  for spec in "docs24|chart_qa doc_qa text_vqa info_qa|max_crops=24" "core|vqa2 ai2d countbench_qa real_world_qa|" \
    "list|mmmu muir_bench mmiu blink mmmu_pro|"; do
    IFS='|' read -r tag tasks kw <<< "$spec"
    [[ " $SUITES " == *" $tag "* ]] || continue
    if [[ -n $(recorded "vision-$tag") ]]; then echo "vision $tag: already launched ($(recorded "vision-$tag"))"; continue; fi
    id=
    for pri in $PRIS; do
      echo "+ (cd $OLMO_EVAL_DIR && WORKSPACE=$WORKSPACE BUDGET=$BUDGET OLMO_CORE_SHA=$OLMO_CORE_SHA PRIORITY=$pri SKIP_LOCAL_CHECK=1 TAG=$tag TASKS=\"$tasks\" PROVIDER_KWARGS=\"$kw\" LIMIT=500 GPUS=2 TIMEOUT=12h OLMO_EVAL_PY=$OLMO_EVAL_PY scripts/olmo35_mmsft/eval_vision.sh $LABEL $OUT/olmo-core-export olmo35)"
      if [[ -n $DRY ]]; then  # olmo-eval's own dry run: builds and prints the experiment, launches nothing
        if (cd "$OLMO_EVAL_DIR" && env DRY_RUN=1 WORKSPACE="$WORKSPACE" BUDGET="$BUDGET" OLMO_CORE_SHA="$OLMO_CORE_SHA" \
          PRIORITY="$pri" SKIP_LOCAL_CHECK=1 TAG="$tag" TASKS="$tasks" PROVIDER_KWARGS="$kw" LIMIT=500 GPUS=2 TIMEOUT=12h \
          OLMO_EVAL_PY="$OLMO_EVAL_PY" timeout 900 scripts/olmo35_mmsft/eval_vision.sh "$LABEL" "$OUT/olmo-core-export" olmo35 \
          > "$SPECS/vision-$tag.dryrun.log" 2>&1); then
          echo "  olmo-eval dry run OK: $SPECS/vision-$tag.dryrun.log"
        else
          echo "  olmo-eval dry run FAILED: $SPECS/vision-$tag.dryrun.log"
        fi
        break
      fi
      (cd "$OLMO_EVAL_DIR" && env WORKSPACE="$WORKSPACE" BUDGET="$BUDGET" OLMO_CORE_SHA="$OLMO_CORE_SHA" PRIORITY="$pri" \
        SKIP_LOCAL_CHECK=1 TAG="$tag" TASKS="$tasks" PROVIDER_KWARGS="$kw" LIMIT=500 GPUS=2 TIMEOUT=12h \
        OLMO_EVAL_PY="$OLMO_EVAL_PY" timeout 900 scripts/olmo35_mmsft/eval_vision.sh "$LABEL" "$OUT/olmo-core-export" olmo35 \
        > "$LAUNCH/vision-$tag.log" 2>&1)
      id=$(sed 's/\x1b\[[0-9;]*m//g' "$LAUNCH/vision-$tag.log" | grep -oE 'beaker.org/ex/01[A-Z0-9]{24}' | head -1)
      [[ -n $id ]] && break
      sed 's/\x1b\[[0-9;]*m//g' "$LAUNCH/vision-$tag.log" | grep -qi priorit || break
    done
    [[ -n $DRY ]] && continue
    note "vision $tag: ${id:-FAILED, see $LAUNCH/vision-$tag.log}"
    [[ -n $id ]] && record "vision-$tag" "${id##*/}"
  done
fi

# ------------------------------------------------------------------------------------------------ emocheck
if has emocheck && [[ -z $(recorded emocheck) ]]; then
  text_export_ok || die "text export failed; see beaker experiment logs $EXP"
  id=$(submit "mmsft-emocheck-$LABEL" "OLMO_CORE_DIR=$OLMO_CORE_DIR bash $(job_script export/olmo35_mm/emo_check_job.sh) $DCP $OUT") \
    || die "emo check submit failed"
  record emocheck "$id"; note "emocheck: $id"
fi

# ------------------------------------------------------------------------------------------------ text
if has text && [[ $TEXT_CELLS != none && -z $(recorded text) ]]; then
  text_export_ok || die "text export failed; see beaker experiment logs $EXP"
  cells() {  # cells CLUSTER WORKSPACE CELL...
    local cluster=$1 ws=$2; shift 2
    local cmd=(python3 "$HERE/launch_text_cells.py" "$OUT/export/hf_text" "$LABEL" --olmo-eval-dir "$OLMO_EVAL_DIR"
      --ledger "$LEDGER_DIR" --cells "$@" --cluster "$cluster" --workspace "$ws" --priority "$PRIORITY"
      --user "$SECRET_USER" --specs "$SPECS/text")
    [[ -n ${PLUGIN_DIR:-} ]] && cmd+=(--plugin-dir "$PLUGIN_DIR")
    [[ -n $DRY ]] && cmd+=(--dry-run)
    echo "+ ${cmd[*]}"
    "${cmd[@]}" 2>&1 | while read -r line; do [[ -n $line ]] && note "text cell $line"; done
  }
  if [[ $TEXT_CELLS == full ]]; then
    cells "$TEXT_CLUSTER" "$WORKSPACE" $FULL_TEXT
    cells "$BFCL_CLUSTER" "$BFCL_WORKSPACE" $FULL_BFCL
  else
    cells "$TEXT_CLUSTER" "$WORKSPACE" $TEXT_CELLS
  fi
  record text launched
fi
note "pipeline launched; results: python3 $HERE/compare.py --run $LABEL=$JOB_LOG [--run OTHER=...]"
