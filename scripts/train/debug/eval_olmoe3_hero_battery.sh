#!/bin/bash
# Launch an eval battery for an Olmo 3.5 hero-small HF export, one Beaker
# experiment per task, through eval_olmoe3_hero.sh.
#
#   eval_olmoe3_hero_battery.sh <hf_checkpoint_dir> <label> dev|paper|both
#
# Appends "<label> <task> <experiment-url>" lines to $LAUNCH_LOG (default
# ./hero_eval_launches.txt) so pull_preds.sh / paired.py can join them later.
#
# dev   : the 12-task selection battery of #1853/#1859/#1880 (ledger continuity),
#         run through the same olmo-eval checkouts the proxy used
#         (OLMO_EVAL_DIR_DEV for the MoE-capable tasks, OLMO_EVAL_DIR_IFEVAL for
#         ifeval, as midrun_eval.sh did).
# paper : the 32K-cap paper protocol of #1875 on olmo-eval abhishekr/paper-protocol:
#         zero-shot chat, T=0.6 / top-p 0.95, think trace stripped, 32K cap for the
#         sampled tasks; aime_2024:olmo3adapt (32 samples, 16K); popqa:chat on the
#         seeded 2,000-prompt sample; the seven minerva_math subjects (macro over
#         subjects is the MATH number). Arguments copied from the #1875 jobs
#         (e.g. 01M21M7DV0J4DA1MVQZ5CC99NA, 01M21M776R870531WC9R6KWW08).
set -euo pipefail

CKPT="${1:?usage: $0 <hf_checkpoint_dir> <label> dev|paper|both}"
LABEL="${2:?usage: $0 <hf_checkpoint_dir> <label> dev|paper|both}"
WHICH="${3:-both}"
if [[ "${SCREEN_PASSED:-0}" != 1 ]]; then
    echo "Full confirmation requires SCREEN_PASSED=1 after paired CI and H008 seed-floor review." >&2
    exit 1
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
LAUNCHER="$HERE/eval_olmoe3_hero.sh"
LAUNCH_LOG="${LAUNCH_LOG:-./hero_eval_launches.txt}"

OLMO_EVAL_DIR_DEV="${OLMO_EVAL_DIR_DEV:-/root/repos/olmo-eval-moe}"
OLMO_EVAL_DIR_IFEVAL="${OLMO_EVAL_DIR_IFEVAL:-/root/repos/olmo-eval-launch-ifeval}"
OLMO_EVAL_DIR_PAPER="${OLMO_EVAL_DIR_PAPER:-/root/repos/olmo-eval-paper}"

DEV_TASKS="gsm8k gsm_symbolic math500 ifeval_ood popqa ifeval_mt_wildchat_unused_withRewrite ifeval_mt_ood_wildchat_unused_withRewrite ruler_all__8192 ruler_all__16384 ruler_all__32768 ruler_all__65536"
DEV_IFEVAL_TASKS="ifeval"
MATH_SUBJECTS="algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus"
SAMPLED="-o max_tokens=32768 -o temperature=0.6 -o top_p=0.95 -o do_sample=true -o strip_thinking=true"

launch() {  # <olmo-eval dir> <task-label> <launcher args...>
    local dir="$1" task="$2"; shift 2
    local url
    url=$(OLMO_EVAL_DIR="$dir" bash "$LAUNCHER" "$CKPT" "$LABEL-$task" "$@" 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' | grep -o 'https://beaker.org/ex/[A-Z0-9]*' | tail -1 || true)
    echo "$LABEL $task ${url:-LAUNCH_FAILED}" | tee -a "$LAUNCH_LOG"
}

if [[ "$WHICH" == "dev" || "$WHICH" == "both" ]]; then
    for t in $DEV_TASKS; do launch "$OLMO_EVAL_DIR_DEV" "$t" -t "$t"; done
    for t in $DEV_IFEVAL_TASKS; do launch "$OLMO_EVAL_DIR_IFEVAL" "$t" -t "$t"; done
fi
if [[ "$WHICH" == "paper" || "$WHICH" == "both" ]]; then
    # shellcheck disable=SC2086
    launch "$OLMO_EVAL_DIR_PAPER" paper-ifeval     -t ifeval $SAMPLED
    # shellcheck disable=SC2086
    launch "$OLMO_EVAL_DIR_PAPER" paper-ifeval_ood -t ifeval_ood $SAMPLED
    launch "$OLMO_EVAL_DIR_PAPER" paper-popqa      -t popqa:chat -o max_tokens=32768 -o strip_thinking=true -o limit=2000
    launch "$OLMO_EVAL_DIR_PAPER" paper-aime2024   -t aime_2024:olmo3adapt
    for s in $MATH_SUBJECTS; do launch "$OLMO_EVAL_DIR_PAPER" "paper-math-$s" -t "minerva_math_$s:olmo3adapt"; done
fi
echo "BATTERY LAUNCHED: $WHICH for $LABEL -> $LAUNCH_LOG"
