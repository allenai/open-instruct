#!/bin/bash
# BFCL v3 (single-turn + multi-turn function calling) on an OLMoE3 KDA MoE checkpoint, on Beaker.
#
#   # 1) export the olmo-core DCP checkpoint to HF (0 GPU, CPU conversion, ~35 GB written)
#   ./scripts/eval/bfcl/olmoe3_kda_bfcl_v3.sh convert <dcp_step_dir> [hf_out_dir]
#   # 2) serve it with vLLM inside the job and run BFCL against the endpoint (1 GPU)
#   ./scripts/eval/bfcl/olmoe3_kda_bfcl_v3.sh eval <hf_dir> <run_name>
#
# e.g. for the continued-SFT runs in this series:
#   ./scripts/eval/bfcl/olmoe3_kda_bfcl_v3.sh convert /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/zlif6wec/step351
#   ./scripts/eval/bfcl/olmoe3_kda_bfcl_v3.sh eval /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/zlif6wec/hf_step351 kda-sft-simfc
#
# How it works, and why this shape:
#
# * The model architecture (olmo3moe: KDA linear attention + latent MoE) is not in stock vLLM
#   or transformers. It is served through the scaling-ladders plugins, which only load on the
#   torch/vLLM combination in akshitab/olmo-core-tch2110cu128-rma-2026-08-04 (see
#   scripts/train/debug/eval_olmoe3_kda.sh for the verified pin set). The eval job uses that
#   image and installs the same dependency set at runtime; the runner is fetched from GitHub at
#   this commit (or read from the checkout when it lives on Weka), since the image has no copy
#   of this repo.
# * BFCL is run in function-calling mode against vLLM's OpenAI-compatible server. vLLM parses
#   the Olmo 3.5 template's Qwen3-Coder-style XML tool calls (--tool-call-parser qwen3_xml)
#   into structured tool_calls and strips the forced <think> block (--reasoning-parser olmo3,
#   which works on plain-text think tags; the qwen3 parser needs them as single tokens),
#   so BFCL's stock OpenAI handler evaluates the model exactly as it evaluates gpt-*-FC. The
#   model is registered into BFCL's config at runtime by bfcl_cli_with_olmo_models.py.
# * TOOL_CALL_PARSER / REASONING_PARSER select vLLM's parsers for the checkpoint's template.
#   Defaults fit the Olmo 3.5 template (qwen3_xml / olmo3). A checkpoint exported with the
#   Olmo 3 instruct-dev template (`<function_calls>name(k=v)</function_calls>`) needs
#   TOOL_CALL_PARSER=olmo3; the reasoning parser is the same for both.
# * TEST_CATEGORY defaults to single_turn,multi_turn -- the BFCL v3 scope. BFCL's `all` also
#   runs the v4 agentic memory and web_search categories, the latter needing a SerpAPI key.
# * Results land under OUT_DIR on Weka: result/<model>/ (raw responses) and score/ (per-category
#   JSON plus data_overall.csv etc.), so they survive the job and can be re-scored offline.
#
# Not verified end to end at the time of writing: the runtime install of vllm 0.19.1 plus plugins
# outside olmo-eval's launcher, and qwen3_xml's handling of every argument type the template
# renders. The runner fires one tool-call smoke request before the suite and warns if no
# structured tool_calls come back; read that line in the job log first.
set -euo pipefail

MODE="${1:?usage: $0 convert <dcp_step_dir> [hf_out_dir] | eval <hf_dir> <run_name>}"
shift

PRIORITY="${PRIORITY:-urgent}"
PY="${PY:-uv run python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Workspace follows the cluster: jobs on jupiter or holmes go to ai2/olmo-instruct, everything
# else must run in ai2/open-instruct-dev. Override with WORKSPACE.
workspace_for() {
    case "$1" in
        *jupiter*|*holmes*) echo "ai2/olmo-instruct" ;;
        *) echo "ai2/open-instruct-dev" ;;
    esac
}

case "$MODE" in
    convert)
        DCP="${1:?convert needs <dcp_step_dir>}"
        HF_OUT="${2:-$(dirname "$DCP")/hf_$(basename "$DCP")}"
        # The training image carries olmo-core at the branch pin and the training config the
        # weights were saved under; conversion needs both. CPU is fine (the KDA kernels are only
        # needed for validation, which is skipped for this family, see the converter's docstring)
        # and avoids queueing for a CUDA 13 GPU node.
        TRAIN_IMAGE="${TRAIN_IMAGE:-$(beaker account whoami --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["name"])')/open-instruct-integration-test-pd-kda-moe-continued-sft-cuda13}"
        CONFIG="${CONFIG:-scripts/train/debug/kda_lc_sft.json}"
        TOKENIZER="${TOKENIZER:-allenai/dolma2-tokenizer-olmo35}"
        # The export embeds the tokenizer and its chat template, which vLLM then serves with. The
        # tokenizer repo's template is edited upstream, so pin the revision the checkpoint was
        # trained with (the same default as the training launcher): the job downloads that
        # snapshot to Weka and hands the converter the directory, since the converter itself
        # takes no revision. A TOKENIZER that is already a local path is used as is.
        TOKENIZER_REVISION="${TOKENIZER_REVISION:-56415cee534a924b0b777d70a888266f4eef65ec}"
        if [[ "$TOKENIZER" == /* ]]; then
            TOKENIZER_PREP=""
            TOKENIZER_ARG="$TOKENIZER"
        else
            TOKENIZER_ARG="/weka/oe-adapt-default/$(beaker account whoami --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["name"])')/tokenizers/$(basename "$TOKENIZER")-${TOKENIZER_REVISION:0:8}"
            TOKENIZER_PREP="uv run hf download $TOKENIZER --revision $TOKENIZER_REVISION --local-dir $TOKENIZER_ARG && "
        fi
        MAX_SEQ="${MAX_SEQ:-65536}"
        CONVERT_CLUSTERS="${CONVERT_CLUSTERS:-ai2/saturn ai2/neptune ai2/ceres}"
        WORKSPACE="${WORKSPACE:-$(workspace_for "$CONVERT_CLUSTERS")}"
        echo "convert $DCP -> $HF_OUT (image $TRAIN_IMAGE, tokenizer $TOKENIZER @ ${TOKENIZER_REVISION:0:8} -> $TOKENIZER_ARG, workspace $WORKSPACE)"
        # shellcheck disable=SC2086
        $PY mason.py \
            --cluster $CONVERT_CLUSTERS \
            --workspace "$WORKSPACE" --priority "$PRIORITY" \
            --image "$TRAIN_IMAGE" --pure_docker_mode \
            --description "Convert $(basename "$(dirname "$DCP")")/$(basename "$DCP") to HF (OLMoE3 KDA)" \
            --timeout "${JOB_TIMEOUT:-4h}" \
            --num_nodes 1 --gpus 0 --non_resumable --no_auto_dataset_cache \
            -- "${TOKENIZER_PREP}uv run python scripts/train/debug/convert_moe_checkpoint_to_hf.py -i $DCP -o $HF_OUT -c $CONFIG -t $TOKENIZER_ARG -s $MAX_SEQ --skip-validation --device cpu"
        ;;
    eval)
        HF_DIR="${1:?eval needs <hf_dir>}"
        RUN_NAME="${2:?eval needs <run_name>}"
        EVAL_IMAGE="${EVAL_IMAGE:-akshitab/olmo-core-tch2110cu128-rma-2026-08-04}"
        CLUSTER="${CLUSTER:-ai2/ceres}"
        WORKSPACE="${WORKSPACE:-$(workspace_for "$CLUSTER")}"
        GPUS="${GPUS:-1}"
        OUT_DIR="${OUT_DIR:-/weka/oe-adapt-default/$(beaker account whoami --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["name"])')/bfcl/$RUN_NAME}"
        TEST_CATEGORY="${TEST_CATEGORY:-single_turn,multi_turn}"
        NUM_THREADS="${NUM_THREADS:-32}"
        BFCL_MODEL_NAME="${BFCL_MODEL_NAME:-${RUN_NAME}-FC}"

        # mason joins the words after -- with spaces and runs them under `bash -c`, so the job
        # command below is passed as a single word and its `$RUNNER`/`$CLI_PY` expand in the job.
        # The image has no copy of this repo. Read the runner and the CLI shim from the checkout
        # when it is on Weka (visible inside the job), otherwise from GitHub at this exact commit,
        # which therefore has to be pushed.
        RUNNER_REL=scripts/eval/bfcl/run_bfcl_v3_in_job.sh
        CLI_REL=scripts/eval/bfcl/bfcl_cli_with_olmo_models.py
        SERVE_REL=scripts/eval/olmoe3_kda_vllm_serve.sh
        if [[ "$REPO_ROOT" == /weka/* ]]; then
            FETCH="RUNNER=$REPO_ROOT/$RUNNER_REL; CLI_PY=$REPO_ROOT/$CLI_REL; SERVE_LIB=$REPO_ROOT/$SERVE_REL"
        else
            if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- scripts/eval)" ]]; then
                echo "scripts/eval has uncommitted changes; commit and push so the job can fetch them" >&2; exit 1
            fi
            COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
            git -C "$REPO_ROOT" branch -r --contains "$COMMIT" | grep -q origin/ || { echo "commit $COMMIT is not pushed to origin" >&2; exit 1; }
            RAW="https://raw.githubusercontent.com/allenai/open-instruct/$COMMIT"
            FETCH="mkdir -p /opt/bfcl-run && curl -sfL $RAW/$RUNNER_REL -o /opt/bfcl-run/run.sh && curl -sfL $RAW/$CLI_REL -o /opt/bfcl-run/cli.py && curl -sfL $RAW/$SERVE_REL -o /opt/bfcl-run/serve.sh && RUNNER=/opt/bfcl-run/run.sh; CLI_PY=/opt/bfcl-run/cli.py; SERVE_LIB=/opt/bfcl-run/serve.sh"
        fi

        echo "eval $HF_DIR as $BFCL_MODEL_NAME on $CLUSTER x$GPUS (workspace $WORKSPACE); results -> $OUT_DIR"
        $PY mason.py \
            --cluster "$CLUSTER" \
            --workspace "$WORKSPACE" --priority "$PRIORITY" \
            --image "$EVAL_IMAGE" --pure_docker_mode \
            --description "BFCL v3 ($TEST_CATEGORY) on $RUN_NAME via vLLM" \
            --timeout "${JOB_TIMEOUT:-12h}" \
            --num_nodes 1 --gpus "$GPUS" --non_resumable --no_auto_dataset_cache \
            --env "CKPT=$HF_DIR" --env "OUT_DIR=$OUT_DIR" \
            --env "BFCL_MODEL_NAME=$BFCL_MODEL_NAME" --env "TEST_CATEGORY=$TEST_CATEGORY" \
            --env "NUM_THREADS=$NUM_THREADS" --env "TENSOR_PARALLEL=$GPUS" \
            --env "MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}" \
            --env "TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_xml}" --env "REASONING_PARSER=${REASONING_PARSER:-olmo3}" \
            --env "VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}" --env "BFCL_ALLOW_OVERWRITE=${BFCL_ALLOW_OVERWRITE:-0}" \
            --env "PLUGIN_DIR=${PLUGIN_DIR:-/weka/oe-adapt-default/abhishekr/repos/scaling-ladders-emo/ladders/olmoe3}" \
            -- "$FETCH && CLI_PY=\$CLI_PY SERVE_LIB=\$SERVE_LIB bash \$RUNNER"
        ;;
    *)
        echo "Unknown mode: $MODE (expected 'convert' or 'eval')" >&2; exit 1
        ;;
esac
