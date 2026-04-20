#!/bin/bash
# Run all remaining VIP test phases (2b–8) sequentially on 2 local GPUs.
# Logs each phase to /tmp/vip_phases/<phase>.log and prints PASS/FAIL summary.
set -uo pipefail

LOGDIR=/tmp/vip_phases
OUTDIR=/tmp/vip_smoke_output
mkdir -p "$LOGDIR"

unset LD_LIBRARY_PATH
export NCCL_CUMEM_ENABLE=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_home/datasets
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

VENV=.venv/bin/python

# ── helpers ──────────────────────────────────────────────────────────────────

ray_stop() { .venv/bin/ray stop --force 2>/dev/null || true; }
ray_start() { ray_stop; .venv/bin/ray start --head --port=8888 --dashboard-host=0.0.0.0 2>/dev/null; mkdir -p "$HOME/.triton/autotune"; }

# Common trainer flags for all 2-GPU smoke runs
COMMON_FLAGS=(
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64
    --dataset_mixer_list_splits train
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16
    --dataset_mixer_eval_list_splits train
    --max_prompt_token_length 512 --response_length 512 --pack_length 1024
    --per_device_train_batch_size 1
    --num_unique_prompts_rollout 8 --num_samples_per_prompt_rollout 4
    --model_name_or_path Qwen/Qwen3-0.6B
    --add_bos --stop_strings "</answer>"
    --apply_r1_style_format_reward --apply_verifiable_reward true
    --ground_truths_key ground_truth
    --chat_template_name r1_simple_chat_postpend_think
    --temperature 0.7 --beta 0.0
    --learning_rate 3e-7 --total_episodes 160
    --deepspeed_stage 2 --num_epochs 1
    --num_learners_per_node 1 --vllm_num_engines 1 --vllm_tensor_parallel_size 1
    --vllm_sync_backend gloo --vllm_gpu_memory_utilization 0.4 --vllm_enforce_eager
    --inflight_updates True --async_steps 2 --seed 3
    --local_eval_every 5 --save_freq 5
    --gradient_checkpointing --with_tracking False --push_to_hub False
    --use_value_model
    --value_learning_rate 5e-6 --gae_lambda 0.95 --gamma 1.0
    --value_loss_coef 0.5 --vf_clip_range 0.2
)

RESULTS=()

run_phase() {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    echo ""
    echo "══════════════════════════════════════════"
    echo " Starting $name"
    echo "══════════════════════════════════════════"
    ray_start
    $VENV open_instruct/grpo_fast.py \
        --exp_name "vip_${name}" \
        --output_dir "$OUTDIR/phase_${name}" \
        "${COMMON_FLAGS[@]}" "$@" >"$log" 2>&1
    local rc=$?
    ray_stop
    if [ $rc -eq 0 ]; then
        echo "  ✓ $name: PASS (exit 0)"
        RESULTS+=("PASS  $name")
    else
        echo "  ✗ $name: FAIL (exit $rc) — see $log"
        RESULTS+=("FAIL  $name")
    fi
    return $rc
}

check_training_args() {
    local phase="$1"; local key="$2"; local expected="$3"
    local f
    f=$(find "$OUTDIR/phase_${phase}" -name "training_args.json" | head -1)
    if [ -z "$f" ]; then echo "  ✗ No training_args.json found for $phase"; return 1; fi
    local val
    val=$($VENV -c "import json; d=json.load(open('$f')); print(d.get('$key','MISSING'))" 2>/dev/null)
    if [ "$val" = "$expected" ]; then
        echo "  ✓ training_args[$key] = $expected"
    else
        echo "  ✗ training_args[$key] = $val (expected $expected)"
    fi
}

check_log() {
    local phase="$1"; local pattern="$2"
    if grep -qE "$pattern" "$LOGDIR/${phase}.log" 2>/dev/null; then
        echo "  ✓ log contains: $pattern"
    else
        echo "  ✗ log missing: $pattern"
    fi
}

# ── Phase 2b: expected_accuracy ───────────────────────────────────────────────
run_phase "2b_expected_accuracy" \
    --value_model_ground_truth_conditioning \
    --gt_conditioning_template expected_accuracy
check_training_args "2b_expected_accuracy" "gt_conditioning_template" "expected_accuracy"

# ── Phase 2c: rollout_context ─────────────────────────────────────────────────
run_phase "2c_rollout_context" \
    --value_model_ground_truth_conditioning \
    --gt_conditioning_template rollout_context \
    --rollout_context_num_siblings 4
check_training_args "2c_rollout_context" "gt_conditioning_template" "rollout_context"

# ── Phase 2d: correct_demo ────────────────────────────────────────────────────
run_phase "2d_correct_demo" \
    --value_model_ground_truth_conditioning \
    --gt_conditioning_template correct_demo
check_training_args "2d_correct_demo" "gt_conditioning_template" "correct_demo"

# ── Phase 3: SAE ──────────────────────────────────────────────────────────────
run_phase "3_sae" \
    --use_sae --sae_threshold 0.2
check_training_args "3_sae" "use_sae" "True"

# ── Phase 4: LM-yesno ─────────────────────────────────────────────────────────
run_phase "4_lm_yesno" \
    --use_lm_value_model \
    --gt_conditioning_template lm_yesno
check_training_args "4_lm_yesno" "use_lm_value_model" "True"
check_log "4_lm_yesno" "LM-value yes_id="

# ── Phase 5: frozen-policy warmup ─────────────────────────────────────────────
# Use warmup_steps=10 so it fires within our 5-step smoke window (we run >5 steps due to async).
run_phase "5_warmup" \
    --value_warmup_steps 10
check_log "5_warmup" "Skipping weight sync.*warmup|value/warmup"

# ── Phase 6: init from pretrained checkpoint ──────────────────────────────────
PRETRAINED_CKPT=$(find "$OUTDIR/phase_5_warmup" -name "value_model.bin" | head -1 | xargs dirname 2>/dev/null)
if [ -z "$PRETRAINED_CKPT" ]; then
    echo ""
    echo "  ✗ Phase 6: no Phase 5 checkpoint found, skipping"
    RESULTS+=("SKIP  6_from_pretrained (no Phase 5 ckpt)")
else
    run_phase "6_from_pretrained" \
        --init_value_from_pretrained_checkpoint "$PRETRAINED_CKPT"
    check_log "6_from_pretrained" "Loaded pretrained value model from"
fi

# ── Phase 8: value estimation harness ────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " Starting Phase 8: value estimation"
echo "══════════════════════════════════════════"

SCALAR_BIN=$(find "$OUTDIR" -name "value_model.bin" | head -1)
SCALAR_CKPT_DIR=$(echo "$SCALAR_BIN" | xargs dirname 2>/dev/null)
VE_DATA=/tmp/vip_ve_data
mkdir -p "$VE_DATA"

if [ -z "$SCALAR_CKPT_DIR" ]; then
    echo "  ✗ No value model checkpoint found, skipping Phase 8"
    RESULTS+=("SKIP  8_value_estimation (no checkpoint)")
else
    # score_dataset uses AutoModelForCausalLM.from_pretrained which needs a full HF model dir.
    # Our value_model.bin has a scalar lm_head (shape [1, hidden]); create a compatible HF dir.
    SCALAR_HF_DIR=/tmp/vip_ve_scalar_hf
    SCALAR_PARENT_DIR=$(dirname "$SCALAR_CKPT_DIR")
    $VENV - << PYEOF
import torch, json, os, shutil
from safetensors.torch import save_file
ckpt = "$SCALAR_PARENT_DIR"
value_bin = "$SCALAR_BIN"
out_dir = "$SCALAR_HF_DIR"
shutil.rmtree(out_dir, ignore_errors=True)
os.makedirs(out_dir, exist_ok=True)
for f in ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
    src = os.path.join(ckpt, f)
    if os.path.exists(src): shutil.copy(src, out_dir)
sd = torch.load(value_bin, map_location="cpu", weights_only=True)
sd_mod = {k: (v[:1] if k == "model.embed_tokens.weight" else v).bfloat16() for k, v in sd.items()}
with open(os.path.join(ckpt, "config.json")) as f: cfg = json.load(f)
cfg["vocab_size"] = 1; cfg["tie_word_embeddings"] = False
with open(os.path.join(out_dir, "config.json"), "w") as f: json.dump(cfg, f, indent=2)
save_file(sd_mod, os.path.join(out_dir, "model.safetensors"))
wm = {k: "model.safetensors" for k in sd_mod}
with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": sum(v.numel()*2 for v in sd_mod.values())}, "weight_map": wm}, f)
print("Built HF value model dir:", out_dir)
PYEOF
    ray_start
    # Stage A: make_dataset (small: 5 pairs, 4 rollouts, no MC continuations for speed)
    $VENV -m open_instruct.value_estimation make_dataset \
        --model_name_or_path Qwen/Qwen3-0.6B \
        --output_path "$VE_DATA/pairs.parquet" \
        --dataset_name ai2-adapt-dev/rlvr_gsm8k_zs \
        --dataset_split train \
        --target_num_pairs 5 \
        --rollouts_per_prompt 4 \
        --probe_interval 99999 \
        --max_response_length 512 \
        --max_prompt_length 512 \
        --gpu_memory_utilization 0.5 \
        >"$LOGDIR/8_make_dataset.log" 2>&1
    A_RC=$?
    ray_stop

    if [ $A_RC -ne 0 ]; then
        echo "  ✗ Phase 8 Stage A (make_dataset) FAILED — see $LOGDIR/8_make_dataset.log"
        RESULTS+=("FAIL  8_value_estimation (make_dataset)")
    else
        echo "  ✓ make_dataset done"
        # Stage B: score with scalar value model
        $VENV -m open_instruct.value_estimation score_dataset \
            --input_dataset_path "$VE_DATA/pairs.parquet" \
            --output_path "$VE_DATA/scalar.parquet" \
            --value_model_path "$SCALAR_HF_DIR" \
            --tokenizer_name_or_path Qwen/Qwen3-0.6B \
            --value_model_type scalar \
            --run_name scalar_smoke \
            >"$LOGDIR/8_score_scalar.log" 2>&1
        B_RC=$?

        if [ $B_RC -ne 0 ]; then
            echo "  ✗ Phase 8 Stage B (score_dataset) FAILED — see $LOGDIR/8_score_scalar.log"
            RESULTS+=("FAIL  8_value_estimation (score_dataset)")
        else
            echo "  ✓ score_dataset done"
            # Stage C: compare
            $VENV -m open_instruct.value_estimation compare_runs \
                --score_dataset_paths "$VE_DATA/scalar.parquet" \
                --output_markdown_path "$VE_DATA/compare.md" \
                >"$LOGDIR/8_compare.log" 2>&1
            C_RC=$?
            if [ $C_RC -eq 0 ]; then
                echo "  ✓ compare_runs done"
                RESULTS+=("PASS  8_value_estimation")
                cat "$LOGDIR/8_compare.log" | head -30
            else
                echo "  ✗ compare_runs FAILED"
                RESULTS+=("FAIL  8_value_estimation (compare_runs)")
            fi
        fi
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " PHASE SUMMARY"
echo "══════════════════════════════════════════"
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo ""
