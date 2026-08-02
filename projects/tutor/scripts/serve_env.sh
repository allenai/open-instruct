#!/bin/bash
# Serve the two frozen models the tutoring reward talks to.
#
# The student (the environment) and the judge (the reward) both run behind
# OpenAI-compatible endpoints rather than inside the trainer. That is not a
# convenience: the environment lives in a Ray actor pool with one actor per
# concurrent rollout, and loading a model in each of those is not an option.
# An endpoint batches across all of them and can be swapped without touching
# the training job.
#
#   ./serve_env.sh                    # student on 8001, judge on 8002
#   STUDENT_GPU=1 JUDGE_GPU=2 ./serve_env.sh
#
# One 7B can serve both roles - point JUDGE_URL at the student's port and skip
# the second server. The student must stay at 0.5B if you want the anchor to
# remain comparable with the earlier runs.

set -euo pipefail

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
STUDENT_PORT=${STUDENT_PORT:-8001}
JUDGE_PORT=${JUDGE_PORT:-8002}
STUDENT_GPU=${STUDENT_GPU:-1}
JUDGE_GPU=${JUDGE_GPU:-2}
LOGS=${LOGS:-logs}

mkdir -p "$LOGS"

echo "student: $STUDENT_MODEL on :$STUDENT_PORT (GPU $STUDENT_GPU)"
CUDA_VISIBLE_DEVICES=$STUDENT_GPU vllm serve "$STUDENT_MODEL" \
    --port "$STUDENT_PORT" \
    --gpu-memory-utilization 0.25 \
    --max-model-len 2048 \
    > "$LOGS/student.log" 2>&1 &

echo "judge:   $JUDGE_MODEL on :$JUDGE_PORT (GPU $JUDGE_GPU)"
CUDA_VISIBLE_DEVICES=$JUDGE_GPU vllm serve "$JUDGE_MODEL" \
    --port "$JUDGE_PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    > "$LOGS/judge.log" 2>&1 &

echo "waiting for both to come up..."
for port in "$STUDENT_PORT" "$JUDGE_PORT"; do
    until curl -sf "http://localhost:$port/v1/models" > /dev/null; do sleep 5; done
    echo "  :$port ready"
done

wait
