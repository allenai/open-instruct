#!/bin/bash
# Launch a vLLM + Gradio chat demo on Beaker for a Weka checkpoint.
#
# Usage:
#   bash scripts/serve/vllm_gradio_demo.sh
#
# Override defaults with env vars:
#   MODEL_PATH=/weka/oe-adapt-default/nathanl/checkpoints/MY_MODEL/step1234-hf \
#   TENSOR_PARALLEL=1 \
#   MAX_MODEL_LEN=16384 \
#     bash scripts/serve/vllm_gradio_demo.sh

MODEL_PATH="${MODEL_PATH:-/weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR2.5e-5/step46412-hf-tokenizer-fix}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
IMAGE="${IMAGE:-yanhongl/oe_eval_olmo3_devel_v6}"
CLUSTER="${CLUSTER:-ai2/jupiter}"

echo "============================================"
echo "Launching vLLM Gradio demo"
echo "  Image:   $IMAGE"
echo "  Model:   $MODEL_PATH"
echo "  TP:      $TENSOR_PARALLEL"
echo "  MaxLen:  $MAX_MODEL_LEN"
echo "  Cluster: $CLUSTER"
echo "============================================"

# Base64-encode the gradio chat script so it survives mason's argument parsing.
GRADIO_B64=$(base64 <<'PYEOF' | tr -d '\n'
import time, sys, gradio as gr
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
for i in range(180):
    try:
        model_name = client.models.list().data[0].id; break
    except Exception:
        if i % 10 == 0: print(f"Waiting for vLLM... ({i}s)", flush=True)
        time.sleep(1)
else:
    sys.exit("vLLM server did not start within 180s")
print(f"Connected to vLLM. Model: {model_name}", flush=True)
def chat(message, history):
    messages = []
    for m in history:
        if isinstance(m, dict):
            messages.append(dict(role=m["role"], content=m["content"]))
        else:
            messages.append(dict(role="user", content=m[0]))
            if m[1]: messages.append(dict(role="assistant", content=m[1]))
    messages.append(dict(role="user", content=message))
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=0.6, top_p=0.95, stream=True)
    partial = ""
    for chunk in resp:
        delta = chunk.choices[0].delta
        if delta.content:
            partial += delta.content
            yield partial
demo = gr.ChatInterface(fn=chat, title=f"vLLM Demo: {model_name}")
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
print(f"\n{'='*60}", flush=True)
print(f"GRADIO SHARE URL: {demo.share_url}", flush=True)
print(f"GRADIO LOCAL URL: {demo.local_url}", flush=True)
print(f"{'='*60}\n", flush=True)
import threading; threading.Event().wait()
PYEOF
)

uv run python mason.py \
    --cluster "$CLUSTER" \
    --image "$IMAGE" \
    --description "vLLM Gradio demo: $(basename $MODEL_PATH)" \
    --pure_docker_mode \
    \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 1 \
    --max_retries 0 \
    --timeout 24h \
    --budget ai2/oe-adapt \
    --gpus "$TENSOR_PARALLEL" \
    --no_auto_dataset_cache \
    --non_resumable \
    --env PYTHONUNBUFFERED=1 \
    -- uv pip install gradio openai \&\& \
       echo "$GRADIO_B64" \| base64 -d \> /tmp/gradio_chat.py \; \
       python -m vllm.entrypoints.openai.api_server \
         --model "$MODEL_PATH" \
         --tensor-parallel-size "$TENSOR_PARALLEL" \
         --max-model-len "$MAX_MODEL_LEN" \
         --trust-remote-code \
         --enforce-eager \
         --disable-cascade-attn \
         --host 0.0.0.0 \
         --port 8000 \& \
       python -u /tmp/gradio_chat.py
