"""Simple Gradio chat UI that connects to a local vLLM OpenAI-compatible server."""

import argparse
import time

import gradio as gr
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-host", default="http://localhost:8000")
    parser.add_argument("--model-path", default=None, help="Display name (cosmetic only)")
    args = parser.parse_args()

    client = OpenAI(base_url=f"{args.vllm_host}/v1", api_key="EMPTY")

    # Wait for vLLM to be ready
    for i in range(120):
        try:
            models = client.models.list()
            model_name = models.data[0].id
            break
        except Exception:
            if i % 10 == 0:
                print(f"Waiting for vLLM server to be ready... ({i}s)")
            time.sleep(1)
    else:
        raise RuntimeError("vLLM server did not start within 120s")

    print(f"Connected to vLLM. Model: {model_name}")

    def chat(message, history):
        messages = []
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=32768,
            temperature=0.6,
            top_p=0.95,
            stream=True,
        )
        partial = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                partial += chunk.choices[0].delta.content
                yield partial

    display = args.model_path or model_name
    demo = gr.ChatInterface(
        fn=chat,
        title=f"vLLM Demo: {display}",
    )
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    main()
