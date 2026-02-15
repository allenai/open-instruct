#!/usr/bin/env python
"""Standalone inference with sandbox environments.

Supports interactive REPL and batch processing with concurrent multi-sample
generation. Uses a vLLM server for efficient concurrent inference and a pool
of sandbox environments for parallel tool execution.

Examples:
  # Interactive
  python scripts/sandbox_inference.py --model_name_or_path Qwen/Qwen3-4B

  # Batch with 32 samples, 16 concurrent envs
  python scripts/sandbox_inference.py --model_name_or_path Qwen/Qwen3-4B \\
    --input_file HuggingFaceH4/aime_2024 --output_file aime24.jsonl \\
    --num_samples 32 --env_pool_size 16

  # Connect to existing vLLM server
  python scripts/sandbox_inference.py --model_name_or_path Qwen/Qwen3-4B \\
    --api_base http://localhost:8000/v1 --input_file tasks.jsonl
"""

import argparse
import asyncio
import concurrent.futures
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import openai
from transformers import AutoTokenizer

from open_instruct import logger_utils
from open_instruct.environments.base import get_env_class
from open_instruct.tools.parsers import create_vllm_parser

logger = logger_utils.setup_logger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Sandbox inference with vLLM")

    # Model
    parser.add_argument("--model_name_or_path", required=True, help="Model to load")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None, help="Max context length (auto if not set)")

    # Server
    parser.add_argument("--api_base", default=None, help="Connect to existing vLLM server (skip auto-start)")
    parser.add_argument("--port", type=int, default=None, help="Port for auto-started vLLM server (auto if not set)")

    # Environment
    parser.add_argument("--env_name", default="sandbox_lm", help="Registered environment name")
    parser.add_argument("--env_backend", default="docker", help="Backend: docker, e2b, daytona")
    parser.add_argument("--env_max_steps", type=int, default=30, help="Max tool calls per episode")
    parser.add_argument("--env_timeout", type=int, default=120, help="Timeout per env operation (seconds)")
    parser.add_argument("--env_pool_size", type=int, default=None, help="Concurrent envs (default: num_samples)")
    parser.add_argument("--env_image", default=None, help="Docker image override (default: ubuntu:24.04)")

    # I/O
    parser.add_argument("--input_file", default=None, help="JSONL file or HF dataset name (omit for interactive)")
    parser.add_argument("--dataset_split", default="train", help="HF dataset split (default: train)")
    parser.add_argument("--output_file", default="results.jsonl", help="Output JSONL path")

    # Generation
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max total response tokens across all turns")
    parser.add_argument("--num_samples", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--tool_parser_type", default="vllm_hermes")

    # Prompt
    parser.add_argument("--system_prompt", default=None, help="System prompt text, or @filepath to load from file")

    # Timeout / retry
    parser.add_argument("--episode_timeout", type=int, default=600, help="Per-episode timeout in seconds (default: 600)")
    parser.add_argument("--max_retries", type=int, default=2, help="Max retries per episode on timeout (default: 2)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_system_prompt(value):
    """Load system prompt from text or from file (prefix with @)."""
    if value is None:
        return None
    if value.startswith("@"):
        return Path(value[1:]).read_text().strip()
    return value


def load_inputs(input_file, dataset_split):
    """Load inputs from JSONL file or HuggingFace dataset."""
    path = Path(input_file)
    if path.exists():
        inputs = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    inputs.append(json.loads(line))
        return inputs

    # Try as HF dataset
    import datasets

    logger.info(f"Loading HF dataset: {input_file} (split={dataset_split})")
    ds = datasets.load_dataset(input_file, split=dataset_split)
    return [dict(row) for row in ds]


def get_prompt_text(item):
    """Extract prompt text from an item, auto-detecting the field name."""
    if "messages" in item:
        return None  # Use messages directly
    for field in ("prompt", "problem", "question", "Question"):
        if field in item:
            return item[field]
    return None


def get_answer(item):
    """Extract ground truth answer from an item, auto-detecting the field name."""
    for field in ("answer", "Answer", "solution"):
        if field in item:
            return str(item[field]).strip()
    return None


def extract_boxed(text):
    r"""Extract the last \boxed{...} value from model output."""
    # Handle nested braces
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_sync(coro):
    """Run an async coroutine synchronously (for use with asyncio.to_thread)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Environment pool (no Ray)
# ---------------------------------------------------------------------------
class EnvPool:
    """Async pool of environment instances with acquire/release semantics."""

    def __init__(self, env_class, pool_size, **env_kwargs):
        self._envs = [env_class(**env_kwargs) for _ in range(pool_size)]
        self._queue: asyncio.Queue | None = None
        self.pool_size = pool_size

    async def initialize(self):
        self._queue = asyncio.Queue()
        for env in self._envs:
            await asyncio.to_thread(_run_sync, env.setup())
            self._queue.put_nowait(env)
        logger.info(f"Initialized env pool with {self.pool_size} instances")

    async def acquire(self):
        return await self._queue.get()

    def release(self, env):
        self._queue.put_nowait(env)

    async def shutdown(self):
        for env in self._envs:
            try:
                await asyncio.to_thread(_run_sync, env.shutdown())
            except Exception as e:
                logger.warning(f"Error shutting down env: {e}")
        logger.info("Env pool shut down")


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------
def start_vllm_server(model_path, port, tp_size, gpu_mem_util, max_model_len=None, dtype="bfloat16"):
    """Start a vLLM OpenAI-compatible server as a subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp_size),
        "--gpu-memory-utilization",
        str(gpu_mem_util),
        "--dtype",
        dtype,
        "--disable-log-requests",
    ]
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])

    logger.info(f"Starting vLLM server on port {port}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


async def wait_for_server(base_url, timeout=600):
    """Poll the vLLM server until it's ready, return an AsyncOpenAI client."""
    client = openai.AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=3600)
    start = time.time()
    while time.time() - start < timeout:
        try:
            await client.models.list()
            return client
        except Exception:
            await asyncio.sleep(3)
    raise TimeoutError(f"vLLM server not ready after {timeout}s")


# ---------------------------------------------------------------------------
# Episode execution
# ---------------------------------------------------------------------------
async def run_episode(client, model_name, tokenizer, tool_parser, env, messages, temperature, max_tokens, max_steps):
    """Run a single multi-turn episode with async generation + env execution.

    Returns the full assistant response text (all turns concatenated).
    """
    reset_result = await asyncio.to_thread(_run_sync, env.reset())
    tool_defs = reset_result.tools or env.get_tool_definitions()

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=tool_defs,
        add_generation_prompt=True,
        tokenize=False,
    )

    stop = tool_parser.stop_sequences or None
    all_text = ""
    step = 0
    response_tokens_used = 0

    while step < max_steps:
        remaining = max_tokens - response_tokens_used
        if remaining <= 0:
            break

        # Async generation — vLLM server handles batching across concurrent episodes
        response = await client.completions.create(
            model=model_name,
            prompt=prompt_text,
            temperature=temperature,
            max_tokens=remaining,
            n=1,
            stop=stop,
            extra_body={
                "skip_special_tokens": False,
                "include_stop_str_in_output": True,
            },
        )
        gen_text = response.choices[0].text
        response_tokens_used += response.usage.completion_tokens

        tool_calls = tool_parser.get_tool_calls(gen_text)
        all_text += gen_text

        if not tool_calls:
            break

        # Execute tool calls (in thread to avoid blocking event loop)
        observations = []
        done = False
        for tc in tool_calls:
            step += 1
            if step > max_steps:
                break
            result = await asyncio.to_thread(_run_sync, env.step(tc))
            observations.append(result.observation)
            if result.done:
                done = True
                break

        if observations:
            formatted = tool_parser.format_tool_outputs(observations)
            prompt_text += gen_text + formatted
            all_text += formatted
            # Count tool output tokens toward budget (they grow the context)
            response_tokens_used += len(tokenizer.encode(formatted, add_special_tokens=False))

        if done:
            break

    return all_text


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------
async def interactive_mode(client, model_name, tokenizer, tool_parser, pool, temperature, max_tokens, max_steps, system_prompt):
    """Interactive REPL: type a message, the model responds using sandbox tools."""
    print("=" * 60)
    print("Sandbox Inference — Interactive Mode")
    print("Type a message to chat. Commands: 'quit', 'reset'")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            print("Episode reset.")
            continue

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_input})

        env = await pool.acquire()
        try:
            response = await run_episode(
                client, model_name, tokenizer, tool_parser, env, messages, temperature, max_tokens, max_steps
            )
        finally:
            pool.release(env)

        print(f"\nAssistant:\n{response}")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------
async def batch_mode(
    client, model_name, tokenizer, tool_parser, pool, temperature, max_tokens, max_steps,
    input_file, dataset_split, output_file, num_samples, system_prompt,
    episode_timeout=600, max_retries=2,
):
    """Process inputs concurrently with multiple samples per prompt."""
    inputs = load_inputs(input_file, dataset_split)
    total_episodes = len(inputs) * num_samples
    logger.info(f"Loaded {len(inputs)} inputs x {num_samples} samples = {total_episodes} episodes")

    # Prepare messages for each input
    all_messages = []
    for item in inputs:
        prompt_text = get_prompt_text(item)
        if "messages" in item:
            messages = list(item["messages"])
        elif prompt_text:
            messages = [{"role": "user", "content": prompt_text}]
        else:
            logger.warning(f"Skipping item: no recognized prompt field. Keys: {list(item.keys())}")
            all_messages.append(None)
            continue

        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + [
                m for m in messages if m["role"] != "system"
            ]
        all_messages.append(messages)

    # Progress tracking
    completed = [0]
    start_time = time.time()

    async def run_one(messages):
        for attempt in range(1 + max_retries):
            env = await pool.acquire()
            try:
                result = await asyncio.wait_for(
                    run_episode(
                        client, model_name, tokenizer, tool_parser, env, messages, temperature, max_tokens, max_steps
                    ),
                    timeout=episode_timeout,
                )
                return result
            except asyncio.TimeoutError:
                logger.warning(
                    f"Episode timed out after {episode_timeout}s (attempt {attempt + 1}/{1 + max_retries})"
                )
                # Force-close the backend so the next reset() starts fresh
                try:
                    await asyncio.to_thread(_run_sync, env.close())
                except Exception:
                    pass
                if attempt < max_retries:
                    continue
                logger.error("Episode failed after all retries")
                return None
            finally:
                pool.release(env)

        return None  # unreachable, but satisfies type checkers

    async def run_one_tracked(messages):
        result = await run_one(messages)
        completed[0] += 1
        if completed[0] % 10 == 0 or completed[0] == total_episodes:
            elapsed = time.time() - start_time
            rate = completed[0] / elapsed
            logger.info(f"  Episodes: {completed[0]}/{total_episodes} ({rate:.1f}/s)")
        return result

    # Resume support: load already-completed items from output file
    results = []
    done_indices = set()
    output_path = Path(output_file)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
        # Match completed results back to input indices by prompt content
        done_prompts = set()
        for r in results:
            key = get_prompt_text(r) or json.dumps(r.get("messages", ""))
            done_prompts.add(key)
        for idx, item in enumerate(inputs):
            key = get_prompt_text(item) or json.dumps(item.get("messages", ""))
            if key in done_prompts:
                done_indices.add(idx)
        if done_indices:
            logger.info(f"Resuming: {len(done_indices)}/{len(inputs)} items already completed")
            completed[0] = len(done_indices) * num_samples

    # Process one item at a time to limit peak memory (responses can be large).
    # Within each item, run num_samples concurrently (bounded by pool size).
    with open(output_file, "a") as out_f:
        for idx, (item, messages) in enumerate(zip(inputs, all_messages)):
            if messages is None or idx in done_indices:
                continue

            samples = await asyncio.gather(*[run_one_tracked(messages) for _ in range(num_samples)])
            # Filter out timed-out episodes (None) from responses
            valid_samples = [s for s in samples if s is not None]
            result = {**item, "responses": valid_samples, "num_timed_out": len(samples) - len(valid_samples)}

            # Score if ground truth available (only count valid samples)
            gt = get_answer(item)
            if gt is not None and valid_samples:
                correct = sum(1 for s in valid_samples if _check_answer(s, gt))
                result["num_correct"] = correct
                result["accuracy"] = correct / len(valid_samples)

            results.append(result)
            # Write incrementally so we don't lose progress on crash
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            n_timed_out = result.get("num_timed_out", 0)
            timeout_str = f" ({n_timed_out} timed out)" if n_timed_out else ""
            logger.info(
                f"Item {idx + 1}/{len(inputs)}: {result.get('accuracy', 'N/A')}"
                f" ({len(valid_samples)}/{num_samples} samples{timeout_str})"
            )

    # Print summary
    elapsed = time.time() - start_time
    logger.info(f"Saved {len(results)} results to {output_file} in {elapsed:.0f}s")

    scored = [r for r in results if "accuracy" in r]
    if scored:
        avg_acc = sum(r["accuracy"] for r in scored) / len(scored)
        total_correct = sum(r["num_correct"] for r in scored)
        total_samples = sum(len(r["responses"]) for r in scored)
        logger.info(f"Avg score @ {num_samples}: {avg_acc:.4f} ({total_correct}/{total_samples})")
        # Per-problem breakdown
        for r in scored:
            label = r.get("id", r.get("problem_idx", ""))
            gt = get_answer(r)
            n_valid = len(r["responses"])
            timed_out = r.get("num_timed_out", 0)
            extra = f", {timed_out} timed out" if timed_out else ""
            logger.info(f"  {label}: {r['num_correct']}/{n_valid} (answer={gt}{extra})")


def _check_answer(response, ground_truth):
    """Check if a model response contains the correct answer using math-verify."""
    predicted = extract_boxed(response)
    if predicted is None:
        return False
    try:
        from math_verify import parse, verify

        gold = parse(str(ground_truth))
        pred = parse(f"\\boxed{{{predicted}}}")
        return verify(gold, pred)
    except Exception:
        # Fallback to simple comparison
        try:
            return math.isclose(float(predicted), float(ground_truth), rel_tol=1e-6)
        except (ValueError, TypeError):
            return predicted.strip() == ground_truth.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    args = parse_args()
    system_prompt = load_system_prompt(args.system_prompt)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Get environment class and tool definitions
    env_class = get_env_class(env_name=args.env_name)
    tool_defs = env_class.get_tool_definitions()

    # Create tool parser
    tool_parser = create_vllm_parser(args.tool_parser_type, tokenizer, tool_definitions=tool_defs)

    # Start or connect to vLLM server
    server_proc = None
    if args.api_base:
        base_url = args.api_base
    else:
        port = args.port or _get_free_port()
        base_url = f"http://localhost:{port}/v1"
        server_proc = start_vllm_server(
            args.model_name_or_path,
            port,
            args.tensor_parallel_size,
            args.gpu_memory_utilization,
            args.max_model_len,
        )

    try:
        logger.info(f"Waiting for vLLM server at {base_url} ...")
        client = await wait_for_server(base_url)
        models = await client.models.list()
        model_name = models.data[0].id
        logger.info(f"vLLM server ready, model={model_name}")

        # Create env pool
        pool_size = args.env_pool_size or args.num_samples
        env_kwargs = {"backend": args.env_backend, "timeout": args.env_timeout}
        if args.env_image:
            env_kwargs["image"] = args.env_image
        pool = EnvPool(env_class, pool_size, **env_kwargs)
        await pool.initialize()

        # Increase default thread pool so all envs can run concurrently
        default_workers = min(32, (os.cpu_count() or 1) + 4)
        if pool_size > default_workers:
            asyncio.get_event_loop().set_default_executor(
                concurrent.futures.ThreadPoolExecutor(max_workers=pool_size)
            )

        try:
            if args.input_file:
                await batch_mode(
                    client, model_name, tokenizer, tool_parser, pool,
                    args.temperature, args.max_tokens, args.env_max_steps,
                    args.input_file, args.dataset_split, args.output_file,
                    args.num_samples, system_prompt,
                    episode_timeout=args.episode_timeout, max_retries=args.max_retries,
                )
            else:
                await interactive_mode(
                    client, model_name, tokenizer, tool_parser, pool,
                    args.temperature, args.max_tokens, args.env_max_steps, system_prompt,
                )
        finally:
            await pool.shutdown()
    finally:
        if server_proc is not None:
            logger.info("Shutting down vLLM server")
            server_proc.terminate()
            server_proc.wait(timeout=30)


if __name__ == "__main__":
    asyncio.run(main())
