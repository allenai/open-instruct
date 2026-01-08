#!/usr/bin/env python
"""
Tool-enabled inference script for generating model completions.

This script loads a HuggingFace dataset or JSONL file, runs inference over the samples
using vLLM with optional tool support, and saves generations along with original samples.

Usage:
    # Basic usage without tools:
    python -m open_instruct.tools.tool_inference \
        --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
        --dataset allenai/tulu-3-sft-mixture \
        --split test \
        --output_file outputs/generations.jsonl

    # With tools:
    python -m open_instruct.tools.tool_inference \
        --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
        --dataset allenai/tulu-3-sft-mixture \
        --split test \
        --output_file outputs/generations.jsonl \
        --tools python \
        --tool_configs '{"api_endpoint": "http://localhost:8000/execute"}' \
        --tool_parser legacy

    # From JSONL file:
    python -m open_instruct.tools.tool_inference \
        --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
        --input_file data/prompts.jsonl \
        --output_file outputs/generations.jsonl
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset
from rich.pretty import pprint
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from open_instruct.tools.config import (
    TOOL_REGISTRY,
    ToolConfig,
    build_tools_from_config,
    create_tool_parser,
    get_available_parsers,
    get_available_tools,
    get_tool_definitions_from_config,
)
from open_instruct.tools.parsers import ToolParser
from open_instruct.tools.utils import Tool
from open_instruct.utils import ArgumentParserPlus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Args:
    """Main arguments for tool-enabled inference."""

    # Model configuration
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    """The model to use for generation."""
    revision: str = "main"
    """Model revision to use."""
    tokenizer_name_or_path: str | None = None
    """Tokenizer to use. Defaults to model_name_or_path if not specified."""
    trust_remote_code: bool = False
    """Whether to trust remote code when loading model/tokenizer."""

    # Input configuration
    dataset: str | None = None
    """HuggingFace dataset name or path. Mutually exclusive with input_file."""
    dataset_config: str | None = None
    """Dataset configuration name."""
    split: str = "test"
    """Dataset split to use."""
    input_file: str | None = None
    """Path to JSONL input file. Mutually exclusive with dataset."""
    messages_key: str = "messages"
    """Key in the dataset/file containing the messages list."""
    max_samples: int | None = None
    """Maximum number of samples to process. None means all."""
    start_idx: int = 0
    """Starting index for processing."""

    # Output configuration
    output_file: str = "outputs/generations.jsonl"
    """Path to save the output JSONL file."""
    save_every: int = 100
    """Save checkpoint every N samples."""

    # Generation configuration
    max_new_tokens: int = 2048
    """Maximum number of new tokens to generate."""
    temperature: float = 0.7
    """Sampling temperature."""
    top_p: float = 1.0
    """Top-p (nucleus) sampling parameter."""
    num_completions: int = 1
    """Number of completions to generate per prompt."""
    seed: int | None = None
    """Random seed for reproducibility."""

    # vLLM configuration
    tensor_parallel_size: int = 1
    """Number of GPUs to use for tensor parallelism."""
    max_model_len: int | None = None
    """Maximum model context length. None uses model default."""
    gpu_memory_utilization: float = 0.9
    """GPU memory utilization for vLLM."""
    enforce_eager: bool = False
    """Enforce eager execution (disable CUDA graphs)."""

    # Tool configuration
    tools: list[str] | None = None
    """List of tools to enable. Available: python, serper_search, massive_ds_search, s2_search, mcp."""
    tool_configs: list[str] | None = None
    """JSON configs for each tool (must match length of --tools)."""
    tool_parser: str = "legacy"
    """Tool parser type. Available: legacy, dr_tulu, vllm_hermes, vllm_llama3_json, vllm_qwen3_coder."""
    tool_tag_names: list[str] | None = None
    """Custom tag names for tools (must match length of --tools)."""
    max_tool_calls: int = 5
    """Maximum number of tool calls per generation."""
    max_concurrent: int = 64
    """Maximum number of concurrent async generations."""

    def __post_init__(self):
        # Validate input source
        if self.dataset is None and self.input_file is None:
            raise ValueError("Either --dataset or --input_file must be specified")
        if self.dataset is not None and self.input_file is not None:
            raise ValueError("Only one of --dataset or --input_file can be specified")

        # Validate tools
        if self.tools:
            available = get_available_tools()
            for tool in self.tools:
                if tool.lower() not in available:
                    raise ValueError(f"Unknown tool: {tool}. Available: {available}")

        # Validate tool_configs length
        if self.tool_configs is not None:
            if not self.tools:
                raise ValueError("--tool_configs requires --tools to be specified")
            if len(self.tool_configs) != len(self.tools):
                raise ValueError(
                    f"--tool_configs must have same length as --tools. "
                    f"Got {len(self.tool_configs)} configs for {len(self.tools)} tools."
                )

        # Validate tool_tag_names length
        if self.tool_tag_names is not None:
            if not self.tools:
                raise ValueError("--tool_tag_names requires --tools to be specified")
            if len(self.tool_tag_names) != len(self.tools):
                raise ValueError(
                    f"--tool_tag_names must have same length as --tools. "
                    f"Got {len(self.tool_tag_names)} for {len(self.tools)} tools."
                )

        # Validate tool parser
        if self.tool_parser not in get_available_parsers():
            raise ValueError(f"Unknown parser: {self.tool_parser}. Available: {get_available_parsers()}")


def load_data(args: Args) -> Dataset:
    """Load data from HuggingFace dataset or JSONL file."""
    if args.dataset:
        logger.info(f"Loading dataset: {args.dataset} (split: {args.split})")
        dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    else:
        logger.info(f"Loading JSONL file: {args.input_file}")
        with open(args.input_file) as f:
            data = [json.loads(line) for line in f]
        dataset = Dataset.from_list(data)

    # Apply slicing
    end_idx = args.start_idx + args.max_samples if args.max_samples else len(dataset)
    dataset = dataset.select(range(args.start_idx, min(end_idx, len(dataset))))

    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def build_tool_config(args: Args) -> ToolConfig:
    """Build ToolConfig from args."""
    if not args.tools:
        return ToolConfig(tools=None)

    # Parse tool configs
    parsed_configs = []
    if args.tool_configs:
        for i, config_str in enumerate(args.tool_configs):
            try:
                parsed_configs.append(json.loads(config_str))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in tool_configs[{i}]: {e}") from e
    else:
        parsed_configs = [{}] * len(args.tools)

    # Build tool configs dict
    tool_configs_dict = {name: config_cls() for name, config_cls in TOOL_REGISTRY.items()}
    for i, tool_name in enumerate(args.tools):
        tool_name_lower = tool_name.lower()
        config_cls = TOOL_REGISTRY[tool_name_lower]
        tool_configs_dict[tool_name_lower] = config_cls(**parsed_configs[i])

    return ToolConfig(
        tools=args.tools,
        max_tool_calls=args.max_tool_calls,
        parser=args.tool_parser,
        tool_tag_names=args.tool_tag_names,
        tool_configs=tool_configs_dict,
    )


def prepare_prompts(
    dataset: Dataset, tokenizer: PreTrainedTokenizer, args: Args, tool_definitions: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Prepare prompts by applying chat template."""
    prompts = []

    for idx, sample in enumerate(tqdm(dataset, desc="Preparing prompts")):
        messages = sample[args.messages_key]

        # Remove the last assistant message if present (we want to generate it)
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]

        # Apply chat template with tool definitions if available
        try:
            if tool_definitions:
                prompt = tokenizer.apply_chat_template(
                    messages, tools=tool_definitions, add_generation_prompt=True, tokenize=False
                )
            else:
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception as e:
            logger.warning(f"Failed to apply chat template for sample {idx}: {e}")
            # Fallback: just join messages
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        prompts.append(
            {"idx": idx, "prompt": prompt, "messages": sample[args.messages_key], "original_sample": dict(sample)}
        )

    return prompts


def execute_tool_call(
    text: str, tools: dict[str, Tool], tool_parser: ToolParser, max_tool_calls: int, num_calls: int
) -> tuple[str | None, dict[str, Any]]:
    """Execute a single tool call and return the formatted output.

    Returns:
        Tuple of (formatted_output, metadata) where formatted_output is None if no tool was called.
    """
    tool_calls = tool_parser.get_tool_calls(text)
    if not tool_calls:
        return None, {}

    tool_call = tool_calls[0]  # Process one at a time
    if tool_call.name not in tools:
        return None, {"error": f"Unknown tool: {tool_call.name}"}

    if num_calls >= max_tool_calls:
        return tool_parser.format_tool_calls("Max tool calls exceeded."), {
            "max_calls_exceeded": True,
            "tool_name": tool_call.name,
        }

    tool = tools[tool_call.name]
    try:
        # Handle both sync and async tools
        if asyncio.iscoroutinefunction(tool.__call__):
            result = asyncio.run(tool(**tool_call.args))
        else:
            result = tool(**tool_call.args)

        formatted_output = tool_parser.format_tool_calls(result.output)
        return formatted_output, {
            "tool_name": tool_call.name,
            "tool_args": tool_call.args,
            "tool_output": result.output,
            "tool_error": result.error,
            "tool_runtime": result.runtime,
            "tool_timeout": result.timeout,
        }
    except Exception as e:
        error_msg = f"Tool execution error: {e!s}"
        formatted_output = tool_parser.format_tool_calls(error_msg)
        return formatted_output, {"tool_name": tool_call.name, "tool_args": tool_call.args, "tool_error": error_msg}


async def generate_single_sample_with_tools(
    engine: AsyncLLMEngine,
    prompt_data: dict[str, Any],
    sampling_params: SamplingParams,
    tools: dict[str, Tool],
    tool_parser: ToolParser,
    max_tool_calls: int,
    pbar: Any,
) -> dict[str, Any]:
    """Generate completion for a single sample with tool support (async).

    Each sample runs as an independent async task, allowing true parallelism.
    """
    current_prompt = prompt_data["prompt"]
    full_generation = ""
    num_calls = 0
    tool_metadata = []

    while True:
        request_id = str(uuid.uuid4())

        # Generate asynchronously
        final_output = None
        async for output in engine.generate(current_prompt, sampling_params, request_id):
            final_output = output

        if final_output is None:
            break

        generation = final_output.outputs[0].text
        full_generation += generation
        finish_reason = final_output.outputs[0].finish_reason

        # Check if we hit a tool stop sequence
        if finish_reason == "stop" and tools:
            tool_output, metadata = execute_tool_call(generation, tools, tool_parser, max_tool_calls, num_calls)

            if tool_output:
                num_calls += 1
                tool_metadata.append(metadata)
                full_generation += tool_output
                current_prompt = current_prompt + generation + tool_output

                if num_calls >= max_tool_calls:
                    break
                continue

        # No more tool calls or max length reached
        break

    pbar.update(1)

    return {
        "idx": prompt_data["idx"],
        "prompt": prompt_data["prompt"],
        "messages": prompt_data["messages"],
        "generation": full_generation,
        "num_tool_calls": num_calls,
        "tool_metadata": tool_metadata,
        "original_sample": prompt_data["original_sample"],
    }


async def generate_with_tools_async(
    engine: AsyncLLMEngine,
    prompts: list[dict[str, Any]],
    sampling_params: SamplingParams,
    tools: dict[str, Tool],
    tool_parser: ToolParser,
    max_tool_calls: int,
    max_concurrent: int = 64,
) -> list[dict[str, Any]]:
    """Generate completions with tool support using async parallelism.

    Each sample runs as an independent async task. A semaphore limits concurrency
    to avoid overwhelming the engine.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_generate(prompt_data: dict[str, Any], pbar: Any) -> dict[str, Any]:
        async with semaphore:
            return await generate_single_sample_with_tools(
                engine, prompt_data, sampling_params, tools, tool_parser, max_tool_calls, pbar
            )

    with atqdm(total=len(prompts), desc="Generating") as pbar:
        tasks = [bounded_generate(p, pbar) for p in prompts]
        results = await asyncio.gather(*tasks)

    return list(results)


async def generate_single_sample_without_tools(
    engine: AsyncLLMEngine, prompt_data: dict[str, Any], sampling_params: SamplingParams, pbar: Any
) -> dict[str, Any]:
    """Generate completion for a single sample without tools (async)."""
    request_id = str(uuid.uuid4())

    final_output = None
    async for output in engine.generate(prompt_data["prompt"], sampling_params, request_id):
        final_output = output

    if final_output is None:
        generation = ""
    else:
        generations = [o.text for o in final_output.outputs]
        generation = generations[0] if len(generations) == 1 else generations

    pbar.update(1)

    return {
        "idx": prompt_data["idx"],
        "prompt": prompt_data["prompt"],
        "messages": prompt_data["messages"],
        "generation": generation,
        "num_tool_calls": 0,
        "tool_metadata": [],
        "original_sample": prompt_data["original_sample"],
    }


async def generate_without_tools_async(
    engine: AsyncLLMEngine, prompts: list[dict[str, Any]], sampling_params: SamplingParams, max_concurrent: int = 64
) -> list[dict[str, Any]]:
    """Generate completions without tool support using async parallelism."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_generate(prompt_data: dict[str, Any], pbar: Any) -> dict[str, Any]:
        async with semaphore:
            return await generate_single_sample_without_tools(engine, prompt_data, sampling_params, pbar)

    with atqdm(total=len(prompts), desc="Generating") as pbar:
        tasks = [bounded_generate(p, pbar) for p in prompts]
        results = await asyncio.gather(*tasks)

    return list(results)


def save_results(results: list[dict[str, Any]], output_file: str):
    """Save results to JSONL file."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Saved {len(results)} results to {output_file}")


async def main_async(args: Args):
    """Main inference function (async)."""
    pprint(args)

    # Load tokenizer
    tokenizer_path = args.tokenizer_name_or_path or args.model_name_or_path
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, revision=args.revision, trust_remote_code=args.trust_remote_code
    )

    # Build tool config
    tool_config = build_tool_config(args)

    # Build tools if configured
    tools: dict[str, Tool] = {}
    stop_strings: list[str] = []
    tool_definitions: list[dict[str, Any]] | None = None

    if tool_config.tools:
        logger.info(f"Building tools: {tool_config.tools}")
        tools, stop_strings = build_tools_from_config(tool_config)
        tool_definitions = get_tool_definitions_from_config(tool_config)
        logger.info(f"Tools ready: {list(tools.keys())}")
        logger.info(f"Stop strings: {stop_strings}")

    # Create tool parser
    tool_parser: ToolParser | None = None
    if tools:
        tool_parser = create_tool_parser(tool_config.parser, tokenizer=tokenizer, tools=tools)
        if tool_parser:
            stop_strings.extend(tool_parser.stop_sequences())
            stop_strings = list(set(stop_strings))

    # Load data
    dataset = load_data(args)

    # Prepare prompts
    prompts = prepare_prompts(dataset, tokenizer, args, tool_definitions)

    # Initialize async vLLM engine
    logger.info(f"Initializing async vLLM engine with model: {args.model_name_or_path}")
    engine_args = AsyncEngineArgs(
        model=args.model_name_or_path,
        revision=args.revision,
        tokenizer_revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Build sampling params
    sampling_params = SamplingParams(
        n=args.num_completions,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=stop_strings if stop_strings else None,
        include_stop_str_in_output=True,
        seed=args.seed,
    )
    logger.info(f"Sampling params: {sampling_params}")
    logger.info(f"Max concurrent requests: {args.max_concurrent}")

    # Generate
    start_time = time.time()
    if tools and tool_parser:
        results = await generate_with_tools_async(
            engine, prompts, sampling_params, tools, tool_parser, tool_config.max_tool_calls, args.max_concurrent
        )
    else:
        results = await generate_without_tools_async(engine, prompts, sampling_params, args.max_concurrent)
    elapsed = time.time() - start_time

    logger.info(f"Generation complete in {elapsed:.2f}s ({len(results) / elapsed:.2f} samples/sec)")

    # Save results
    save_results(results, args.output_file)

    # Print summary
    if tools:
        total_calls = sum(r["num_tool_calls"] for r in results)
        samples_with_calls = sum(1 for r in results if r["num_tool_calls"] > 0)
        logger.info(
            f"Tool usage summary: {total_calls} total calls across {samples_with_calls}/{len(results)} samples"
        )


def main(args: Args):
    """Entry point that runs the async main function."""
    asyncio.run(main_async(args))


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args,))
    args = parser.parse()
    main(args)
