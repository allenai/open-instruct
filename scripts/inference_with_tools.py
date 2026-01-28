#!/usr/bin/env python3
"""
Inference script with tool calling support.

Supports two modes:
1. Batch mode: Process JSONL files
2. Interactive mode: Chat in terminal

Reuses the tool-calling infrastructure from open_instruct.

Example usage:
    # Batch mode
    python scripts/inference_with_tools.py \
        --input_file data.jsonl \
        --output_file results.jsonl \
        --model_name_or_path allenai/OLMo-2-1124-7B-Instruct \
        --tools python \
        --tool_parser_type vllm_olmo3

    # Interactive mode
    python scripts/inference_with_tools.py \
        --interactive \
        --model_name_or_path allenai/OLMo-2-1124-7B-Instruct \
        --tools python \
        --tool_parser_type vllm_olmo3
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any

import ray
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from open_instruct import logger_utils
from open_instruct.grpo_fast import create_tools
from open_instruct.tools.parsers import ToolParser, create_tool_parser
from open_instruct.tools.utils import ToolsConfig
from open_instruct.utils import ArgumentParserPlus

logger = logger_utils.setup_logger(__name__)


@dataclass
class InferenceConfig:
    model_name_or_path: str
    """Model to use for generation."""

    input_file: str | None = None
    """Input JSONL file (required for batch mode)."""

    output_file: str | None = None
    """Output JSONL file (required for batch mode)."""

    interactive: bool = False
    """Run in interactive chat mode."""

    system_prompt: str | None = None
    """System prompt for interactive mode."""

    model_revision: str | None = None
    messages_key: str = "messages"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    seed: int = 42
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    num_samples: int = 1
    """Number of samples to generate per prompt (for majority voting / pass@k)."""


@dataclass
class ToolsConfigArgs:
    tools: list[str] = field(default_factory=list)
    tool_call_names: list[str] = field(default_factory=list)
    tool_configs: list[str] = field(default_factory=list)
    tool_parser_type: str = "legacy"
    max_tool_calls: int = 5


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_tool_calls_batch(
    tool_calls_by_sample: dict[int, list],
    tool_actor_map: dict[str, ray.actor.ActorHandle],
    tool_parser: ToolParser,
    tokenizer,
    states: list[dict],
    max_tool_calls: int,
) -> None:
    """
    Execute tool calls in batch, mirroring the logic in vllm_utils.process_request().

    This reuses:
    - tool_actor.safe_execute.remote() for execution
    - tool_parser.format_tool_outputs() for formatting
    """
    if not tool_calls_by_sample:
        return

    # Submit all tool calls to Ray in parallel (like process_request does with await)
    pending: list[tuple[int, str, ray.ObjectRef]] = []
    for sample_idx, tool_calls in tool_calls_by_sample.items():
        state = states[sample_idx]
        for tc in tool_calls:
            if state["num_tool_calls"] >= max_tool_calls:
                state["tool_errors"].append("Max tool calls exceeded")
                continue
            state["num_tool_calls"] += 1
            # This mirrors: tool_result = await actor.tool_actor_map[tc.name].safe_execute.remote(**tc.args)
            future = tool_actor_map[tc.name].safe_execute.remote(**tc.args)
            pending.append((sample_idx, tc.name, future))

    # Collect results (like process_request collects tool_result)
    results_by_sample: dict[int, list[str]] = {i: [] for i in tool_calls_by_sample}
    for sample_idx, tool_name, future in pending:
        state = states[sample_idx]
        try:
            # This mirrors: tool_result: ToolOutput = await ...
            tool_result = ray.get(future)
            results_by_sample[sample_idx].append(tool_result.output)
            state["tool_outputs"].append(tool_result.output)
            if tool_result.error:
                state["tool_errors"].append(tool_result.error)
            if tool_result.timeout:
                state["tool_errors"].append(f"Tool {tool_name} timed out")
        except TypeError as e:
            # Mirrors the TypeError handling in process_request
            error_msg = f"Tool call '{tool_name}' failed: {e}"
            logger.warning(error_msg)
            results_by_sample[sample_idx].append(error_msg)
            state["tool_errors"].append(error_msg)

    # Format and append tool outputs (mirrors process_tool_tokens logic)
    for sample_idx, outputs in results_by_sample.items():
        if not outputs:
            continue
        state = states[sample_idx]
        # This mirrors: tool_parser.format_tool_outputs(outputs)
        formatted = tool_parser.format_tool_outputs(outputs)
        state["generated_text"] += formatted
        new_tokens = tokenizer.encode(formatted, add_special_tokens=False)
        state["current_tokens"].extend(new_tokens)
        state["remaining_tokens"] -= len(new_tokens)


def run_batched_generation(
    llm: LLM,
    tokenizer,
    initial_prompts: list[list[int]],
    sampling_params: SamplingParams,
    tool_parser: ToolParser | None,
    tool_actor_map: dict[str, ray.actor.ActorHandle],
    max_tool_calls: int,
) -> list[dict[str, Any]]:
    """
    Run batched generation with tool calling loop.

    This mirrors the structure of vllm_utils.process_request() but for batch processing:
    1. Generate for all active samples
    2. Parse tool calls using tool_parser.get_tool_calls()
    3. Execute tools via tool actors
    4. Format outputs with tool_parser.format_tool_outputs()
    5. Continue until no more tool calls or max reached
    """
    # Initialize states (like process_request initializes response tracking)
    states = [
        {
            "idx": i,
            "current_tokens": list(p),
            "generated_text": "",
            "num_tool_calls": 0,
            "tool_outputs": [],
            "tool_errors": [],
            "done": False,
            "finish_reason": "",
            "remaining_tokens": sampling_params.max_tokens,
        }
        for i, p in enumerate(initial_prompts)
    ]

    iteration = 0
    while True:
        active_indices = [i for i, s in enumerate(states) if not s["done"]]
        if not active_indices:
            break

        iteration += 1
        logger.info(f"Iteration {iteration}: {len(active_indices)} active samples")

        # Batch generate (like process_request calls actor.client.completions.create)
        active_tokens = [states[i]["current_tokens"] for i in active_indices]
        active_params = [
            SamplingParams(
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                max_tokens=states[i]["remaining_tokens"],
                seed=sampling_params.seed,
            )
            for i in active_indices
        ]

        # Pass token IDs as list of dicts (vLLM API format)
        prompts = [{"prompt_token_ids": tokens} for tokens in active_tokens]
        outputs = llm.generate(prompts, sampling_params=active_params)

        # Process outputs and collect tool calls
        tool_calls_by_sample: dict[int, list] = {}

        for idx, output in zip(active_indices, outputs):
            state = states[idx]
            gen_output = output.outputs[0]

            # Update state (like process_request updates response_tokens, current_prompt)
            state["generated_text"] += gen_output.text
            state["current_tokens"] = list(output.prompt_token_ids) + list(gen_output.token_ids)
            state["remaining_tokens"] -= len(gen_output.token_ids)
            state["finish_reason"] = gen_output.finish_reason

            # Check for tool calls (mirrors: tool_calls = actor.tool_parser.get_tool_calls(output.text))
            if tool_parser and tool_actor_map:
                tool_calls = tool_parser.get_tool_calls(gen_output.text)
                # Filter to allowed tools (mirrors: tool_calls = [tc for tc in tool_calls if tc.name in allowed_tools])
                tool_calls = [tc for tc in tool_calls if tc.name in tool_actor_map]
                if tool_calls:
                    tool_calls_by_sample[idx] = tool_calls
                else:
                    state["done"] = True
            else:
                state["done"] = True

            if state["remaining_tokens"] <= 0:
                state["done"] = True

        # Execute tool calls in batch
        if tool_calls_by_sample:
            logger.info(f"Executing tool calls for {len(tool_calls_by_sample)} samples...")
            process_tool_calls_batch(
                tool_calls_by_sample, tool_actor_map, tool_parser, tokenizer, states, max_tool_calls
            )

            # Mark samples as done if out of tokens after tool output
            for idx in tool_calls_by_sample:
                if states[idx]["remaining_tokens"] <= 0:
                    states[idx]["done"] = True

    return states


def run_interactive(
    llm: LLM,
    tokenizer,
    sampling_params: SamplingParams,
    tool_parser,
    tool_actor_map: dict,
    tool_definitions: list,
    max_tool_calls: int,
    system_prompt: str | None = None,
):
    """Run interactive chat mode."""
    print("\n" + "=" * 60)
    print("Interactive Chat Mode")
    print("=" * 60)
    print("Commands:")
    print("  /quit or /exit - Exit the chat")
    print("  /clear - Clear conversation history")
    print("  /system <prompt> - Set system prompt")
    print("=" * 60 + "\n")

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        print(f"[System prompt set: {system_prompt[:50]}...]\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ["/quit", "/exit"]:
            print("Goodbye!")
            break
        elif user_input.lower() == "/clear":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            print("[Conversation cleared]\n")
            continue
        elif user_input.lower().startswith("/system "):
            new_system = user_input[8:].strip()
            # Update or add system message
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = new_system
            else:
                messages.insert(0, {"role": "system", "content": new_system})
            print(f"[System prompt updated]\n")
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Tokenize
        tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tools=tool_definitions or None
        )

        # Generate with tool calling
        states = run_batched_generation(
            llm=llm,
            tokenizer=tokenizer,
            initial_prompts=[tokens],
            sampling_params=sampling_params,
            tool_parser=tool_parser,
            tool_actor_map=tool_actor_map,
            max_tool_calls=max_tool_calls,
        )

        response = states[0]["generated_text"]
        num_tool_calls = states[0]["num_tool_calls"]

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response})

        # Print response
        print(f"\nAssistant: {response}")
        if num_tool_calls > 0:
            print(f"[Used {num_tool_calls} tool call(s)]")
        print()


def main(config: InferenceConfig, tools_config: ToolsConfigArgs):
    # Validate config
    if not config.interactive and (not config.input_file or not config.output_file):
        raise ValueError("--input_file and --output_file are required for batch mode. Use --interactive for chat mode.")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, revision=config.model_revision)

    # Initialize tools using open_instruct infrastructure
    tool_actors = []
    tool_definitions = []
    tool_actor_map = {}
    tool_parser = None

    if tools_config.tools:
        ray.init(ignore_reinit_error=True)

        # Reuse create_tools() from grpo_fast.py
        full_config = ToolsConfig(
            tools=tools_config.tools,
            tool_call_names=tools_config.tool_call_names or tools_config.tools,
            tool_configs=tools_config.tool_configs or ["{}"] * len(tools_config.tools),
            tool_parser_type=tools_config.tool_parser_type,
            max_tool_calls=tools_config.max_tool_calls,
        )
        tool_actors, _ = create_tools(full_config._parsed_tools)

        # Get tool definitions and build actor map (like LLMRayActor._init_config)
        tool_definitions = ray.get([a.get_openai_tool_definitions.remote() for a in tool_actors])
        call_names = ray.get([a.get_call_name.remote() for a in tool_actors])
        tool_actor_map = dict(zip(call_names, tool_actors))

        # Reuse create_tool_parser() from tools/parsers.py (like LLMRayActor._init_tool_parser)
        tool_parser = create_tool_parser(tools_config.tool_parser_type, tokenizer, tool_actors, tool_definitions)
        logger.info(f"Tools initialized: {call_names}")

    # Determine max_model_len based on mode
    if config.interactive:
        # For interactive mode, use a reasonable default
        max_model_len = 8192 + config.max_tokens
        logger.info(f"Interactive mode: max_model_len={max_model_len}")
        input_data = None
        prompts = None
    else:
        # For batch mode, calculate from input data
        logger.info(f"Loading {config.input_file}")
        input_data = load_jsonl(config.input_file)
        logger.info(f"Loaded {len(input_data)} samples")

        logger.info("Preparing prompts...")
        prompts = []
        for sample in input_data:
            messages = sample[config.messages_key]
            if len(messages) > 1 and messages[-1]["role"] == "assistant":
                messages = messages[:-1]
            tokens = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tools=tool_definitions or None
            )
            prompts.append(tokens)

        max_prompt_len = max(len(p) for p in prompts)
        max_model_len = max_prompt_len + config.max_tokens
        logger.info(f"Max prompt length: {max_prompt_len}")

    logger.info(f"Creating vLLM engine (max_model_len={max_model_len})...")
    llm = LLM(
        model=config.model_name_or_path,
        revision=config.model_revision,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=config.seed,
    )

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        seed=config.seed,
    )

    if config.interactive:
        # Interactive chat mode
        run_interactive(
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            tool_parser=tool_parser,
            tool_actor_map=tool_actor_map,
            tool_definitions=tool_definitions,
            max_tool_calls=tools_config.max_tool_calls,
            system_prompt=config.system_prompt,
        )
    else:
        # Batch mode
        # Duplicate prompts for num_samples > 1
        if config.num_samples > 1:
            logger.info(f"Duplicating prompts {config.num_samples}x for multi-sampling...")
            expanded_prompts = []
            sample_indices = []  # Track which original sample each prompt came from
            for i, prompt in enumerate(prompts):
                for _ in range(config.num_samples):
                    expanded_prompts.append(prompt)
                    sample_indices.append(i)
            prompts = expanded_prompts
            logger.info(f"Total prompts after expansion: {len(prompts)}")

        logger.info("Running batched generation...")
        states = run_batched_generation(
            llm=llm,
            tokenizer=tokenizer,
            initial_prompts=prompts,
            sampling_params=sampling_params,
            tool_parser=tool_parser,
            tool_actor_map=tool_actor_map,
            max_tool_calls=tools_config.max_tool_calls,
        )

        # Build output
        logger.info("Building output...")
        output_data = []

        if config.num_samples > 1:
            # Group states by original sample
            from collections import defaultdict
            grouped_states = defaultdict(list)
            for state, orig_idx in zip(states, sample_indices):
                grouped_states[orig_idx].append(state)

            for i, sample in enumerate(input_data):
                out = dict(sample)
                sample_states = grouped_states[i]
                # Store all samples as lists
                out["generated_responses"] = [s["generated_text"] for s in sample_states]
                out["finish_reasons"] = [s["finish_reason"] for s in sample_states]
                out["num_tool_calls"] = [s["num_tool_calls"] for s in sample_states]
                out["tool_outputs"] = [s["tool_outputs"] for s in sample_states]
                out["tool_errors"] = [s["tool_errors"] for s in sample_states]
                out["num_samples"] = config.num_samples
                output_data.append(out)
        else:
            for sample, state in zip(input_data, states):
                out = dict(sample)
                out["generated_response"] = state["generated_text"]
                out["finish_reason"] = state["finish_reason"]
                out["num_tool_calls"] = state["num_tool_calls"]
                if state["tool_outputs"]:
                    out["tool_outputs"] = state["tool_outputs"]
                if state["tool_errors"]:
                    out["tool_errors"] = state["tool_errors"]
                output_data.append(out)

        logger.info(f"Saving to {config.output_file}")
        save_jsonl(output_data, config.output_file)

    # Cleanup
    for actor in tool_actors:
        ray.kill(actor)
    if tool_actors:
        ray.shutdown()

    logger.info("Done!")


if __name__ == "__main__":
    parser = ArgumentParserPlus((InferenceConfig, ToolsConfigArgs))
    main(*parser.parse())
