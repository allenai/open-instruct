import argparse
import subprocess
import json
from openai import OpenAI

from beaker import Beaker

# New default task suites
BENCHMARKS = [
    # Knowledge
    ("popqa::hamish_zs_reasoning_deepseek", 1760),

    # Reasoning
    #("agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek", None),
    ("agi_eval_lsat-ar:0shot_cot::hamish_zs_reasoning_deepseek", None),
    ("agi_eval_lsat-lr:0shot_cot::hamish_zs_reasoning_deepseek", None),
    ("agi_eval_lsat-rc:0shot_cot::hamish_zs_reasoning_deepseek", None),

    # Math
    ("minerva_math_500::hamish_zs_reasoning_deepseek", None),
    
    # Coding
    ("codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek", None),
    
    # Chat / IF / Vibes
    ("alpaca_eval_v3::hamish_zs_reasoning_deepseek", None),
    ("ifeval::hamish_zs_reasoning_deepseek", None),
]

TOOL_USE_BENCHMARK = ("bfcl_all::std", None)
SAFETY_BENCHMARK = ("trustllm_jailbreaktrigger::default", None)
SAFETY_REASONING_BENCHMARK = ("trustllm_jailbreaktrigger::wildguard_reasoning_answer", None)


def test_tool_use_enabled(model_name, model_url, api_key_secret):
    # Tests whether the API endpoint has tool use enabled.

    api_key = None
    with Beaker.from_env(default_workspace="ai2/lm-eval") as beaker:
        api_key = beaker.secret.read(beaker.secret.get(api_key_secret))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "What's the weather like in Seattle?"},
    ]

    client = OpenAI(
        base_url=model_url,
        api_key=api_key,
    )

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        tools=tools,
        stream=False,
    )
    assert len(chat_completion.choices[0].message.tool_calls) > 0
    assert chat_completion.choices[0].message.tool_calls[0].function.name == "get_current_weather"
    print("Tool use enabled and working correctly.")


def test_identity(model_name, model_url, api_key_secret):
    api_key = None
    with Beaker.from_env(default_workspace="ai2/lm-eval") as beaker:
        api_key = beaker.secret.read(beaker.secret.get(api_key_secret))
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    client = OpenAI(
        base_url=model_url,
        api_key=api_key,
    )

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        stream=False,
    )
    assert "Olmo" in chat_completion.choices[0].message.content
    print("Identity test passed.")


def test_reasoning_parsing(model_name, model_url, api_key_secret):
    api_key = None
    with Beaker.from_env(default_workspace="ai2/lm-eval") as beaker:
        api_key = beaker.secret.read(beaker.secret.get(api_key_secret))
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    client = OpenAI(
        base_url=model_url,
        api_key=api_key,
    )

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        stream=False,
    )
    assert chat_completion.choices[0].message.content.reasoning is not None
    print("Reasoning parsing test passed.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate API Endpoints on Benchmark Suites")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to evaluate.",
    )
    parser.add_argument(
        "--model_url",
        type=str,
        required=True,
        help="URL of the model API endpoint.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["cirrascale", "modal", "parasail"],
        help="Provider of the model API endpoint. Currently supported: cirrascale, modal.",
    )
    parser.add_argument(
        "--api_key_secret",
        type=str,
        help="Secret name for the API key if not using one of the providers above.",
    )
    parser.add_argument(
        "--tests",
        required=True,
        nargs="+",
        choices=["benchmarks", "identity", "reasoning_parsing", "tool_use"],
        help="List of tests to run. `benchmarks` runs benchmark evaluations, `identity` checks whether the model identifies as Olmo, `reasoning_parsing` checks whether the reasoning traces are being parsed correctly, and `tool_use` checks whether tool use is enabled for the model.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["general", "tool_use", "safety", "reasoning_safety"],
        default=["general"],
        help="List of benchmarks to evaluate on. Defaults to general.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32768,
        help="Maximum generation length for the model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--beaker_workspace",
        type=str,
        help="Beaker workspace to use for evaluation runs. If not specified, oe-eval used ai2/lm-eval by default.",
    )
    parser.add_argument(
        "--gsheet",
        type=str,
        help="Google Sheet name to log results. See oe-eval documentation for authentication details.",
    )
    args = parser.parse_args()
    api_key_secret = None
    if args.provider:
        api_key_secret = f"{args.provider}_api_key"
    elif args.api_key_secret:
        api_key_secret = args.api_key_secret
    else:
        raise ValueError("Either --provider or --api_key_secret must be specified.")

    if "identity" in args.tests:
        print("\nTesting whether the model identifies as Olmo..")
        test_identity(args.model_name, args.model_url, api_key_secret)
    if "reasoning_parsing" in args.tests:
        print("\nTesting whether the reasoning traces are being parsed correctly..")
        test_reasoning_parsing(args.model_name, args.model_url, api_key_secret)
    if "tool_use" in args.tests:
        print("\nTesting whether tool use is enabled for the model..")
        test_tool_use_enabled(args.model_name, args.model_url, api_key_secret)
    if "benchmarks" in args.tests:
        print("\nRunning benchmark evaluations..")
        benchmarks = BENCHMARKS
        if "tool_use" in args.benchmarks:
            benchmarks.append(TOOL_USE_BENCHMARK)
        if "reasoning_safety" in args.benchmarks:
            benchmarks.append(SAFETY_REASONING_BENCHMARK)
        elif "safety" in args.benchmarks:
            benchmarks.append(SAFETY_BENCHMARK)
        for benchmark, limit in benchmarks:
            print(f"\n\n{benchmark}")
            print("=" * len(benchmark))
            # Run the evaluation command
            model_args = {
                "api_base_url": args.model_url,
                "max_length": args.max_length,
                "process_output": "r1_style",
            }
            command_list = [
                "uv",
                "run",
                "python",
                "oe-eval-internal/oe_eval/launch.py",
                f"--task {benchmark}",
                "--gpus 0",
                f"--model hosted_vllm/{args.model_name}",
                "--model-type litellm",
                f"--model-args '{json.dumps(model_args)}'"
            ]

            gantry_args = {
                "env-secret": "OPENAI_API_KEY=openai_api_key",
                "env-secret#1": f"LITELLM_API_KEY={api_key_secret}"
            }
            command_list.append(
                f"--gantry-args '{json.dumps(gantry_args)}'"
            )
            if limit:
                command_list.append(f"--limit {limit}")
            if args.max_length or args.temperature:
                generation_kwargs = {}
                if args.max_length:
                    generation_kwargs["max_gen_toks"] = args.max_length
                if args.temperature:
                    generation_kwargs["temperature"] = args.temperature
                task_args = {"chat_overrides": {"generation_kwargs": generation_kwargs}}
                command_list.append(f"--task-args '{json.dumps(task_args)}'")
            if args.beaker_workspace:
                command_list.append(f"--beaker-workspace {args.beaker_workspace}")
            if args.gsheet:
                command_list.append(f"--gsheet {args.gsheet}")
            command = " ".join(command_list)
            print("Command:\n", command)
            subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()