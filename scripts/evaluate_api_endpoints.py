import argparse
import subprocess
import json


# New default task suites
BENCHMARKS = [
    # Knowledge
    ("popqa::hamish_zs_reasoning_deepseek", 1760),

    # Reasoning
    #("agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek", None),
    ("agi_eval_english_lsat-ar:0shot_cot::hamish_zs_reasoning_deepseek", None),
    ("agi_eval_english_lsat-lr:0shot_cot::hamish_zs_reasoning_deepseek", None),
    ("agi_eval_english_lsat-rc:0shot_cot::hamish_zs_reasoning_deepseek", None),

    # Math
    ("minerva_math_500::hamish_zs_reasoning", None),
    #("aime:zs_cot_r1::pass_at_32_2025_deepseek", 420),
    
    # Coding
    ("codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek", None),
    
    # Chat / IF / Vibes
    ("alpaca_eval_v3::hamish_zs_reasoning_deepseek", None),
    ("ifeval::hamish_zs_reasoning_deepseek", None),
]

TOOL_USE_BENCHMARK = ("bfcl_all::std", None)
SAFETY_BENCHMARK = ("trustllm_jailbreaktrigger::default", None)
SAFETY_REASONING_BENCHMARK = ("trustllm_jailbreaktrigger::wildguard_reasoning_answer", None)
def main():
    parser = argparse.ArgumentParser(description="Evaluate API Endpoints on Benchmark Suites")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        help="List of benchmark to evaluate on. Defaults to all predefined benchmarks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of examples per benchmark.",
    )
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
        required=True,
        choices=["cirrascale", "modal"],
        help="Provider of the model API endpoint. Currently supported: cirrascale, modal.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32768,
        help="Maximum generation length for the model.",
    )
    parser.add_argument(
        "--reasoning_model",
        action="store_true",
        help="Flag indicating if the model is a reasoning model.",
    )
    parser.add_argument(
        "--evaluate_tool_use",
        action="store_true",
        help="Flag to include tool use benchmark in the evaluation.",
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
    if args.benchmarks:
        benchmarks = [(b, args.limit) for b in args.benchmarks]
    else:
        benchmarks = BENCHMARKS
        if args.evaluate_tool_use:
            benchmarks.append(TOOL_USE_BENCHMARK)
        if args.reasoning_model:
            benchmarks.append(SAFETY_REASONING_BENCHMARK)
        else:
            benchmarks.append(SAFETY_BENCHMARK)
    for benchmark, limit in benchmarks:
        print(f"\n\n{benchmark}")
        print("=" * len(benchmark))
        # Run the evaluation command
        model_args = {"api_base_url": args.model_url}
        command_list = [
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
            "env-secret#1": "LITELLM_API_KEY=cirrascale_api_key" if args.provider == "cirrascale" else "LITELLM_API_KEY=modal_api_key"
        }
        command_list.append(
            f"--gantry-args '{json.dumps(gantry_args)}'"
        )
        if limit:
            command_list.append(f"--limit {limit}")
        if args.max_length:
            task_args = {"chat_overrides": {"generation_kwargs": {"max_gen_toks": args.max_length}}}
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