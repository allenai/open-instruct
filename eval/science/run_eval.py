"""
Run evaluation for one of the science instruction datasets.

Use VLLM for inference.
"""

import argparse
import vllm
import json
import evaluate

from eval.utils import dynamic_import_function


def load_jsonl(fname):
    with open(fname, "r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(xs, fname):
    with open(fname, "w") as f:
        for x in xs:
            print(json.dumps(x), file=f)


def compute_metrics(output_text, targets):
    "Compute BLEU and ROUGE."
    evaluators = {"bleu": evaluate.load("bleu"), "rouge": evaluate.load("rouge")}

    results = {}
    for eval_name, evaluator in evaluators.items():
        results_loop = evaluator.compute(predictions=output_text, references=targets)
        if eval_name == "bleu":
            results["bleu"] = results_loop["bleu"]
        else:
            results.update(results_loop)

    return results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, help="File to evaluate on.")
    parser.add_argument("--metrics_file", type=str, help="File to store metrics.")
    parser.add_argument(
        "--prediction_file", type=str, help="File to store predictions."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Model to generate predictions.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer.",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts.",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Max prompts to evaluate.",
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_llama2_chat_format",
        help=(
            "The function to use to create the chat format. This function will be "
            "dynamically imported. Please see examples in `eval/templates.py`."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3, help="Decoding temperature."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512, help="Max tokens to generate."
    )
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    model = vllm.LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
    )

    examples = load_jsonl(args.eval_file)
    if args.max_prompts is not None:
        examples = examples[:args.max_prompts]

    chat_formatting_function = (
        dynamic_import_function(args.chat_formatting_function)
        if args.use_chat_format
        else None
    )

    # Format prompts.
    prompts = []
    for example in examples:
        if args.use_chat_format:
            messages = [{"role": "user", "content": example["input"]}]
            prompt = chat_formatting_function(messages, add_bos=False)
        else:
            prompt = example["input"]
        prompts.append(prompt)

    sampling_params = vllm.SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )
    outputs = model.generate(prompts, sampling_params)
    output_text = [it.outputs[0].text for it in outputs]

    # Add model output to examples and write to file.
    for example, output_instance in zip(examples, output_text):
        example["model_output"] = output_instance
    write_jsonl(examples, args.prediction_file)

    # Compute metrics.
    targets = [entry["output"] for entry in examples]
    metrics = compute_metrics(output_text, targets)
    with open(args.metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
