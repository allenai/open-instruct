import argparse
import random
import json
import os
import tqdm
from eval.utils import load_hf_lm_and_tokenizer, score_completions


def main(args):
    random.seed(42)

    test_data = [json.loads(line) for line in open(args.data_file)]
    if args.max_num_examples is not None and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)
    print("Number of examples:", len(test_data))

    print("Loading model and tokenizer...")
    model, tokenizer = load_hf_lm_and_tokenizer(
        model_name_or_path=args.model_name_or_path, 
        tokenizer_name_or_path=args.tokenizer_name_or_path, 
        load_in_8bit=args.load_in_8bit, 
        load_in_half=True,
        gptq_model=args.gptq
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    if args.use_chat_format:
        prompts = ["<|user|>\n" + example["prompt"] + "\n<|assistant|>\n" for example in test_data]
    else:
        prompts = [example["prompt"] + "\n" if not example["prompt"].endswith("\n") else "" for example in test_data]

    targets = [example[args.output_field_name] for example in test_data]

    scoring_examples = [{"prompt": prompt, "completions": [target]} for prompt, target in zip(prompts, targets)]
    log_likihoods_scores = score_completions(
        model=model,
        tokenizer=tokenizer,
        scoring_examples=scoring_examples,
    )
    average_log_likelihood = sum(log_likihoods_scores[prompt][target] for prompt, target in zip(prompts, targets)) / len(test_data)
    print("Average log likelihood:", average_log_likelihood)
    
    perplexity = 2 ** -average_log_likelihood
    print("Perplexity:", perplexity)

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump({
            "average_log_likelihood": average_log_likelihood,
            "perplexity": perplexity,
        }, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/eval/creative_tasks/gpt4_outputs.jsonl")
    parser.add_argument("--output_field_name", type=str, default="gpt-4-0314_output")
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate.")
    parser.add_argument("--model_name_or_path", type=str, default="../hf_llama_models/7B")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/creative_tasks_eval")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_chat_format", action="store_true", help="If given, the prompt will be encoded as a chat format with the roles in prompt.")
    args = parser.parse_args()
    main(args)