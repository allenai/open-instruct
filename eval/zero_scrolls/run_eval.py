import argparse
import os
import re
import json
from collections import defaultdict, Counter

from tqdm import tqdm
import glob
import torch
import random
import vllm
import evaluate
from rouge_score import rouge_scorer

from vllm import LLM, SamplingParams
from datasets import load_dataset
from eval.utils import (
    load_hf_lm,
    generate_completions,
    query_openai_chat_model,
    dynamic_import_function,
    load_hf_tokenizer,
    upload_results_to_hf
)
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_to_max_input_tokens = {
    "google/flan-t5-xxl": 8192,
    "google/flan-t5-xl": 8192,
    "google/flan-t5-large": 8192,
    "google/flan-t5-base": 8192,
    "google/flan-t5-small": 8192,
    "google/flan-ul2": 8192,
    "bigscience/T0pp": 8192,
    "allenai/tulu-v2.5-ppo-13b-hh-rlhf-60k": 4096,
    "allenai/tulu-2-dpo-7b": 8192,
    "allenai/tulu-2-7b":2100,
    "meta-llama/Meta-Llama-3-8B-Instruct": 1000000000000000019884624838656
}


datasets = [
            'gov_report',
            'summ_screen_fd',
            'qmsum',
            'qasper',
            'narrative_qa',
            'quality',
]


def load_model(model_name_or_path, use_vllm):
    if model_name_or_path:
        tokenizer = load_hf_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
        if use_vllm:
            print("Loading vllm model...")
            model = vllm.LLM(
                model=args.model_name_or_path,
                tensor_parallel_size=torch.cuda.device_count(),
            )
        else:
            print("Loading model and tokenizer with huggingface...")
            model = load_hf_lm(
                model_name_or_path=args.model_name_or_path,
                load_in_8bit=args.load_in_8bit,
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
            )
            # modify tokenizer if required
            from transformers import GPTNeoXForCausalLM, OPTForCausalLM
            if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
                tokenizer.model_max_length = model.config.max_position_embeddings
                print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(
                    model.config.max_position_embeddings))
    else:
        raise ValueError("No model specified")

    return model, tokenizer

def process_model_input(tokenizer, example, max_tokens, device):

    tokenized_input_full = tokenizer(example["input"], return_tensors="pt").input_ids.to(device)

    if tokenized_input_full.shape[1] <= max_tokens:
        return tokenized_input_full

    seperator_and_query_text = example['truncation_seperator'] + example["input"][example['query_start_index']:]
    tokenized_seperator_and_query = tokenizer(seperator_and_query_text, return_tensors="pt").input_ids.to(device)
    input_without_query = example['input'][:example['query_start_index']]
    tokenized_input_without_query = tokenizer(input_without_query, return_tensors="pt").input_ids.to(device)
    tokenized_input_without_query = tokenized_input_without_query[:,
                                    :max_tokens - tokenized_seperator_and_query.shape[1]]
    tokenized_input = torch.cat([tokenized_input_without_query, tokenized_seperator_and_query], dim=1)
    return tokenized_input

def process_model_input_with_vllm(tokenizer, example, max_tokens, device):
    input_full = example["input"]
    # tokenized_input_full = tokenizer(input_full, return_tensors="pt").input_ids.to(device)
    messages = [{"role": "user", "content": input_full}]

    return messages, input_full


def unigram_f1_score(response, ground_truth):
    # Tokenize the text into unigrams
    response_unigrams = response.split()
    ground_truth_unigrams = ground_truth.split()

    # Count the occurrences of each unigram
    response_counter = Counter(response_unigrams)
    ground_truth_counter = Counter(ground_truth_unigrams)

    # Compute the intersection of unigrams
    intersection = sum((response_counter & ground_truth_counter).values())

    # Compute precision, recall, and F1 score
    precision = intersection / len(response_unigrams) if len(response_unigrams) > 0 else 0
    recall = intersection / len(ground_truth_unigrams) if len(ground_truth_unigrams) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score


def exact_match_contains(response, gold_answer):
    """
    Check if the gold answer is contained within the model's response.

    Args:
        response (str): The model's response.
        gold_answer (str): The correct answer.

    Returns:
        bool: True if the gold answer is contained in the response, False otherwise.
    """
    return gold_answer.strip() in response.strip()

def compute_metric(task_name, generated_response, gold):
    if task_name in ['gov_report', 'summ_screen_fd', 'qmsum']:
        scorer = rouge_scorer.RougeScorer(
            metrics=['rouge1', 'rouge2', 'rougeL'],
            max_n=2,  # For ROUGE-2
            apply_avg=True
        )
        metric = scorer.score(generated_response, gold)

    elif task_name in ['qasper', 'narrative_qa']:
        metric = unigram_f1_score(generated_response, gold)

    else:
        metric = int(exact_match_contains(generated_response, gold))

    return metric



def main(args):
    random.seed(42)
    # Load model if not using OpenAI API
    model, tokenizer = load_model(args.model_name_or_path, args.use_vllm)
    ###### compute performance ####

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "predictions"), exist_ok=True)
    generations_dir = os.path.join(args.save_dir, "predictions")
    print("Will write to:", generations_dir)
    max_examples_per_task = -1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_input_length = model_to_max_input_tokens[args.model_name_or_path]
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
    for dataset in tqdm(datasets):
        generations = dict()
        print(f"Processing {dataset}")
        time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        print(f"time as start {dataset}: {time}")
        print(f"Loading {dataset}")
        # load the data using HF
        data = load_dataset("tau/zero_scrolls", dataset, trust_remote_code=True)
        print(f"Loaded {dataset}")

        prompts = defaultdict(list)
        tokenized_prompts = []
        nb_examples_more_seq_length = 0
        nb_examples_less_seq_length = 0
        metrics = defaultdict(list)
        for i, example in tqdm(enumerate(data["validation"]), desc="Reading data"):
            task_name = dataset
            if 0 < max_examples_per_task == i:
                print(f"Reached {max_examples_per_task} for {dataset}. Breaking")
                break
            input_full = example["input"]
            tokenized_input_full = tokenizer(input_full, return_tensors="pt").input_ids.to(device)
            if tokenized_input_full.shape[1] >= max_input_length:
                nb_examples_more_seq_length+=1
            else:
                breakpoint()
                nb_examples_less_seq_length+=1
                messages, full_input = process_model_input_with_vllm(tokenizer, example,max_input_length, device)
                if args.use_chat_format:
                    breakpoint()
                    prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
                    breakpoint()
                    prompts[task_name].append(prompt)

                if args.use_vllm:
                    breakpoint()
                    sampling_params = vllm.SamplingParams(
                        temperature=1,
                        max_tokens=512,
                        top_p=1.0,
                        include_stop_str_in_output=True,
                    )
                    generations = model.generate(prompts[task_name], sampling_params)
                    prompt_to_output = {
                        g.prompt: g.outputs[0].text for g in generations
                    }
                    for g in generations:
                        prompt = g.prompt
                        generation = g.outputs[0].text
                        gold = 1
                        breakpoint()
                        # if task_name in ['gov_report', 'summ_screen_fd', 'qmsum']:
                        #     scores = compute_metric(task_name, generated_response, gold)
                        #     metrics[task_name] = {""}

                    outputs = {
                        "task_name": task_name
                    }
                    for prompt in prompts:
                        outputs[prompt] = prompt_to_output.get(prompt, "")
                else:
                    # generate with hf model
                    outputs = []
                    for model_input in tokenized_prompts:
                        prediction_token_ids = model.generate(model_input,
                                                              max_new_tokens=1024,
                                                              do_sample=False,
                                                              top_p=0,
                                                              top_k=0,
                                                              temperature=1)

                        predicted_text = tokenizer.decode(prediction_token_ids[0], skip_special_tokens=True)
                        outputs.append(predicted_text)

        print(f"Nb examples exceeding max_seq_length: {nb_examples_more_seq_length}")
        print(f"Nb examples not exceeding max_seq_length: {nb_examples_less_seq_length}")



        out_file_path = os.path.join(generations_dir, f"preds_{dataset}.json")
        with open(out_file_path, 'w') as f_out:
            json.dump(generations, f_out, indent=4)

        print(f"Done generating {len(generations)} examples from {dataset}")
        time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        print(f"time at end: {time}")
        print(f"Look for predictions in {generations_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/zero_scrolls"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--no_cot",
        action="store_true",
        help="if specified, chain of thoughts will be removed from the prompts."
    )
    parser.add_argument(
        "--max_num_examples_per_task",
        type=int,
        default=None,
        help="maximum number of examples to evaluate per task."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        '--additional_stop_sequence',
        type=str,
        nargs="+",
        default=[],
        help="Additional stop sequences to use when generating completions. Useful for e.g. llama-3-instruct."
    )
    parser.add_argument(
        '--stop_at_double_newline',
        action="store_true",
        help="If given, we will stop generation at the first double newline. Turn on to match older eval settings."
    )
    parser.add_argument(
        "--upload_to_hf",
        type=str,
        default=None,
        help="If specified, we will upload the results to Hugging Face Datasets. "
             "This should be the name of the dataset to upload to."
    )
    parser.add_argument(
        "--hf_upload_name",
        type=str,
        default=None,
        help="If uploading to hf, this is the model name"
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
