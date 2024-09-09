import argparse
import os
import json
import random
import torch
import vllm
from datasets import load_dataset
from eval.utils import (
    generate_completions, 
    load_hf_lm, 
    query_openai_chat_model,
    dynamic_import_function,
    load_hf_tokenizer,
    upload_results_to_hf,
    check_and_upload_model_metadata,
)
from eval.codex_humaneval.data import write_jsonl
from eval.mbpp.evaluation import compute_code_eval


def main(args):
    random.seed(42)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    dataset = load_dataset("evalplus/mbppplus")['test']
    dataset.shuffle(seed=42)
    # Always head-out first 100 examples
    if args.max_num_examples is None:
        args.max_num_examples = len(dataset) - 100
    if args.max_num_examples > len(dataset) - 100:
        Warning("The number of examples is larger than the test set size. Will use the maximum number of examples.")
        args.max_num_examples = len(dataset) - 100
    test_data = dataset.select(range(100, min(100+args.max_num_examples, len(dataset))))
    print("Number of examples:", len(test_data))
    
    if args.use_chat_format:
        prompts = []
        chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
        answer = "Here is the completed function:\n\n\n```python\n"

        def apply_chat_format(tokenizer, inst, suffix):
            messages = [{"role": "user", "content": inst}]
            prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
            prefix = "" if prompt[-1] in ["\n", " "] else " "
            return prompt + prefix + suffix
        
        if args.use_evalplus_prompt:
            instruction = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
            suffix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
            for example in test_data:
                data_inst = instruction + f"\n```\n{example['prompt'].strip()}\n{random.choice(example['test_list'])}```\n"
                suffix_inst = f"\n{suffix}\n```python\n" + example['code'].split(":")[0] + ":"
                prompts.append((data_inst, suffix_inst)) 
        else:
            instruction = "Complete the following python function.\n\n\n"
            for example in test_data:
                prompts.append((instruction + example["prompt"] + example['code'].split(":")[0], answer))
    else:
        prompts = [example["prompt"] + example['code'].split(":")[0] for example in test_data]
    
    stop_sequences = ['```'] + args.additional_stop_sequence
    if args.use_evalplus_prompt:
        stop_sequences += ['\n"""', "\nassert", "\n#"]
        
    if not args.results_file:
        if args.model_name_or_path:
            tokenizer = load_hf_tokenizer(
                model_name_or_path=args.model_name_or_path,
                revision=args.hf_revision,
                tokenizer_name_or_path=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )
            if args.use_vllm:
                model = vllm.LLM(
                    model=args.model_name_or_path,
                    tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                    tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                    tensor_parallel_size=torch.cuda.device_count(),
                    tokenizer_revision=args.hf_revision,
                    revision=args.hf_revision,
                )
                sampling_params = vllm.SamplingParams(
                    n=args.unbiased_sampling_size_n,
                    temperature=args.temperature,
                    max_tokens=512,
                    stop=stop_sequences,
                )
                if args.use_chat_format:
                    prompts = [apply_chat_format(tokenizer, inst, suffix) for (inst, suffix) in prompts]
                generations = model.generate(prompts, sampling_params)
                outputs = [output.text for it in generations for output in it.outputs]
                # Note: early vllm might ignore the first space in the generation, because the processing of _token.
                # This is not a problem for chat, but for codex, we need to keep the first space.
                # Be careful here!
                outputs = [output for output in outputs]
            else:
                print("Loading model and tokenizer...")
                model = load_hf_lm(
                    model_name_or_path=args.model_name_or_path, 
                    revision=args.hf_revision,
                    load_in_8bit=args.load_in_8bit, 
                    # device map is determined by the number of gpus available.
                    device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                    gptq_model=args.gptq,
                )
                from transformers import GPTNeoXForCausalLM, OPTForCausalLM
                if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
                    tokenizer.model_max_length = model.config.max_position_embeddings
                    print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))
                
                if args.use_chat_format:
                    prompts = [apply_chat_format(tokenizer, inst, suffix) for (inst, suffix) in prompts]

                # Because many tokenizers will treat the word after space differently from the original word alone, 
                # to be consistent, we add a space before tokenization and remove it after tokenization.
                stop_sequences = [tokenizer.encode(" " + x, add_special_tokens=False)[1:] for x in stop_sequences]
                outputs_per_sampling_iter = []
                for sampling_iter in range(args.unbiased_sampling_size_n):
                    print(f"Sampling iter: {sampling_iter} / {args.unbiased_sampling_size_n}")
                    sampling_outputs = generate_completions(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        max_new_tokens=512,
                        batch_size=args.eval_batch_size,
                        stop_id_sequences=stop_sequences,
                        num_return_sequences=1,  # we don't use the hf num_return_sequences, because otherwise the real batch size will be multiplied by it and often cause oom.
                        do_sample=True,  # if only pass@1 is evaluated, we do greedy decoding.
                        temperature=args.temperature,
                    )
                    outputs_per_sampling_iter.append(sampling_outputs)
                # regroup the outputs to match the number of test data.
                outputs = []
                for i in range(len(prompts)):
                    for j in range(args.unbiased_sampling_size_n):
                        outputs.append(outputs_per_sampling_iter[j][i])
        else:
            instances = [{
                "id": examle["task_id"], 
                "prompt": "Complete the following python function. Please only output the code for the completed function.\n\n\n" + prompt,
            } for examle, prompt in zip(test_data, prompts)]
            results = query_openai_chat_model(
                engine=args.openai_engine,
                instances=instances,
                output_path=os.path.join(args.save_dir, "openai_query_results.jsonl"),
                batch_size=args.eval_batch_size,
                top_p=0.95,
                temperature=args.temperature,
                n=args.unbiased_sampling_size_n,
            )
            outputs = []
            for result in results:
                for choice in result["response_metadata"]["choices"]:
                    outputs.append(choice["message"]["content"])
    else:
        with open(args.results_file, "r") as f:
            outputs = [json.loads(line)['completion'] for line in f]

    # duplicates test data to match the number of outputs.
    duplicate_test_data = [
        example for example in test_data for _ in range(args.unbiased_sampling_size_n)
    ]
    duplicate_prompts = [
        prompt for prompt in prompts for _ in range(args.unbiased_sampling_size_n)
    ]
    # if evalplus setup, we have to re-add the code prefix to the output.
    if args.use_evalplus_prompt:
        predictions = [{"task_id": example["task_id"], "prompt": prompt, "completion": example['code'].split(":")[0] + ":" + output, "test_cases": example['test']} 
                    for example, prompt, output in zip(duplicate_test_data, duplicate_prompts, outputs)]
    else:
        predictions = [{"task_id": example["task_id"], "prompt": prompt, "completion": output, "test_cases": example['test']} 
                   for example, prompt, output in zip(duplicate_test_data, duplicate_prompts, outputs)]
    predictions_noresult = [{"task_id":pred["task_id"], "prompt":pred['prompt'], "completion": pred['completion']} for pred in predictions]
    if args.use_chat_format:
        prediction_save_path = os.path.join(args.save_dir, "mbpp_chat_predictions.jsonl")
    else:
        prediction_save_path = os.path.join(args.save_dir, "mbpp_predictions.jsonl")
    write_jsonl(prediction_save_path, predictions_noresult)
    pass_at_k_results, results = compute_code_eval(
        predictions=predictions,
        k=args.eval_pass_at_ks,
        num_workers=64,
        timeout=10.0
    )

    if args.use_chat_format:
        with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
            json.dump(pass_at_k_results, fout)
    else:
        with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
            json.dump(pass_at_k_results, fout)

    if args.upload_to_hf is not None:
        # upload metrics to HF.
        # main metric is p@10 for temp=.8,
        # p@1 for temp=.1, maybe default to p@10 otherwise.
        results = pass_at_k_results
        pass_at = 1 if args.temperature == 0.1 else 10
        task_name = f"oi_mbpp_p@{str(pass_at)}"
        primary_score = results[f"pass@{str(pass_at)}"]
        upload_results_to_hf(
            results,
            args.upload_to_hf,
            args.hf_upload_name,
            task_name=task_name,
            primary_score=primary_score,
            prepend_timestamp=True,
        )
        check_and_upload_model_metadata(
            args.model_name_or_path, args.upload_to_hf, args.hf_upload_name, hf_revision=args.hf_revision
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default=None, 
        help="If specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default=None,
        help="if specified, we will load the model from a revision of the model in the hub"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default=None, 
        help="If specified, we will load the tokenizer from here."
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
        help="If specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/codex_eval", 
        help="Directory to save the results."
    )
    parser.add_argument(
        "--max_num_examples",
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--eval_pass_at_ks", 
        nargs="+", 
        type=int, 
        default=[1], 
        help="Multiple k's that we will report pass@k."
    )
    parser.add_argument(
        "--unbiased_sampling_size_n", 
        type=int, 
        default=20,
        help="Codex HumanEval requires `n` sampled generations per prompt, to estimate the unbiased pass@k. "
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for sampling. This is should be low for evaluating smaller pass@k, and high for larger pass@k."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="Load model in 8bit mode, which will reduce memory and speed up inference."
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
        default="minimal_multitask.eval.templates.create_prompt_with_tulu_chat_format", 
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
        '--use_evalplus_prompt',
        action="store_true",
        help="If given, we will use the evalplus prompting setup, to better match scores on the evalplus leaderboard."
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
    parser.add_argument("--results_file", type=str)
    args = parser.parse_args()
    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    assert args.unbiased_sampling_size_n >= max(args.eval_pass_at_ks), "n should be larger than the largest k in eval_pass_at_ks."
    main(args)