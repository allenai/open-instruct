import argparse
import os
import re
import json
import tqdm
import glob
import torch
import random
import vllm
import evaluate
from eval.utils import (
    load_hf_lm,
    generate_completions,
    query_openai_chat_model,
    dynamic_import_function,
    load_hf_tokenizer,
    upload_results_to_hf,
    check_and_upload_model_metadata
)


exact_match = evaluate.load("exact_match")


def main(args):
    random.seed(42)

    all_tasks = {}
    task_files = glob.glob(os.path.join(args.data_dir, "bbh", "*.json"))
    for task_file in tqdm.tqdm(task_files, desc="Loading tasks"):
        with open(task_file, "r") as f:
            task_name = os.path.basename(task_file).split(".")[0]
            all_tasks[task_name] = json.load(f)["examples"]
            if args.max_num_examples_per_task:
                all_tasks[task_name] = random.sample(all_tasks[task_name], args.max_num_examples_per_task)

    all_prompts = {}
    cot_prompt_files = glob.glob(os.path.join(args.data_dir, "cot-prompts", "*.txt"))
    for cot_prompt_file in tqdm.tqdm(cot_prompt_files, desc="Loading prompts"):
        with open(cot_prompt_file, "r") as f:
            task_name = os.path.basename(cot_prompt_file).split(".")[0]
            task_prompt = "".join(f.readlines()[2:])
            if args.no_cot:
                prompt_fields = task_prompt.split("\n\n")
                new_prompt_fields = []
                for prompt_field in prompt_fields:
                    if prompt_field.startswith("Q:"):
                        assert "So the answer is" in prompt_field, f"`So the answer is` not found in prompt field of {task_name}.txt."
                        assert "\nA:" in prompt_field, "`\nA:` not found in prompt field."
                        answer = prompt_field.split("So the answer is")[-1].strip()
                        question = prompt_field.split("\nA:")[0].strip()
                        new_prompt_fields.append(question + "\nA: " + answer)
                    else:
                        new_prompt_fields.append(prompt_field)
                task_prompt = "\n\n".join(new_prompt_fields)
            all_prompts[task_name] = task_prompt

    assert set(all_tasks.keys()) == set(all_prompts.keys()), "task names in task data and task prompts are not the same."

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "predictions"), exist_ok=True)

    # Load model if not using OpenAI API
    if args.model_name_or_path:
        tokenizer = load_hf_tokenizer(
            model_name_or_path=args.model_name_or_path,
            revision=args.hf_revision,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
        if args.use_vllm:
            print("Loading vllm model...")
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                tensor_parallel_size=torch.cuda.device_count(),
                tokenizer_revision=args.hf_revision,
                revision=args.hf_revision,
            )
        else:
            print("Loading model and tokenizer with huggingface...")
            model = load_hf_lm(
                model_name_or_path=args.model_name_or_path, 
                revision=args.hf_revision,
                load_in_8bit=args.load_in_8bit, 
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
            )
            # modify tokenizer if required
            from transformers import GPTNeoXForCausalLM, OPTForCausalLM
            if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
                tokenizer.model_max_length = model.config.max_position_embeddings
                print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))

    performance = {}
    for task_name in tqdm.tqdm(all_tasks.keys(), desc="Evaluating"):
        task_examples = all_tasks[task_name]
        task_prompt = all_prompts[task_name]
        if args.model_name_or_path:
            # prepare prompts    
            if args.use_chat_format:
                prompts = []
                chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
                for example in task_examples:
                    prompt = task_prompt.strip() + "\n\nQ: " + example["input"]
                    messages = [{"role": "user", "content": prompt}]
                    prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
                    prompt += "A:" if prompt[-1] in ["\n", " "] else " A:"
                    prompts.append(prompt)
            else:
                prompts = [task_prompt.strip() + "\n\nQ: " + example["input"] + "\nA:" for example in task_examples]

            # generate with vllm
            if args.use_vllm:
                stop = args.additional_stop_sequence
                if not args.use_chat_format or args.stop_at_double_newline:
                    stop += ["\n\n"]
                sampling_params = vllm.SamplingParams(
                    temperature=0,
                    max_tokens=512,
                    stop=stop,
                )
                # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
                generations = model.generate(prompts, sampling_params)
                prompt_to_output = {
                    g.prompt: g.outputs[0].text for g in generations
                }
                outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]
            # generate with hf model
            else:
                stop_sequence = tokenizer.encode("\n\n", add_special_tokens=False)[-2:] # get the last token because the tokenizer may add space tokens at the start.
                outputs = generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=512,
                    temperature=0,
                    batch_size=args.eval_batch_size if args.eval_batch_size else 1,
                    stop_id_sequences=[[stop_sequence] + [tokenizer.encode(stop, add_special_tokens=False) for stop in args.additional_stop_sequence]],
                )
        else:
            instances = []
            for i, example in enumerate(task_examples):
                prompt = task_prompt.strip() + "\n\nQ: " + example["input"] + "\nA:"
                instances.append({
                    "id": example["id"] if "id" in example else i,
                    "prompt": prompt,
                })
            results = query_openai_chat_model(
                engine=args.openai_engine,
                instances=instances,
                batch_size=args.eval_batch_size if args.eval_batch_size else 10,
                output_path=os.path.join(args.save_dir, "predictions", f"{task_name}_openai_prediction_cache.jsonl"),
            )
            outputs = [result["output"] for result in results]

        targets = [example["target"] for example in task_examples]
        predictions = []
        for example, output in zip(task_examples, outputs):
            example["raw_output"] = output
            
            # extract the first answer after `the answer is` and before the next period.
            # if there is no such answer, we will just use the raw output.
            extracted_answer = re.search(r"[t|T]he answer is (.*?)\.", output)
            if extracted_answer:
                example["prediction"] = extracted_answer.group(1).strip()
            else:
                example["prediction"] = output.strip()
            predictions.append(example["prediction"])
        
        with open(os.path.join(args.save_dir, "predictions", f"{task_name}.jsonl"), "w") as fout:
            for example in task_examples:
                fout.write(json.dumps(example) + "\n")        

        assert len(predictions) == len(targets), "number of predictions and targets are not the same."
        performance[task_name] = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]

        print(f"Task {task_name} - EM: {performance[task_name]}")

    # save the performance
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        performance["average_exact_match"] = sum(performance.values()) / len(performance)
        print(f"Average EM: {performance['average_exact_match']}")
        json.dump(performance, fout, indent=4)

    if args.upload_to_hf is not None:
        # upload metrics to HF. Main metric is the accuracy
        results = performance
        task_name = "oi_bbh_cot"
        primary_score = results["average_exact_match"]
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
        "--data_dir", 
        type=str, 
        default="data/bbh"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/bbh"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the model to generate the predictions."
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
