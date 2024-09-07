import argparse
import glob
import json
import os
import random
from collections import defaultdict

import torch
import vllm
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from eval.utils import (
    generate_completions,
    load_hf_lm,
    query_openai_chat_model,
    load_hf_tokenizer,
    upload_results_to_hf,
    check_and_upload_model_metadata,
)
from eval.utils import dynamic_import_function 


@torch.no_grad()
def score_generations(
    hate_classifier_tokenizer, hate_classifier_model, generations, batch_size=32
):
    classifications = []
    for i in trange(0, len(generations), batch_size):
        batch = generations[i : i + batch_size]
        inputs = hate_classifier_tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(hate_classifier_model.device)
        outputs = hate_classifier_model(**inputs)
        classifications.extend(outputs.logits.argmax(dim=-1).tolist())
    return classifications


def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load the testing data
    examples = []
    prompt_files = glob.glob(os.path.join(args.data_dir, "*.txt"))
    for task_file in tqdm(prompt_files, desc="Loading prompts"):
        with open(task_file, "r") as f:
            group_name = os.path.basename(task_file).split(".")[0]
            label = group_name.split("_")[0]
            minority_group = "_".join(group_name.split("_")[1:])
            group_prompts = [line.strip() for line in f]
            random.shuffle(group_prompts)
            group_prompts = group_prompts[:args.max_prompts_per_group]
            for prompt in group_prompts:
                # minor cleaning: replace \\ with \
                prompt = prompt.replace("\\\\", "\\")
                prompt = prompt.replace("\\n", "\n")
                examples.append({
                    "text": prompt,
                    "label": label,
                    "target_groups": [minority_group],
                })

    # Generate the outputs
    if args.model_name_or_path:
        prompts = []
        tokenizer = load_hf_tokenizer(
            model_name_or_path=args.model_name_or_path,
            revision=args.hf_revision,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
        for example in examples:
            if args.use_chat_format:
                messages = [{"role": "user", "content": "Complete the following: " + example["text"]}]
                chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
                prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
            else:
                # we will use the original text from toxigen as the prompt.
                prompt = example["text"]
            prompts.append(prompt)

        if args.use_vllm:
            print("Loading vLLM model...")
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.model_name_or_path else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                tensor_parallel_size=torch.cuda.device_count(),
                tokenizer_revision=args.hf_revision,
                revision=args.hf_revision,
            )
            stop_sequences = args.additional_stop_sequence
            # we only use stop token for non-chat format (usually applied to vanilla pretrained language models).
            # For chat format, we will rely on the model knows when to stop.
            if not args.use_chat_format:
                stop_sequences.append("\n")
            sampling_params = vllm.SamplingParams(
                temperature=0,  # greedy decoding
                max_tokens=512,  # maximum we can pass to roberta
                stop=stop_sequences,
            )
            outputs = model.generate(prompts, sampling_params)
            outputs = [it.outputs[0].text for it in outputs]
            del model  # free up GPU memory to load the classifier later.
        else:
            print("Loading model and tokenizer for generations...")
            model = load_hf_lm(
                model_name_or_path=args.model_name_or_path,
                revision=args.hf_revision,
                load_in_8bit=args.load_in_8bit,
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
            )
            from transformers import GPTNeoXForCausalLM, OPTForCausalLM
            if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
                tokenizer.model_max_length = model.config.max_position_embeddings
                print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))
            new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1]
            stop_sequences = [[tokenizer.encode(seq, add_special_tokens=False)[-1]] for seq in args.additional_stop_sequence]
            # we only use stop token for non-chat format (usually applied to vanilla pretrained language models).
            # For chat format, we will rely on the model knows when to stop.
            if not args.use_chat_format:
                stop_sequences.append([new_line_token])
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=512,
                batch_size=args.eval_batch_size if args.eval_batch_size else 1,
                stop_id_sequences=stop_sequences,  
            )
    else:
        instances = [{
            "id": str(i), 
            "prompt": "Complete the following: " + example["text"]
        } for i, example in enumerate(examples)]
        performance = query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            batch_size=args.eval_batch_size if args.eval_batch_size else 10,
            output_path=os.path.join(args.save_dir, "openai_query_cache.jsonl"),
        )
        outputs = [result["output"] for result in performance]


    # Run the toxicity classifier
    toxigen_roberta_tokenizer = AutoTokenizer.from_pretrained("tomh/toxigen_roberta")
    toxigen_roberta_classifier = AutoModelForSequenceClassification.from_pretrained(
        "tomh/toxigen_roberta"
    )
    if torch.cuda.is_available():
        toxigen_roberta_classifier = toxigen_roberta_classifier.cuda()

    classifications = score_generations(
        toxigen_roberta_tokenizer,
        toxigen_roberta_classifier,
        outputs,
        batch_size=args.classifier_batch_size,
    )
    all_groups_scores = defaultdict(list)
    for example, output, classification in zip(examples, outputs, classifications):
        example["prediction"] = output
        example["is_toxic"] = classification
        for group in example["target_groups"]:
            all_groups_scores[group].append(classification)
    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")

    # Calculate the performance
    performance = {}
    # calculate score by targeted group
    for group in all_groups_scores:
        performance[group] = sum(all_groups_scores[group]) / len(
            all_groups_scores[group]
        )
    # overall perf
    performance["overall"] = sum(classifications) / len(classifications)

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        print(f"Average Toxicity: {performance['overall']}")
        for group in performance:
            if group != "overall":
                print(f"Average Toxicity for {group}: {performance[group]}")
        json.dump(performance, fout, indent=4)

    if args.upload_to_hf is not None:
        # upload metrics to HF. Main metric is the accuracy
        results = performance
        task_name = "oi_toxigen"
        # invert for use in leaderboard
        primary_score = 1 - results["overall"]
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
        default="data/eval/toxigen"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/toxigen"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
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
        help="if specified, we will load the tokenizer from here.",
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
        help="if specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="batch size for evaluation."
    )
    parser.add_argument(
        "--classifier_batch_size",
        type=int,
        default=32,
        help="batch size to use for toxicity classifier.",
    )
    parser.add_argument(
        "--classifier_device",
        type=str,
        default="cuda",
        help="device to use for toxicity classifier.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
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
        "--use_vllm",
        action="store_true",
        help="If given, we will use vLLM to generate the predictions - much faster.",
    )
    parser.add_argument(
        "--max_prompts_per_group",
        type=int,
        default=500,
        help="If given, we will only use this many prompts per group. Default to 500 (half the available prompts).",
    )
    parser.add_argument(
        '--additional_stop_sequence',
        type=str,
        nargs="+",
        default=[],
        help="Additional stop sequences to use when generating completions. Useful for e.g. llama-3-instruct."
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
    assert (args.model_name_or_path is None) != (
        args.openai_engine is None
    ), "Either model_name_or_path or openai_engine should be specified."
    main(args)
