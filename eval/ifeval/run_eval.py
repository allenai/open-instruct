'''
This script is adapted from the official IFEVAL evaluation script:
https://github.com/google-research/google-research/tree/master/instruction_following_eval
'''

import argparse
import os
import re
import json
import torch
import random
import vllm
import dataclasses
import collections
from typing import Dict, List, Optional, Union

from eval.utils import (
    load_hf_lm,
    generate_completions,
    query_openai_chat_model,
    dynamic_import_function,
    load_hf_tokenizer,
    upload_results_to_hf,
    check_and_upload_model_metadata
)
from eval.ifeval import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: List[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: List[bool]


def read_prompt_list(input_jsonl_filename):
    """Read inputs from jsonl."""
    inputs = []
    with open(input_jsonl_filename, "r") as f:
        for l in f:
            example = json.loads(l)
            inputs.append(
                InputExample(key=example["key"],
                            instruction_id_list=example["instruction_id_list"],
                            prompt=example["prompt"],
                            kwargs=example["kwargs"]))
    return inputs


def write_outputs(output_jsonl_filename, outputs):
    """Writes outputs to jsonl."""
    assert outputs
    with open(output_jsonl_filename, "w") as f:
        for o in outputs:
            f.write(
                json.dumps(
                    {
                        attr_name: o.__getattribute__(attr_name)
                        for attr_name in [
                            name for name in dir(o) if not name.startswith("_")
                        ]
                    }
                )
            )
            f.write("\n")


def test_instruction_following_strict(
    inp,
    prompt_to_response,
):
    """Tests response to see if instrutions are followed."""
    response = prompt_to_response[inp.prompt]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
    """Tests response for an upper bound for following instructions."""
    response = prompt_to_response[inp.prompt]
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def print_report(outputs):
    """Prints a report on accuracy scores."""

    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    tier0_total = collections.defaultdict(int)
    tier0_correct = collections.defaultdict(int)

    tier1_total = collections.defaultdict(int)
    tier1_correct = collections.defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(
            instruction_id_list, follow_instruction_list
        ):
            instruction_id = instruction_id.split(":")[0]
            tier0_total[instruction_id] += 1
            if followed_or_not:
                tier0_correct[instruction_id] += 1

        for instruction_id, followed_or_not in zip(
            instruction_id_list, follow_instruction_list
        ):
            tier1_total[instruction_id] += 1
            if followed_or_not:
                tier1_correct[instruction_id] += 1
            
    metrics = {
        "prompt-leval accuracy": prompt_correct / prompt_total,
        "instruction-level accuracy": instruction_correct / instruction_total,
        "tier0 accuracy": {instruction_id: tier0_correct[instruction_id] / tier0_total[instruction_id] for instruction_id in tier0_total},
        "tier1 accuracy": {instruction_id: tier1_correct[instruction_id] / tier1_total[instruction_id] for instruction_id in tier1_total},
    }

    print(json.dumps(metrics, indent=4))
    return metrics




def main(args):
    random.seed(42)

    inputs = read_prompt_list(os.path.join(args.data_dir, "input_data.jsonl"))

    os.makedirs(args.save_dir, exist_ok=True)

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

    if args.model_name_or_path:
        # prepare prompts    
        if args.use_chat_format:
            prompts = []
            chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
            for inp in inputs:
                prompts.append(
                    chat_formatting_function(
                        [{"role": "user", "content": inp.prompt}], tokenizer, add_bos=False
                    )
                )
        else:
            prompts = [inp.prompt for inp in inputs]

        # generate with vllm
        if args.use_vllm:
            sampling_params = vllm.SamplingParams(
                temperature=0,
                max_tokens=2048,
                stop=args.additional_stop_sequence,
            )
            # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
            generations = model.generate(prompts, sampling_params)
            prompt_to_output = {
                g.prompt: g.outputs[0].text for g in generations
            }
            outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]
        # generate with hf model
        else:
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=2048,
                temperature=0,
                batch_size=args.eval_batch_size if args.eval_batch_size else 1,
                stop_id_sequences=[tokenizer.convert_tokens_to_ids(stop) for stop in args.additional_stop_sequence],
            )
    else:
        instances = []
        for i, inp in enumerate(inputs):
            instances.append({
                "id": i,
                "prompt": inp.prompt,
            })
        results = query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            batch_size=args.eval_batch_size if args.eval_batch_size else 10,
            output_path=os.path.join(args.save_dir, f"openai_prediction_cache.jsonl"),
        )
        outputs = [result["output"] for result in results]

    assert len(inputs) == len(outputs), "Number of inputs and outputs are not the same."
    response_dict = {inp.prompt: output for inp, output in zip(inputs, outputs)}

    # get instruction following results
    results = {}
    for eval_setup, func in [
        ("strict", test_instruction_following_strict),
        ("loose", test_instruction_following_loose),
    ]:
        print(f"Running {eval_setup} evaluation...")
        outputs = []
        for inp in inputs:
            outputs.append(func(inp, response_dict))
        follow_all_instructions = [o.follow_all_instructions for o in outputs]
        accuracy = sum(follow_all_instructions) / len(outputs)
        print("Accuracy: %f", accuracy)
        results[eval_setup] = {"Accuracy": accuracy}

        output_file_name = os.path.join(
            args.save_dir, f"eval_results_{eval_setup}" + ".jsonl"
        )
        write_outputs(output_file_name, outputs)
        print(f"Results written to {output_file_name}")

        # Prints instruction following accuracy report.
        print("=" * 64)
        print(f"Detailed Scores:")
        detailed_scores = print_report(outputs)
        results[eval_setup].update(detailed_scores)

    # save the performance
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(results, fout, indent=4)

    if args.upload_to_hf is not None:
        # upload metrics to HF. Main metric is the loose acc.
        task_name = "oi_ifeval"
        primary_score = results["loose"]["Accuracy"]
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
        default="data/eval/ifeval/"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/ifeval/"
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
        "--max_num_examples", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate."
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
