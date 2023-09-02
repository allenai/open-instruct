
'''
This script is used to get models' predictions on a set of prompts (put in files with *.jsonl format, with the prompt in a `prompt` field).

For example, to get predictions on a set of prompts, you should put them in a file with the following format:
    {"id": <uniq_id>, "prompt": "Plan a trip to Paris."}
    ...

Then you can run this script with the following command:
    python eval/predict.py \
        --model_name_or_path <huggingface_model_name_or_path> \
        --input_files <input_files> \
        --output_file <output_file> \
        --batch_size <batch_size> \
        --load_in_8bit
'''


import argparse
import json
import os
import vllm
import torch
from eval.utils import generate_completions, load_hf_lm_and_tokenizer, query_openai_chat_mode
from eval.templates import llama2_prompting_template, tulu_prompting_template


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Huggingface model name or path.")
    parser.add_argument(
        "--openai_engine", 
        type=str,
        default=None,
        help="OpenAI engine name.")
    parser.add_argument(
        "--input_files", 
        type=str, 
        nargs="+",
        help="Input .jsonl files containing `id`s and `prompt`s.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="output/model_outputs.jsonl",
        help="Output .jsonl file containing `id`s, `prompt`s, and `output`s.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size for prediction.")
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument(
        "--load_in_float16",
        action="store_true",
        help="By default, huggingface model will be loaded in the torch.dtype specificed in its model_config file."
             "If specified, the model dtype will be converted to float16 using `model.half()`.")
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument(
        "--use_vllm",
        action="store_true", 
        help="If given, we will use the vllm library, which will likely increase the inference throughput.")
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="plain",
        choices=["plain", "tulu-chat", "llama2-chat"], 
        help="encoding format of the prompt; this is only effective for local huggingface models.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="maximum number of new tokens to generate.")
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="whether to use sampling ; use greedy decoding otherwise.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature for sampling.")
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="top_p for sampling.")
    args = parser.parse_args()

    # model_name_or_path and openai_engine should be exclusive.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "model_name_or_path and openai_engine should be exclusive."
    return args


if __name__ == "__main__":
    args = parse_args()

    # check if output directory exists
    if args.output_file is not None:
        output_dir = os.path.dirname(args.output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # load the data
    for input_file in args.input_files:
        with open(input_file, "r") as f:
            instances = [json.loads(x) for x in f.readlines()]

    if args.model_name_or_path is not None:
        if args.prompt_format == "tulu-chat":
            prompts = [
                tulu_prompting_template.format(prompt=x["prompt"]) for x in instances
            ]
        elif args.prompt_format == "llama2-chat":
            prompts = [
                llama2_prompting_template.format(prompt=x["prompt"]) for x in instances
            ]
        else:
            prompts = [x["prompt"] for x in instances]
        if args.use_vllm:
            model = vllm.LLM(model=args.model_name_or_path)
            sampling_params = vllm.SamplingParams(
                temperature=args.temperature if args.do_sample else 0, 
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
            )
            outputs = model.generate(prompts, sampling_params)
            outputs = [it.outputs[0].text for it in outputs]
        else:
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path, 
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                load_in_8bit=args.load_in_8bit, 
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq
            )
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        with open(args.output_file, "w") as f:
            for instance, output in zip(instances, outputs):
                instance["output"] = output
                f.write(json.dumps(instance) + "\n")
                
    elif args.openai_engine is not None:
        query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            output_path=args.output_file,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
    else:
        raise ValueError("Either model_name_or_path or openai_engine should be provided.")

    print("Done.")