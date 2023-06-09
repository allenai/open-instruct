
import argparse
import json
import os
from eval.utils import generate_completions, load_hf_lm_and_tokenizer, query_openai_chat_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None, help="huggingface model name or path.")
    parser.add_argument("--openai_engine", type=str, default=None, help="openai engine name.")
    parser.add_argument("--input_files", type=str, nargs="+", default=["data/eval/creative_tasks/self_instruct_test.jsonl", "data/eval/creative_tasks/vicuna_test.jsonl"])
    parser.add_argument("--output_file", type=str, default="data/eval_creative_tasks/model_outputs.jsonl")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for prediction.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_chat_format", action="store_true", help="whether to use chat format to encode the prompt.")
    args = parser.parse_args()

    # model_name_or_path and openai_engine should be exclusive.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "model_name_or_path and openai_engine should be exclusive."

    # check if output directory exists
    if args.output_file is not None:
        output_dir = os.path.dirname(args.output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # load the data
    instances = []
    for input_file in args.input_files:
        with open(input_file, "r") as f:
            data = [json.loads(x) for x in f.readlines()]

        if "self_instruct" in os.path.basename(input_file):
            for instance in data:
                if instance["instances"][0]["input"]:
                    prompt = instance["instruction"].strip() + "\n\n" + instance["instances"][0]["input"].strip() + "\n"
                else:
                    prompt = instance["instruction"].strip() + "\n"
                instances.append({
                    "id": instance["id"],
                    "prompt": prompt,
                    "reference": instance["instances"][0]["output"]
                })
        elif "vicuna" in os.path.basename(input_file):
            for instance in data:
                instances.append({
                    "id": "vicuna_" + str(instance["question_id"]),
                    "prompt": instance["text"].strip() + "\n",
                })
        elif "koala" in os.path.basename(input_file):
            for instance in data:
                instances.append({
                    "id": instance["id"],
                    "prompt": instance["prompt"].strip() + "\n",
                })
        else:
            raise ValueError("Unsupported dataset: {input_file}.")
    
    
    # filter out instances that the prompt is too long
    instances = [x for x in instances if len(x["prompt"].split(" ")) <= 2048]
    print(f"Total number of instances: {len(instances)}")

    if args.model_name_or_path is not None:
        if args.use_chat_format:
            prompts = [
                "<|user|>\n" + x["prompt"] + "\n<|assistant|>\n" for x in instances
            ]
        else:
            prompts = [x["prompt"] for x in instances]
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path, 
            load_in_8bit=args.load_in_8bit, 
            load_in_half=True,
            gptq_model=args.gptq
        )
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=args.batch_size,
            max_new_tokens=1024
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
            temperature=0.0,
        )
    else:
        raise ValueError("Either model_name_or_path or openai_engine should be provided.")

    print("Done.")