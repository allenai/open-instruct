"""
Generate from json file using CodeViewTool.
Used for code understanding tasks

python open_instruct/code_utils/toolvllm_code_generate.py \
    --json_path code_tasks.json \
    --model_path ai2-adapt-dev/tulu_3_model \
    --output_dir code_view_results \
    --max_eval_samples 1000 \
    --offset 0 \
    --code_view_api_endpoint http://localhost:1234/view_file
"""

import argparse
import json
import os
import re
import time
import signal
from tqdm import tqdm

import ray
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct.tool_utils.tool_vllm import ToolUseLLM, CodeViewTool

SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer to analyze code.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "str_replace_editor", "description": "Custom tool for viewing files * State is persistent across command calls and discussions with the user * If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep * If a `command` generates a long output, it will be truncated and marked with `<response clipped>`\\n", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The commands to run. The only allowed option is: `view`.", "enum": ["view"]}, "path": {"type": "string", "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`."}, "view_range": {"type": "array", "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.", "items": {"type": "integer"}}}, "required": ["command", "path"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


def main():
    try:
        parser = argparse.ArgumentParser(description="Evaluate code understanding tasks using CodeViewTool.")
        parser.add_argument(
            "--json_path", type=str, help="Path to the json file with code tasks."
        )
        parser.add_argument("--model_path", type=str, help="Path to the model.")
        parser.add_argument("--model_revision", type=str, default="main", help="Model revision.")
        parser.add_argument("--tokenizer_revision", type=str, default="main", help="Tokenizer revision.")
        parser.add_argument("--model_len", type=int, default=8192, help="Max model length.")
        parser.add_argument("--output_dir", type=str, default="tmp", help="Output directory.")
        parser.add_argument("--max_eval_samples", type=int, default=2000, help="Max eval samples.")
        parser.add_argument("--offset", type=int, default=0, help="Offset for the eval samples.")
        parser.add_argument("--code_view_api_endpoint", type=str, required=True, help="Code view API endpoint.")
        parser.add_argument("--dont_use_system_prompt", action="store_true", help="Don't use the system prompt.")
        parser.add_argument("--max_tool_calls", type=int, default=5, help="Maximum number of tool calls allowed.")
        args = parser.parse_args()

        # make output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.tokenizer_revision)

        # load the json data
        with open(args.json_path, "r") as f:
            dataset = json.load(f)

        original_dataset = dataset
        
        # Handle different question key formats
        if "query" in dataset[0]:
            question_key = "query"
        elif "question" in dataset[0]:
            question_key = "question"
        elif "prompt" in dataset[0]:
            question_key = "prompt"
        elif "messages" in dataset[0]:
            # Already formatted as messages
            question_key = None
        else:
            raise ValueError(f"Question key not found in dataset: {dataset[0]}")

        # Initialize the CodeViewTool
        tool = CodeViewTool(
            start_str="<tool_call>",
            end_str="</tool_call>",
            api_endpoint=args.code_view_api_endpoint,
        )

        # Format dataset
        if not args.dont_use_system_prompt:
            initial_message = [{"role": "system", "content": SYSTEM_PROMPT}]
        else:
            initial_message = []

        if question_key:
            ds = [{"messages": initial_message + [{"role": "user", "content": data[question_key]}]} for data in dataset]
        else:
            # Messages already formatted
            ds = [{"messages": initial_message + data["messages"]} for data in dataset]
            
        ds = Dataset.from_list(ds)

        if args.max_eval_samples > -1 and args.max_eval_samples < len(ds):
            ds = ds.select(range(args.offset, min(args.offset + args.max_eval_samples, len(ds))))
            original_dataset = original_dataset[args.offset:min(args.offset + args.max_eval_samples, len(original_dataset))]

        prompt_token_ids = [tokenizer.apply_chat_template(data["messages"], add_generation_prompt=True) for data in ds]

        # Create actor with CodeViewTool
        actor = ToolUseLLM(
            model=args.model_path,
            revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            tools={tool.end_str: tool},
            max_tool_calls=args.max_tool_calls,
            max_model_len=args.model_len,
        )
        
        # Use greedy decoding
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.model_len,
            include_stop_str_in_output=True,
            n=1,
            stop=[tool.end_str, "</answer>", "</solution>"],
        )
        
        # Generate output using the actor
        result = actor.generate(
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
        )
        
        # Grab text answers - for tool vllm, we have to decode the output ids
        generations = [x.outputs[0].token_ids for x in result]
        generations = [tokenizer.decode(x, skip_special_tokens=True) for x in generations]
        
        # Parse out answer/solution
        predictions = []
        for gen in generations:
            if "<answer>" in gen:
                pred = gen.split("<answer>")[-1].replace("</answer>", "")
            elif "<solution>" in gen:
                pred = gen.split("<solution>")[-1].replace("</solution>", "")
            else:
                pred = gen  # Use full generation if no tags found
            predictions.append(pred)

        # Save predictions with sample data
        with open(f"{args.output_dir}/predictions.jsonl", "w") as f:
            for sample, prediction, generation, original_sample in zip(ds, predictions, generations, original_dataset):
                f.write(json.dumps({
                    **sample, 
                    **original_sample, 
                    "prediction": prediction, 
                    "generation": generation
                }) + "\n")
                
        print(f"Completed generation for {len(predictions)} samples")
        print(f"Results saved to {args.output_dir}/predictions.jsonl")
        
    except Exception as e:
        print(f"Error: {e}")
        ray.shutdown()
        os._exit(1)
        raise


if __name__ == "__main__":
    main()