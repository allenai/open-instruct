"""
This code is partially borrowed and adapted from: https://github.com/tencent-ailab/persona-hub

Example commands:
# generate 20 if prompts:
python persona_driven_generate_ifdata.py --model "gpt-4o" --start_index 0 --end_index 20 --output_path if_prompts.jsonl --openai_key Z --org_id YYY --dataset ai2-adapt-dev/personahub_personas --template instruction_following

# generate 20 IF responses
python persona_driven_generate_ifdata.py --model "gpt-4o" --start_index 0 --end_index 20 --output_path if_solutions.jsonl --openai_key Z --org_id YYY --dataset if_prompts.jsonl --template instruction_following_solution
"""

import argparse
import json
import random
import string

import openai
from datasets import load_dataset
from prompt_templates import (
    instruction_following,
    instruction_following_solution,
    instruction_template,
    knowledge_template,
    math_solution_template,
    math_template,
    npc_template,
    rewrite_if_prompt,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential  # for exponential backofff
from tqdm import tqdm

import open_instruct.utils as open_instruct_utils


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


system_prompt = """You are a helpful mathematician assistant."""
# client = OpenAI()   # set up your config/env/api for calling openai models
MODEL_COSTS = {
    "gpt-4": [0.00003, 0.00006],
    "gpt-3.5-turbo": [0.0000015, 0.000002],
    "gpt-4-1106-preview": [0.00001, 0.00003],
    "gpt-4o": [0.000005, 0.000015],
}


CONSTRAINTS = json.load(open("./data/if_constraint_fewshots_handwritten.json"))


def get_response(args, user_prompt):
    # completion = client.chat.completions.create(
    completion = completion_with_backoff(
        model=args.model,
        temperature=0.7,
        top_p=0.95,
        messages=[{"role": "system", "content": f"{system_prompt}"}, {"role": "user", "content": f"{user_prompt}"}],
    )
    total_input_tokens = completion.usage.prompt_tokens
    total_output_tokens = completion.usage.completion_tokens
    return completion.choices[0].message.content, total_input_tokens, total_output_tokens


process = (
    lambda x: x.replace("Math problem:\n", "")
    .replace("User instruction:\n", "")
    .replace("User instruction:", "")
    .lstrip("\n")
)


def main(args):
    # Load the appropriate template
    if args.template == "instruction":
        template = instruction_template
    elif args.template == "knowledge":
        template = knowledge_template
    elif args.template == "npc":
        template = npc_template
    elif args.template == "math":
        template = math_template
    elif args.template == "math_solution":
        template = math_solution_template
    elif args.template == "instruction_following":
        template = instruction_following
    elif args.template == "instruction_following_solution":
        template = instruction_following_solution
    elif args.template == "rewrite_if_prompt":
        template = rewrite_if_prompt
    else:
        raise ValueError("Invalid template type. Choose from 'instruction', 'knowledge', 'npc', or 'math'.")

    total_input_tokens, total_output_tokens = 0, 0
    in_cost_per_token, out_cost_per_token = MODEL_COSTS[args.model]

    # Load the dataset
    if args.dataset.endswith(".jsonl"):
        persona_dataset = load_dataset(
            "json", data_files=args.dataset, num_proc=open_instruct_utils.max_num_processes()
        )["train"]
    else:
        persona_dataset = load_dataset(args.dataset, num_proc=open_instruct_utils.max_num_processes())["train"]

    if args.sanity_check > 0:
        persona_dataset = persona_dataset.select(range(0, args.sanity_check))
    print(f"Total number of input personas: {len(persona_dataset)}")

    input_field = (
        "synthesized text" if args.template in ["instruction_following_solution", "rewrite_if_prompt"] else "persona"
    )
    with open(args.output_path, "w") as out:
        for idx, example in enumerate(tqdm(persona_dataset.select(range(args.start_index, args.end_index)))):
            if args.template == "rewrite_if_prompt":
                candid_constraints = example["constraints"]
                few_shot = ""
                # only rewrite prompts for example with 2 or more constarints
                if len(candid_constraints) < 2:
                    continue
            else:
                # randomly sample 1-3 constraints from the IFEval constraints list
                candid_constraints = random.sample(list(CONSTRAINTS.keys()), random.choice([1, 2, 3]))
                few_shot = random.choice(CONSTRAINTS[random.choice(candid_constraints)])

            id = "personahub_" + "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(24))
            input_text = example[input_field].strip()
            persona = (
                example["input persona"]
                if args.template in ["instruction_following_solution", "rewrite_if_prompt"]
                else input_text
            )
            user_prompt = template.format(
                persona=input_text,
                example=few_shot,
                constraints=", ".join(candid_constraints),
                category=", ".join([ex.split(":")[0] for ex in candid_constraints]),
            )
            gpt4o_out_text, in_tokens, out_tokens = get_response(args, user_prompt)
            o = (
                {
                    "id": id,
                    "prompt": input_text,
                    "input_persona": persona,
                    "messages": [
                        {"role": "user", "content": input_text},
                        {"role": "assistant", "content": gpt4o_out_text},
                    ],
                    "constraints": example.get("constraints", []),
                }
                if args.template in ["instruction_following_solution"]
                else {
                    "input persona": persona,
                    "synthesized text": process(gpt4o_out_text),
                    "constraints": candid_constraints
                    if args.template in ["instruction_following", "rewrite_if_prompt"]
                    else [],
                    "description": f"{args.template} problem",
                }
            )
            out.write(json.dumps(o, ensure_ascii=False) + "\n")
            # breakpoint()

            total_input_tokens += in_tokens
            total_output_tokens += out_tokens
            if idx % 20 == 0:
                print(f"estimated cost so far= ${in_cost_per_token * in_tokens + out_cost_per_token * out_tokens}")

    print(f"Outputted the results to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize text using a specified model and template.")
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        choices=[
            "math",
            "math_solution",
            "instruction_following",
            "instruction_following_solution",
            "rewrite_if_prompt",
        ],
        help=(
            "Prompt templates. Choose from 'instruction', 'knowledge', 'math' or 'npc'. "
            "You can also add more customized templates in prompt_templates.py"
        ),
    )
    parser.add_argument("--dataset", required=False, default="proj-persona/PersonaHub")
    parser.add_argument("--openai_key", required=True)
    parser.add_argument("--org_id", required=True)
    parser.add_argument(
        "--model", default="gpt-4o", choices=["gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4o"]
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file.")
    parser.add_argument("--chat_format", type=str, required=False, help="whether to put in chat format")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--sanity_check", type=int, default=0)

    args = parser.parse_args()
    openai.api_key = args.openai_key
    openai.organization = args.org_id

    main(args)
