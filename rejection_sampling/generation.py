from collections import defaultdict
from dataclasses import dataclass
from vllm import LLM, SamplingParams
from transformers import (
    HfArgumentParser,
)
from datasets import load_dataset
import json


# 1. first sample a bunch of completions given prompts
# 2. tokenize them
# 3. run a reward model to filter them
# 4. run the SFT loss on the best ones

"""
python generation.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --sanity_check \
    --n 3 \
"""


@dataclass
class Args:
    model_name_or_path: str = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
    save_filename: str = "completions.jsonl"


@dataclass
class GenerationArgs:
    n: int = 1
    """the number of samples to generate per prompt"""
    temperature: float = 0.8
    response_length: int = 53
    tensor_parallel_size: int = 1


@dataclass
class DatasetArgs:
    dataset_name: str = None
    dataset_text_field: str = "prompt"
    dataset_train_split: str = "train"
    dataset_test_split: str = "validation"
    sanity_check: bool = False


def main(args: Args, dataset_args: DatasetArgs, gen_args: GenerationArgs):
    raw_datasets = load_dataset(dataset_args.dataset_name)
    if dataset_args.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(min(10, len(raw_datasets[key]))))

    # DATASET specific logic: in this dataset the prompt is simply just a list of strings
    prompts = raw_datasets[dataset_args.dataset_train_split]

    # Generate using vLLM
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=gen_args.tensor_parallel_size)
    outputs = llm.generate(
        prompts, 
        SamplingParams(
            n=gen_args.n,
            temperature=gen_args.temperature,
            top_p=1.0,
            max_tokens=gen_args.response_length,
            include_stop_str_in_output=True,
        ),
    )
    # Assuming we generate n=3 completions per prompt, the outputs will look like:
    # prompt | completions
    # -------|------------
    # q1     | a1
    # q1     | a2
    # q1     | a3
    # q2     | a1
    # ...
    table = defaultdict(list)
    for output in outputs:
        prompt = output.prompt
        for item in output.outputs:
            table["prompt"].append(prompt)
            table["completion"].append(item.text)
    
    # Save results
    with open(args.save_filename, 'w') as outfile:
        for i in range(len(table["prompt"])):
            json.dump({
                "prompt": table["prompt"][i],
                "completion": table["completion"][i],
            }, outfile)
            outfile.write('\n')


if __name__ == "__main__":
    parser = HfArgumentParser((Args, DatasetArgs, GenerationArgs))
    args, dataset_args, gen_args = parser.parse_args_into_dataclasses()
    main(args, dataset_args, gen_args)
