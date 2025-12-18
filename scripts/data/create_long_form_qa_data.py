import datasets
from tqdm import tqdm

from open_instruct import utils

no_prompt = True
if no_prompt:
    prompt = ""
else:
    prompt = " Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <query> query </query>, and it will return the top searched results between <document> and </document>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <finish> and </finish>. For example, <finish> xxx </finish>. Question: <question>"

example_data = [
    {
        "messages": [ { "content": f"Who played Creon in Antigone at the Epidaurus Festival 2022?{prompt}", "role": "user" } ],
        "ground_truth": "Vasilis Bisbikis played Creon in Antigone at the Epidaurus Festival in 2022.",
        "dataset": "re_search",
    }
]


def convert_deepseek_long_form_reasoning_to_rl_format():
    # Load the HotpotQA dataset
    dataset = datasets.load_dataset(
        "glaiveai/reasoning-v1-20m",
        # cache_dir="/checkpoint/comem/rulin/cache/huggingface",
        # download_timeout=1000,
        num_proc=utils.max_num_processes(),
    )
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = dataset["train"]
    test_data = dataset["test"]

    rl_data = []
    for example in tqdm(train_data):
        question = example["prompt"]
        message = [ { "content": f"{question}{prompt}", "role": "user" } ]
        rl_data.append({
            "messages": message,
            "ground_truth": example["response"], # Convert to string to match schema
            "no_reasoning_ground_truth": example["response"].split("</think>")[-1],
            "dataset": "long_re_search"})

    # upload to huggingface
    dataset = datasets.Dataset.from_list(rl_data)
    dataset.push_to_hub("rulins/reasoning-v1-1m_rl" + ("_no_prompt" if no_prompt else ""), split="train")

    # Prepare test set
    rl_data = []
    for example in tqdm(test_data):
        question = example["prompt"]
        message = [ { "content": f"{question}{prompt}", "role": "user" } ]
        rl_data.append({
            "messages": message,
            "ground_truth": example["response"],
            "no_reasoning_ground_truth": example["response"].split("</think>")[-1],
            "dataset": "long_re_search"})

    # upload to huggingface
    dataset = datasets.Dataset.from_list(rl_data)
    dataset.push_to_hub("rulins/reasoning-v1-1m_rl" + ("_no_prompt" if no_prompt else ""), split="test")


def convert_os_to_rl_format():
    # Load the OS dataset
    dataset = datasets.load_dataset(
        "OpenSciLM/OS_Train_Data",
        split="train",
        num_proc=utils.max_num_processes(),
    )
    # Remove rows with dataset=="hamishivi/rs-llama-3-stuff"
    dataset = dataset.filter(lambda x: x["dataset"] != "hamishivi/rs-llama-3-stuff")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = dataset["train"]
    test_data = dataset["test"]

    rl_data = []
    for example in train_data:
        message = example["messages"]
        rl_data.append({
            "messages": message,
            "ground_truth": message[-1]["content"],
            "no_reasoning_ground_truth": message[-1]["content"],
            "dataset": "long_re_search"})

    # upload to huggingface
    dataset = datasets.Dataset.from_list(rl_data)
    dataset.push_to_hub("rulins/open_scholar_rl" + ("_no_prompt" if no_prompt else ""), split="train")

    # Prepare test set
    rl_data = []
    for example in test_data:
        message = example["messages"]
        rl_data.append({
            "messages": message,
            "ground_truth": message[-1]["content"],
            "no_reasoning_ground_truth": message[-1]["content"],
            "dataset": "long_re_search"})

    # upload to huggingface
    dataset = datasets.Dataset.from_list(rl_data)
    dataset.push_to_hub("rulins/open_scholar_rl" + ("_no_prompt" if no_prompt else ""), split="test")


if __name__ == "__main__":
    convert_deepseek_long_form_reasoning_to_rl_format()
