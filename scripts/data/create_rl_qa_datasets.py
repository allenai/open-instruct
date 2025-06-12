import json
import datasets
from tqdm import tqdm
from datasets import DatasetDict

PROMPT  = "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <query> query </query>, and it will return the top searched results between <output> and </output>. You can search up to 3 times. If you search more, you will receive an error message. If you find no further external knowledge needed or have reached the maximum number of searches, you can directly provide the answer inside <finish> and </finish> without detailed illustrations. For example, <finish> xxx </finish>. Question: "

example_data = [
    {
        "messages": [ { "content": f"{PROMPT}Who played Creon in Antigone at the Epidaurus Festival 2022?", "role": "user" } ],
        "ground_truth": "Vasilis Bisbikis played Creon in Antigone at the Epidaurus Festival in 2022.",
        "dataset": "re_search",
    }
]


def convert_hotpotqa_to_rlvr_format(no_prompt: bool):
    # Load the HotpotQA dataset
    hotpotqa_dataset = datasets.load_dataset("hotpot_qa", "fullwiki")
    hotpotqa_data = hotpotqa_dataset["train"]

    if no_prompt:
        prompt = ""
    else:
        prompt = PROMPT
    
    rlvr_data = []
    for example in hotpotqa_data:
        question = example["question"]
        message = [ { "content": f"{prompt}{question}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": str(example["answer"]), # Convert to string to match schema
            "dataset": "re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("hamishivi/hotpotqa_rlvr" + ("_no_prompt" if no_prompt else ""), split="train")
    
    # Prepare test set
    hotpotqa_test_data = hotpotqa_dataset["validation"]
    rlvr_data = []
    for example in hotpotqa_test_data:
        question = example["question"]
        message = [ { "content": f"{prompt}{question}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": str(example["answer"]), # Convert to string to match schema
            "dataset": "re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("hamishivi/hotpotqa_rlvr" + ("_no_prompt" if no_prompt else ""), split="test")


def convert_nq_to_rlvr_format_with_context(no_prompt: bool):
    # Load the NQ dataset
    nq_dataset = datasets.load_dataset("google-research-datasets/nq_open")
    nq_data = nq_dataset["train"]

    if no_prompt:
        prompt = ""
    else:
        prompt = PROMPT
    
    rlvr_data = []
    for example in nq_data:
        question = example["question"]
        message = [ { "content": f"{prompt}{question}?", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": json.dumps(example["answer"]),
            "dataset": "re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("hamishivi/nq_rlvr" + ("_no_prompt" if no_prompt else ""), split="train")
    
    # Prepare test set
    nq_test_data = nq_dataset["validation"]
    rlvr_data = []
    for example in nq_test_data:
        question = example["question"]
        message = [ { "content": f"{prompt}{question}?", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": json.dumps(example["answer"]),
            "dataset": "re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("hamishivi/nq_rlvr" + ("_no_prompt" if no_prompt else ""), split="test")


def check_nq_rlvr_dataset():
    # Load the NQ dataset
    nq_dataset = datasets.load_dataset("hamishivi/nq_rlvr")
    nq_data = nq_dataset["train"]
    import pdb; pdb.set_trace()
    

def convert_tqa_to_rlvr_format(no_prompt: bool):
    # Load the TQA dataset
    tqa_dataset = datasets.load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
    tqa_data = tqa_dataset["train"]

    if no_prompt:
        prompt = ""
    else:
        prompt = PROMPT
    
    rlvr_data = []
    for example in tqa_data:
        question = example["question"]
        message = [ { "content": f"{prompt}{question}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": json.dumps(example["answer"]["aliases"]),
            "dataset": "re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("hamishivi/tqa_rlvr" + ("_no_prompt" if no_prompt else ""), split="train")
    
    # Prepare test set
    tqa_test_data = tqa_dataset["validation"]
    rlvr_data = []
    for example in tqa_test_data:
        question = example["question"]
        message = [ { "content": f"{prompt}{question}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": json.dumps(example["answer"]["aliases"]),
            "dataset": "re_search"})
        
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("hamishivi/tqa_rlvr" + ("_no_prompt" if no_prompt else ""), split="test")


def convert_2wiki_to_rlvr_format(no_prompt: bool):
    # Load the 2Wiki dataset
    two_wiki_dataset = datasets.load_dataset("hzy/kr1_2wiki")
    two_wiki_data = two_wiki_dataset["train"]

    if no_prompt:
        prompt = ""
    else:
        prompt = PROMPT
    
    rlvr_data = []
    for example in two_wiki_data:
        question = example["problem"]
        message = [ { "content": f"{prompt}{question}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": example["golden_answers"][0],
            "dataset": "re_search"})   
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("hamishivi/2wiki_rlvr" + ("_no_prompt" if no_prompt else ""), split="train")
    
    # Prepare test set
    two_wiki_test_data = two_wiki_dataset["test"]
    rlvr_data = []
    for example in two_wiki_test_data:
        question = example["problem"]
        message = [ { "content": f"{prompt}{question}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": example["golden_answers"][0],
            "dataset": "re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("hamishivi/2wiki_rlvr" + ("_no_prompt" if no_prompt else ""), split="test")

def convert_simple_qa_to_rlvr_format(no_prompt: bool):
    # Load the SimpleQA dataset
    simple_qa_dataset = datasets.load_dataset("basicv8vc/SimpleQA", split="test")
    simple_qa_dataset =  simple_qa_dataset.train_test_split(test_size=0.1, seed=42)
    simple_qa_data = simple_qa_dataset["train"]
    simple_qa_test_data = simple_qa_dataset["test"]

    if no_prompt:
        prompt = ""
    else:
        prompt = PROMPT
    
    rlvr_data = []
    for example in simple_qa_data:
        question = example["problem"]
        message = [ { "content": f"{prompt}{question}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": example["answer"],
            "dataset": "re_search"})

    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("hamishivi/simple_qa_rlvr" + ("_no_prompt" if no_prompt else ""), split="train")

    # Prepare test set
    simple_qa_test_data = simple_qa_dataset["test"]
    rlvr_data = []
    for example in simple_qa_test_data:
        question = example["problem"]
        message = [ { "content": f"{prompt}{question}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": example["answer"],
            "dataset": "re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("hamishivi/simple_qa_rlvr" + ("_no_prompt" if no_prompt else ""), split="test")

def convert_deepseek_long_form_reasoning_to_rl_format(no_prompt: bool):
    # Load the HotpotQA dataset
    dataset = datasets.load_dataset(
        "glaiveai/reasoning-v1-20m",
    )["train"].select(range(1000000))
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    if no_prompt:
        prompt = ""
    else:
        prompt = PROMPT
    
    rl_data = []
    for example in tqdm(train_data):
        question = example["prompt"]
        message = [ { "content": f"{question}{prompt}", "role": "user" } ]
        rl_data.append({
            "messages": message,
            "ground_truth": example["response"].split("</think>")[-1],
            "dataset": "long_re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rl_data)
    dataset.push_to_hub("hamishivi/glaiveai-reasoning-v1-1m_rl" + ("_no_prompt" if no_prompt else ""), split="train")
    
    # Prepare test set
    rl_data = []
    for example in tqdm(test_data):
        question = example["prompt"]
        message = [ { "content": f"{question}{prompt}", "role": "user" } ]
        rl_data.append({
            "messages": message,
            "ground_truth": example["response"].split("</think>")[-1],
            "dataset": "long_re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rl_data)
    dataset.push_to_hub("hamishivi/glaiveai-reasoning-v1-1m_rl" + ("_no_prompt" if no_prompt else ""), split="test")


def convert_os_to_rl_format(no_prompt: bool):
    # Load the OS dataset
    dataset = datasets.load_dataset("OpenSciLM/OS_Train_Data", split="train")
    # Remove rows with dataset=="hamishivi/rs-llama-3-stuff"
    dataset = dataset.filter(lambda x: x["dataset"] != "hamishivi/rs-llama-3-stuff")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    if no_prompt:
        prompt = ""
    else:
        prompt = PROMPT
    
    rl_data = []
    for example in train_data:
        message = example["messages"]
        rl_data.append({
            "messages": message,
            "ground_truth": message[-1]["content"],
            "dataset": "long_re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rl_data)
    dataset.push_to_hub("hamishivi/open_scholar_rl" + ("_no_prompt" if no_prompt else ""), split="train")
    
    # Prepare test set
    rl_data = []
    for example in test_data:
        message = example["messages"]
        rl_data.append({
            "messages": message,
            "ground_truth": message[-1]["content"],
            "dataset": "long_re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rl_data)
    dataset.push_to_hub("hamishivi/open_scholar_rl" + ("_no_prompt" if no_prompt else ""), split="test")

def convert_asqa_to_rlvr_format(no_prompt: bool):
    # Load the ASQA dataset
    dataset = datasets.load_dataset("din0s/asqa")
    train_data = dataset["train"]
    validation_data = dataset["dev"]
    
    if no_prompt:
        prompt = ""
    else:
        prompt = PROMPT
    
    # Process training data
    rlvr_data = []
    for item in tqdm(train_data):
        question = item['ambiguous_question']
        answer = item['annotations'][0]['long_answer']
        message = [ { "content": f"{prompt}{question}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": answer,
            "dataset": "re_search"
        })
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("hamishivi/asqa_long_form_rlvr" + ("_no_prompt" if no_prompt else ""), split="train")
    
    # Process validation data
    rlvr_data = []
    for item in tqdm(validation_data):
        question = item['ambiguous_question']
        answer = item['annotations'][0]['long_answer']
        message = [ { "content": f"{prompt}{question}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": answer,
            "dataset": "re_search"
        })
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("hamishivi/asqa_long_form_rlvr" + ("_no_prompt" if no_prompt else ""), split="test")

if __name__ == "__main__":
    
    # you might get rate limited running all this at once, so just comment a few if you need to
    # convert_hotpotqa_to_rlvr_format(True)
    # convert_nq_to_rlvr_format_with_context(True)
    # convert_tqa_to_rlvr_format(True)
    # convert_2wiki_to_rlvr_format(True)
    # convert_simple_qa_to_rlvr_format(True)
    convert_deepseek_long_form_reasoning_to_rl_format(True)
    convert_os_to_rl_format(True)
    convert_asqa_to_rlvr_format(True)
    # convert_hotpotqa_to_rlvr_format(False)
    # convert_nq_to_rlvr_format_with_context(False)
    # convert_tqa_to_rlvr_format(False)
    # convert_2wiki_to_rlvr_format(False)
    # convert_simple_qa_to_rlvr_format(False)
    convert_deepseek_long_form_reasoning_to_rl_format(False)
    convert_os_to_rl_format(False)
    convert_asqa_to_rlvr_format(False)
