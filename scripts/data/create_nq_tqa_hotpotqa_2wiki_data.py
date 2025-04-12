import json
import datasets


no_prompt = False
if no_prompt:
    prompt = ""
else:   
    prompt = " Search the web by wrapping a query in query tags like so: <query></query> Then, based on the snippet, provide the answer, or another query if you need. Finally, output your answer wrapped in answer tags: <final></final>."

example_data = [
    {
        "messages": [ { "content": f"Who played Creon in Antigone at the Epidaurus Festival 2022?{prompt}", "role": "user" } ],
        "ground_truth": "Vasilis Bisbikis played Creon in Antigone at the Epidaurus Festival in 2022.",
        "dataset": "re_search",
    }
]


def convert_hotpotqa_to_rlvr_format():
    # Load the HotpotQA dataset
    hotpotqa_dataset = datasets.load_dataset("hotpot_qa", "fullwiki")
    hotpotqa_data = hotpotqa_dataset["train"]
    
    rlvr_data = []
    for example in hotpotqa_data:
        question = example["question"]
        message = [ { "content": f"{question}{prompt}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": str(example["answer"]), # Convert to string to match schema
            "dataset": "re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("rulins/hotpotqa_rlvr" + ("_no_prompt" if no_prompt else ""), split="train")
    
    # Prepare test set
    hotpotqa_test_data = hotpotqa_dataset["validation"]
    rlvr_data = []
    for example in hotpotqa_test_data:
        question = example["question"]
        message = [ { "content": f"{question}{prompt}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": str(example["answer"]), # Convert to string to match schema
            "dataset": "re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("rulins/hotpotqa_rlvr" + ("_no_prompt" if no_prompt else ""), split="test")


def convert_nq_to_rlvr_format_with_context():
    # Load the NQ dataset
    nq_dataset = datasets.load_dataset("google-research-datasets/nq_open")
    nq_data = nq_dataset["train"]
    
    rlvr_data = []
    for example in nq_data:
        question = example["question"]
        message = [ { "content": f"{question}?{prompt}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": json.dumps(example["answer"]),
            "dataset": "re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("rulins/nq_rlvr" + ("_no_prompt" if no_prompt else ""), split="train")
    
    # Prepare test set
    nq_test_data = nq_dataset["validation"]
    rlvr_data = []
    for example in nq_test_data:
        question = example["question"]
        message = [ { "content": f"{question}?{prompt}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": json.dumps(example["answer"]),
            "dataset": "re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("rulins/nq_rlvr" + ("_no_prompt" if no_prompt else ""), split="test")


def check_nq_rlvr_dataset():
    # Load the NQ dataset
    nq_dataset = datasets.load_dataset("rulins/nq_rlvr")
    nq_data = nq_dataset["train"]
    import pdb; pdb.set_trace()
    

def convert_tqa_to_rlvr_format():
    # Load the TQA dataset
    tqa_dataset = datasets.load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
    tqa_data = tqa_dataset["train"]
    
    rlvr_data = []
    for example in tqa_data:
        question = example["question"]
        message = [ { "content": f"{question}{prompt}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": json.dumps(example["answer"]["aliases"]),
            "dataset": "re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("rulins/tqa_rlvr" + ("_no_prompt" if no_prompt else ""), split="train")
    
    # Prepare test set
    tqa_test_data = tqa_dataset["validation"]
    rlvr_data = []
    for example in tqa_test_data:
        question = example["question"]
        message = [ { "content": f"{question}{prompt}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": json.dumps(example["answer"]["aliases"]),
            "dataset": "re_search"})
        
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("rulins/tqa_rlvr" + ("_no_prompt" if no_prompt else ""), split="test")


def convert_2wiki_to_rlvr_format():
    # Load the 2Wiki dataset
    two_wiki_dataset = datasets.load_dataset("hzy/kr1_2wiki")
    two_wiki_data = two_wiki_dataset["train"]
    
    rlvr_data = []
    for example in two_wiki_data:
        question = example["problem"]
        message = [ { "content": f"{question}{prompt}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": example["golden_answers"][0],
            "dataset": "re_search"})   
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("rulins/2wiki_rlvr" + ("_no_prompt" if no_prompt else ""), split="train")
    
    # Prepare test set
    two_wiki_test_data = two_wiki_dataset["test"]
    rlvr_data = []
    for example in two_wiki_test_data:
        question = example["problem"]
        message = [ { "content": f"{question}{prompt}", "role": "user" } ]
        rlvr_data.append({
            "messages": message,
            "ground_truth": example["golden_answers"][0],
            "dataset": "re_search"})
    
    # upload to huggingface
    dataset = datasets.Dataset.from_list(rlvr_data)
    dataset.push_to_hub("rulins/2wiki_rlvr" + ("_no_prompt" if no_prompt else ""), split="test")


if __name__ == "__main__":
    convert_hotpotqa_to_rlvr_format()
    convert_nq_to_rlvr_format_with_context()
    convert_tqa_to_rlvr_format()
    convert_2wiki_to_rlvr_format()