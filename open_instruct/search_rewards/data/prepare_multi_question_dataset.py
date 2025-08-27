"""
Generate synthetic multi-question dataset by combining questions from seed datasets.
Each example contains two questions that must both be answered correctly.
"""

import os
import random
import pandas as pd
from datasets import Dataset
from typing import List, Dict, Any


MULTI_QUESTION_PROMPT = """You are a research assistant that answers multiple questions through iterative reasoning and research.

PROCESS:
- Use <think></think> tags to show your reasoning at any point
- Use <search>query</search> tags when you need information
- You can alternate between thinking and searching multiple times
- Answer each question sequentially, one at a time

SEARCH RESULTS:
- Results appear as: <snippet id=UNIQUE_ID>content</snippet>

MULTI-QUESTION ANSWERING FORMAT:
- You will be given multiple questions to answer
- Answer the FIRST question completely, then move to the SECOND question
- For the first question: Use <answer1>\\boxed{{your_answer}}</answer1>
- For the second question: Use <answer2>\\boxed{{your_answer}}</answer2>
- Each answer should be concise and placed within the \\boxed{{}} format
- Both answers must be correct for the response to be considered successful

WORKFLOW EXAMPLE:
<think>I need to research the first question about renewable energy</think>
<search>2024 renewable energy market trends</search>
[results provided]
<answer1>\\boxed{{15.2%}}</answer1>

<think>Now I need to research the second question about solar panels</think>
<search>latest solar panel efficiency 2024</search>
[results provided]
<answer2>\\boxed{{March 2024}}</answer2>

REQUIREMENTS:
- Answer each question in sequence (Question 1 first, then Question 2)
- Use <answer1>\\boxed{{answer}}</answer1> for the first question
- Use <answer2>\\boxed{{answer}}</answer2> for the second question
- Think and search as needed for each question before providing its answer
- Ensure both answers are complete and correct

{question}
"""


def load_seed_datasets():
    """Load the seed datasets (HotpotQA and 2Wiki) using direct parquet loading."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    
    print("Loading HotpotQA dataset...")
    try:
        # Download parquet file directly
        hotpot_file = hf_hub_download(
            repo_id='rulins/hotpotqa_rlvr_no_prompt', 
            filename='data/test-00000-of-00001.parquet', 
            repo_type='dataset'
        )
        df_hotpot = pd.read_parquet(hotpot_file)
        hotpotqa = Dataset.from_pandas(df_hotpot)
        print(f"HotpotQA loaded: {len(hotpotqa)} examples")
    except Exception as e:
        print(f"Error loading HotpotQA: {e}")
        raise
    
    print("Loading 2Wiki dataset...")
    try:
        # Download parquet file directly
        wiki2_file = hf_hub_download(
            repo_id='rulins/2wiki_rlvr_no_prompt', 
            filename='data/test-00000-of-00001.parquet', 
            repo_type='dataset'
        )
        df_wiki2 = pd.read_parquet(wiki2_file)
        wiki2 = Dataset.from_pandas(df_wiki2)
        print(f"2Wiki loaded: {len(wiki2)} examples")
    except Exception as e:
        print(f"Error loading 2Wiki: {e}")
        raise
    
    return hotpotqa, wiki2


def create_multi_question_examples(hotpotqa_dataset, wiki2_dataset, num_examples: int = 1000, seed: int = 42):
    """
    Create multi-question examples by pairing questions from different datasets.
    
    Args:
        hotpotqa_dataset: HotpotQA dataset
        wiki2_dataset: 2Wiki dataset  
        num_examples: Number of multi-question examples to generate
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Convert to lists for easier sampling
    hotpotqa_list = list(hotpotqa_dataset)
    wiki2_list = list(wiki2_dataset)
    
    multi_question_examples = []
    
    print(f"Generating {num_examples} multi-question examples...")
    
    for i in range(num_examples):
        # Randomly sample one question from each dataset
        hotpotqa_sample = random.choice(hotpotqa_list)
        wiki2_sample = random.choice(wiki2_list)
        
        # Extract questions and ground truth answers
        hotpotqa_question = hotpotqa_sample["messages"][0]["content"]
        hotpotqa_answer = hotpotqa_sample["ground_truth"]
        
        wiki2_question = wiki2_sample["messages"][0]["content"]
        wiki2_answer = wiki2_sample["ground_truth"]
        
        # Create combined question
        combined_question = f"Question 1: {hotpotqa_question}\nQuestion 2: {wiki2_question}"
        
        # Format the prompt
        formatted_prompt = MULTI_QUESTION_PROMPT.format(question=combined_question)
        
        # Create combined ground truth in the required format
        combined_ground_truth = f"{hotpotqa_answer}; {wiki2_answer}"
        
        # Create the example in the same format as the original datasets
        example = {
            "messages": [{"role": "user", "content": formatted_prompt}],
            "ground_truth": combined_ground_truth,
            "dataset": "rl_rag_toy_case_multi_dataset_finegrained",
            "source_datasets": ["hotpotqa", "2wiki"],
            "individual_ground_truths": [hotpotqa_answer, wiki2_answer],
            "individual_questions": [hotpotqa_question, wiki2_question]
        }
        
        multi_question_examples.append(example)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_examples} examples")
    
    return multi_question_examples


def create_multi_question_dataset(num_examples: int = 1000, seed: int = 42, push_to_hub: bool = False, hub_id: str = None):
    """
    Create and optionally push the multi-question dataset.
    
    Args:
        num_examples: Number of examples to generate
        seed: Random seed for reproducibility
        push_to_hub: Whether to push to Hugging Face Hub
        hub_id: Hub ID for pushing (e.g., "username/dataset_name")
    """
    # Load seed datasets
    hotpotqa, wiki2 = load_seed_datasets()
    
    # Create multi-question examples
    examples = create_multi_question_examples(hotpotqa, wiki2, num_examples, seed)
    
    # Create dataset
    dataset = Dataset.from_list(examples)
    
    print(f"Created dataset with {len(dataset)} examples")
    print("\nSample example:")
    print("Question:", dataset[0]["messages"][0]["content"][:200] + "...")
    print("Ground truth:", dataset[0]["ground_truth"])
    print("Individual questions:", dataset[0]["individual_questions"])
    print("Individual ground truths:", dataset[0]["individual_ground_truths"])
    
    # Optionally push to hub
    if push_to_hub and hub_id:
        print(f"Pushing dataset to {hub_id}...")
        dataset.push_to_hub(hub_id)
        print("Dataset pushed successfully!")
    
    return dataset


def create_balanced_multi_question_dataset(num_examples: int = 1000, seed: int = 42, push_to_hub: bool = False, hub_id: str = None, dataset_name: str = "rl_rag_toy_case_multi_dataset_finegrained"):
    """
    Create a balanced multi-question dataset with equal representation from both datasets.
    This version ensures we get exactly num_examples/2 from each pairing direction.
    """
    random.seed(seed)
    
    # Load seed datasets
    hotpotqa, wiki2 = load_seed_datasets()
    
    # Convert to lists
    hotpotqa_list = list(hotpotqa)
    wiki2_list = list(wiki2)
    
    multi_question_examples = []
    
    print(f"Generating {num_examples} balanced multi-question examples...")
    
    # Generate half with HotpotQA first, half with 2Wiki first
    half_examples = num_examples // 2
    
    # First half: HotpotQA as Question 1, 2Wiki as Question 2
    for i in range(half_examples):
        hotpotqa_sample = random.choice(hotpotqa_list)
        wiki2_sample = random.choice(wiki2_list)
        
        hotpotqa_question = hotpotqa_sample["messages"][0]["content"]
        hotpotqa_answer = hotpotqa_sample["ground_truth"]
        
        wiki2_question = wiki2_sample["messages"][0]["content"]
        wiki2_answer = wiki2_sample["ground_truth"]
        
        combined_question = f"Question 1: {hotpotqa_question}\nQuestion 2: {wiki2_question}"
        formatted_prompt = MULTI_QUESTION_PROMPT.format(question=combined_question)
        combined_ground_truth = f"{hotpotqa_answer}; {wiki2_answer}"
        
        example = {
            "messages": [{"role": "user", "content": formatted_prompt}],
            "ground_truth": combined_ground_truth,
            "dataset": dataset_name,
            "source_datasets": ["hotpotqa", "2wiki"],
            "individual_ground_truths": [hotpotqa_answer, wiki2_answer],
            "individual_questions": [hotpotqa_question, wiki2_question],
            "question_order": "hotpotqa_first"
        }
        
        multi_question_examples.append(example)
    
    # Second half: 2Wiki as Question 1, HotpotQA as Question 2
    remaining_examples = num_examples - half_examples
    for i in range(remaining_examples):
        hotpotqa_sample = random.choice(hotpotqa_list)
        wiki2_sample = random.choice(wiki2_list)
        
        hotpotqa_question = hotpotqa_sample["messages"][0]["content"]
        hotpotqa_answer = hotpotqa_sample["ground_truth"]
        
        wiki2_question = wiki2_sample["messages"][0]["content"]
        wiki2_answer = wiki2_sample["ground_truth"]
        
        combined_question = f"Question 1: {wiki2_question}\nQuestion 2: {hotpotqa_question}"
        formatted_prompt = MULTI_QUESTION_PROMPT.format(question=combined_question)
        combined_ground_truth = f"{wiki2_answer}; {hotpotqa_answer}"
        
        example = {
            "messages": [{"role": "user", "content": formatted_prompt}],
            "ground_truth": combined_ground_truth,
            "dataset": dataset_name,
            "source_datasets": ["2wiki", "hotpotqa"],
            "individual_ground_truths": [wiki2_answer, hotpotqa_answer],
            "individual_questions": [wiki2_question, hotpotqa_question],
            "question_order": "2wiki_first"
        }
        
        multi_question_examples.append(example)
    
    # Shuffle the final list
    random.shuffle(multi_question_examples)
    
    # Create dataset
    dataset = Dataset.from_list(multi_question_examples)
    
    print(f"Created balanced dataset with {len(dataset)} examples")
    print(f"- {half_examples} examples with HotpotQA first")
    print(f"- {remaining_examples} examples with 2Wiki first")
    
    print("\nSample example:")
    print("Question:", dataset[0]["messages"][0]["content"][:300] + "...")
    print("Ground truth:", dataset[0]["ground_truth"])
    print("Question order:", dataset[0]["question_order"])
    
    # Optionally push to hub
    if push_to_hub and hub_id:
        print(f"Pushing dataset to {hub_id}...")
        dataset.push_to_hub(hub_id)
        print("Dataset pushed successfully!")
    
    return dataset


if __name__ == "__main__":
    # Create a balanced dataset with 1000 examples and push to HF Hub
    dataset = create_balanced_multi_question_dataset(
        num_examples=1000,
        seed=42,
        push_to_hub=True,  # Push to hub
        hub_id="rulins/multi_question_synthetic_concat_of_2wiki_and_hotpotqa_finegrained",
        dataset_name="rl_rag_toy_case_multi_dataset_finegrained",
    )
    
    dataset = create_balanced_multi_question_dataset(
        num_examples=1000,
        seed=42,
        push_to_hub=True,  # Push to hub
        hub_id="rulins/multi_question_synthetic_concat_of_2wiki_and_hotpotqa_averaged",
        dataset_name="rl_rag_toy_case_multi_dataset_averaged",
    )
    
    print(f"\nDataset created and pushed successfully with {len(dataset)} examples!")
    print("Dataset available at: https://huggingface.co/datasets/rulins/multi_question_synthetic_concat_of_2wiki_and_hotpotqa")
