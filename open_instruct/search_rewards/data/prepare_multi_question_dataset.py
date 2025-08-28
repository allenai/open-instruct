"""
Generate synthetic multi-question dataset by combining questions from seed datasets.
Each example contains two questions that must both be answered correctly.
"""

import os
import random
import pandas as pd
from datasets import Dataset
from typing import List, Dict, Any


def generate_multi_question_prompt(num_questions: int = 2):
    """Generate a multi-question prompt template for variable number of questions."""
    
    # Generate answer format instructions
    answer_format_lines = []
    for i in range(1, num_questions + 1):
        if i == 1:
            answer_format_lines.append(f"- For the first question: Use <answer{i}>\\boxed{{{{your_answer}}}}</answer{i}>")
        elif i == num_questions:
            answer_format_lines.append(f"- For the final question: Use <answer{i}>\\boxed{{{{your_answer}}}}</answer{i}>")
        else:
            answer_format_lines.append(f"- For question {i}: Use <answer{i}>\\boxed{{{{your_answer}}}}</answer{i}>")
    
    # Generate workflow example
    workflow_examples = []
    for i in range(1, num_questions + 1):
        workflow_examples.extend([
            f"<think>I need to research question {i}</think>",
            f"<search>relevant query for question {i}</search>",
            "[results provided]",
            f"<answer{i}>\\boxed{{{{example_answer_{i}}}}}</answer{i}>",
            ""
        ])
    
    # Generate requirements
    requirements_lines = []
    sequence_text = "Question 1"
    if num_questions > 2:
        sequence_text += f", then Question 2, ..., then Question {num_questions}"
    elif num_questions == 2:
        sequence_text += ", then Question 2"
    
    requirements_lines.append(f"- Answer each question in sequence ({sequence_text})")
    
    for i in range(1, num_questions + 1):
        requirements_lines.append(f"- Use <answer{i}>\\boxed{{{{answer}}}}</answer{i}> for question {i}")
    
    # Build the prompt
    prompt_parts = [
        "You are a research assistant that answers multiple questions through iterative reasoning and research.",
        "",
        "PROCESS:",
        "- Use <think></think> tags to show your reasoning at any point",
        "- Use <search>query</search> tags when you need information",
        "- You can alternate between thinking and searching multiple times",
        "- Answer each question sequentially, one at a time",
        "",
        "SEARCH RESULTS:",
        "- Results appear as: <snippet id=UNIQUE_ID>content</snippet>",
        "",
        "MULTI-QUESTION ANSWERING FORMAT:",
        f"- You will be given {num_questions} questions to answer"
    ]
    
    if num_questions == 2:
        prompt_parts.append("- Answer the FIRST question completely, then move to the SECOND question")
    else:
        prompt_parts.append(f"- Answer each question completely in sequence (1 through {num_questions})")
    
    prompt_parts.extend(answer_format_lines)
    prompt_parts.extend([
        "- Each answer should be concise and placed within the \\boxed{} format",
        f"- All {num_questions} answers must be correct for the response to be considered successful",
        "",
        "WORKFLOW EXAMPLE:"
    ])
    
    prompt_parts.extend(workflow_examples)
    prompt_parts.extend([
        "REQUIREMENTS:"
    ])
    
    prompt_parts.extend(requirements_lines)
    prompt_parts.extend([
        "- Think and search as needed for each question before providing its answer",
        "- Ensure all answers are complete and correct",
        "",
        "{question}"
    ])
    
    return "\n".join(prompt_parts)


MULTI_QUESTION_PROMPT = generate_multi_question_prompt(2)  # Default 2-question prompt for backward compatibility


def load_tqa_dataset():
    """Load the TQA dataset using direct parquet loading."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    
    print("Loading TQA dataset...")
    try:
        # Download parquet file directly
        tqa_file = hf_hub_download(
            repo_id='rulins/tqa_rlvr_no_prompt', 
            filename='data/train-00000-of-00001.parquet', 
            repo_type='dataset'
        )
        df_tqa = pd.read_parquet(tqa_file)
        tqa_dataset = Dataset.from_pandas(df_tqa)
        print(f"TQA loaded: {len(tqa_dataset)} examples")
        return tqa_dataset
    except Exception as e:
        print(f"Error loading TQA: {e}")
        raise


def load_2wiki_dataset():
    """Load the 2Wiki dataset using direct parquet loading."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    
    print("Loading 2Wiki dataset...")
    try:
        # Download parquet file directly
        wiki2_file = hf_hub_download(
            repo_id='rulins/2wiki_rlvr_no_prompt', 
            filename='data/train-00000-of-00001.parquet', 
            repo_type='dataset'
        )
        df_wiki2 = pd.read_parquet(wiki2_file)
        wiki2_dataset = Dataset.from_pandas(df_wiki2)
        print(f"2Wiki loaded: {len(wiki2_dataset)} examples")
        return wiki2_dataset
    except Exception as e:
        print(f"Error loading 2Wiki: {e}")
        raise


def load_tqa_test_dataset():
    """Load the TQA test dataset using direct parquet loading."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    
    print("Loading TQA test dataset...")
    try:
        # Download parquet file directly
        tqa_test_file = hf_hub_download(
            repo_id='rulins/tqa_rlvr_no_prompt', 
            filename='data/test-00000-of-00001.parquet', 
            repo_type='dataset'
        )
        df_tqa_test = pd.read_parquet(tqa_test_file)
        tqa_test_dataset = Dataset.from_pandas(df_tqa_test)
        print(f"TQA test loaded: {len(tqa_test_dataset)} examples")
        return tqa_test_dataset
    except Exception as e:
        print(f"Error loading TQA test: {e}")
        raise


def load_2wiki_test_dataset():
    """Load the 2Wiki test dataset using direct parquet loading."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    
    print("Loading 2Wiki test dataset...")
    try:
        # Download parquet file directly
        wiki2_test_file = hf_hub_download(
            repo_id='rulins/2wiki_rlvr_no_prompt', 
            filename='data/test-00000-of-00001.parquet', 
            repo_type='dataset'
        )
        df_wiki2_test = pd.read_parquet(wiki2_test_file)
        wiki2_test_dataset = Dataset.from_pandas(df_wiki2_test)
        print(f"2Wiki test loaded: {len(wiki2_test_dataset)} examples")
        return wiki2_test_dataset
    except Exception as e:
        print(f"Error loading 2Wiki test: {e}")
        raise


def load_seed_datasets():
    """Load the seed datasets (HotpotQA and 2Wiki) using direct parquet loading."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    
    print("Loading HotpotQA dataset...")
    try:
        # Download parquet file directly
        hotpot_file = hf_hub_download(
            repo_id='rulins/hotpotqa_rlvr_no_prompt', 
            filename='data/train-00000-of-00001.parquet', 
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
            filename='data/train-00000-of-00001.parquet', 
            repo_type='dataset'
        )
        df_wiki2 = pd.read_parquet(wiki2_file)
        wiki2 = Dataset.from_pandas(df_wiki2)
        print(f"2Wiki loaded: {len(wiki2)} examples")
    except Exception as e:
        print(f"Error loading 2Wiki: {e}")
        raise
    
    return hotpotqa, wiki2


def create_multi_question_examples_from_2wiki_and_hotpotqa(hotpotqa_dataset, wiki2_dataset, num_examples: int = 1000, seed: int = 42):
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


def create_multi_question_examples_from_tqa(tqa_dataset, num_examples: int = 1000, seed: int = 42, num_questions: int = 2):
    """
    Create multi-question examples by concatenating multiple questions from TQA dataset.
    Ensures no overlap of questions among different examples.
    Filters out samples with semicolons in ground truth to avoid parsing issues.
    
    Args:
        tqa_dataset: The TQA dataset to sample questions from
        num_examples: Number of multi-question examples to generate. If -1, generate maximum possible examples.
        seed: Random seed for reproducibility
        num_questions: Number of questions to concatenate in each example
    """
    random.seed(seed)
    
    tqa_list = list(tqa_dataset)
    
    # Filter out samples that contain semicolons in their ground truth
    # This prevents parsing issues when combining multiple ground truths
    filtered_tqa_list = []
    skipped_count = 0
    
    for sample in tqa_list:
        ground_truth = sample["ground_truth"]
        if ";" in ground_truth:
            skipped_count += 1
        else:
            filtered_tqa_list.append(sample)
    
    print(f"Filtered out {skipped_count} samples containing semicolons in ground truth")
    print(f"Remaining samples: {len(filtered_tqa_list)}")
    
    # Calculate maximum possible examples
    max_possible_examples = len(filtered_tqa_list) // num_questions
    
    # Handle num_examples = -1 (generate maximum possible)
    if num_examples == -1:
        num_examples = max_possible_examples
        print(f"num_examples set to -1: generating maximum possible examples = {num_examples}")
    else:
        # Check if we have enough questions after filtering
        total_questions_needed = num_examples * num_questions
        if len(filtered_tqa_list) < total_questions_needed:
            raise ValueError(f"Not enough questions in TQA dataset after filtering. Need {total_questions_needed} but only have {len(filtered_tqa_list)}. Maximum possible examples: {max_possible_examples}")
    
    # Shuffle the filtered dataset to ensure randomness
    random.shuffle(filtered_tqa_list)
    
    multi_question_examples = []
    
    print(f"Generating {num_examples} multi-question examples with {num_questions} questions each from TQA dataset...")
    
    # Generate the prompt template for this number of questions
    prompt_template = generate_multi_question_prompt(num_questions)
    
    for i in range(num_examples):
        # Sample num_questions without overlap
        start_idx = i * num_questions
        end_idx = start_idx + num_questions
        selected_questions = filtered_tqa_list[start_idx:end_idx]
        
        # Extract questions and ground truth answers
        questions = []
        answers = []
        
        for j, sample in enumerate(selected_questions):
            question_content = sample["messages"][0]["content"]
            ground_truth = sample["ground_truth"]
            
            questions.append(question_content)
            answers.append(ground_truth)
        
        # Create combined question
        combined_question_parts = []
        for j, question in enumerate(questions, 1):
            combined_question_parts.append(f"Question {j}: {question}")
        
        combined_question = "\n".join(combined_question_parts)
        
        # Format the prompt (use replace instead of format to avoid brace conflicts)
        formatted_prompt = prompt_template.replace("{question}", combined_question)
        
        # Create combined ground truth (semicolon-separated)
        # Note: We've filtered out samples with semicolons to avoid parsing conflicts
        combined_ground_truth = "; ".join(answers)
        
        # Create the example in the same format as the original datasets
        example = {
            "messages": [{"role": "user", "content": formatted_prompt}],
            "ground_truth": combined_ground_truth,
            "dataset": "rl_rag_toy_case_multi_dataset_single_source_tqa",
            "source_datasets": ["tqa"],
            "individual_ground_truths": answers,
            "individual_questions": questions,
            "num_questions": num_questions,
            "question_indices": list(range(start_idx, end_idx))  # Track which original questions were used
        }
        
        multi_question_examples.append(example)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_examples} examples")
    
    return multi_question_examples


def create_multi_question_examples_from_2wiki(wiki2_dataset, num_examples: int = 1000, seed: int = 42, num_questions: int = 2):
    """
    Create multi-question examples by concatenating multiple questions from 2Wiki dataset.
    Ensures no overlap of questions among different examples.
    Filters out samples with semicolons in ground truth to avoid parsing issues.
    
    Args:
        wiki2_dataset: The 2Wiki dataset to sample questions from
        num_examples: Number of multi-question examples to generate. If -1, generate maximum possible examples.
        seed: Random seed for reproducibility
        num_questions: Number of questions to concatenate in each example
    """
    random.seed(seed)
    
    wiki2_list = list(wiki2_dataset)
    
    # Filter out samples that contain semicolons in their ground truth
    # This prevents parsing issues when combining multiple ground truths
    filtered_wiki2_list = []
    skipped_count = 0
    
    for sample in wiki2_list:
        ground_truth = sample["ground_truth"]
        if ";" in ground_truth:
            skipped_count += 1
        else:
            filtered_wiki2_list.append(sample)
    
    print(f"Filtered out {skipped_count} samples containing semicolons in ground truth")
    print(f"Remaining samples: {len(filtered_wiki2_list)}")
    
    # Calculate maximum possible examples
    max_possible_examples = len(filtered_wiki2_list) // num_questions
    
    # Handle num_examples = -1 (generate maximum possible)
    if num_examples == -1:
        num_examples = max_possible_examples
        print(f"num_examples set to -1: generating maximum possible examples = {num_examples}")
    else:
        # Check if we have enough questions after filtering
        total_questions_needed = num_examples * num_questions
        if len(filtered_wiki2_list) < total_questions_needed:
            raise ValueError(f"Not enough questions in 2Wiki dataset after filtering. Need {total_questions_needed} but only have {len(filtered_wiki2_list)}. Maximum possible examples: {max_possible_examples}")
    
    # Shuffle the filtered dataset to ensure randomness
    random.shuffle(filtered_wiki2_list)
    
    multi_question_examples = []
    
    print(f"Generating {num_examples} multi-question examples with {num_questions} questions each from 2Wiki dataset...")
    
    # Generate the prompt template for this number of questions
    prompt_template = generate_multi_question_prompt(num_questions)
    
    for i in range(num_examples):
        # Sample num_questions without overlap
        start_idx = i * num_questions
        end_idx = start_idx + num_questions
        selected_questions = filtered_wiki2_list[start_idx:end_idx]
        
        # Extract questions and ground truth answers
        questions = []
        answers = []
        
        for j, sample in enumerate(selected_questions):
            question_content = sample["messages"][0]["content"]
            ground_truth = sample["ground_truth"]
            
            questions.append(question_content)
            answers.append(ground_truth)
        
        # Create combined question
        combined_question_parts = []
        for j, question in enumerate(questions, 1):
            combined_question_parts.append(f"Question {j}: {question}")
        
        combined_question = "\n".join(combined_question_parts)
        
        # Format the prompt (use replace instead of format to avoid brace conflicts)
        formatted_prompt = prompt_template.replace("{question}", combined_question)
        
        # Create combined ground truth (semicolon-separated)
        # Note: We've filtered out samples with semicolons to avoid parsing conflicts
        combined_ground_truth = "; ".join(answers)
        
        # Create the example in the same format as the original datasets
        example = {
            "messages": [{"role": "user", "content": formatted_prompt}],
            "ground_truth": combined_ground_truth,
            "dataset": "rl_rag_toy_case_multi_dataset_single_source_2wiki",
            "source_datasets": ["2wiki"],
            "individual_ground_truths": answers,
            "individual_questions": questions,
            "num_questions": num_questions,
            "question_indices": list(range(start_idx, end_idx))  # Track which original questions were used
        }
        
        multi_question_examples.append(example)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_examples} examples")
    
    return multi_question_examples


def create_multi_question_dataset_from_tqa(num_examples: int = 1000, seed: int = 42, num_questions: int = 2, push_to_hub: bool = False, hub_id: str = None, dataset_name: str = "rl_rag_toy_case_multi_dataset_single_source_tqa", val_size: int = 300):
    """
    Create a multi-question dataset from TQA dataset (single source).
    
    Args:
        num_examples: Number of examples to generate. If -1, generate maximum possible examples.
        seed: Random seed for reproducibility
        num_questions: Number of questions to concatenate in each example
        push_to_hub: Whether to push to Hugging Face Hub
        hub_id: Hub ID for pushing (e.g., "username/dataset_name")
        dataset_name: Name to assign to the dataset
        val_size: Number of examples to reserve for validation set
    """
    # Load TQA train dataset
    tqa_dataset = load_tqa_dataset()
    
    # Create multi-question examples from train set
    examples = create_multi_question_examples_from_tqa(tqa_dataset, num_examples, seed, num_questions)
    
    # Create full dataset from train examples
    full_dataset = Dataset.from_list(examples)
    
    # Split into train and validation sets
    if len(full_dataset) > val_size:
        # Use a different seed for splitting to ensure reproducibility
        split_seed = seed + 1000
        dataset_dict = full_dataset.train_test_split(test_size=val_size, seed=split_seed)
        train_dataset = dataset_dict['train']
        val_dataset = dataset_dict['test']
        
        print(f"Created TQA single-source dataset with {len(full_dataset)} total examples from train set")
        print(f"Split into: {len(train_dataset)} train examples, {len(val_dataset)} validation examples")
    else:
        print(f"Warning: Dataset has only {len(full_dataset)} examples, which is <= val_size ({val_size})")
        print("Using all examples as training set, no validation split created.")
        train_dataset = full_dataset
        val_dataset = None
    
    # Load TQA test dataset and create test examples using all test samples
    print("\nCreating test set from TQA test dataset...")
    tqa_test_dataset = load_tqa_test_dataset()
    
    # Use all test samples (set num_examples to -1 for maximum)
    test_examples = create_multi_question_examples_from_tqa(tqa_test_dataset, -1, seed, num_questions)
    test_dataset = Dataset.from_list(test_examples)
    
    print(f"Created test set with {len(test_dataset)} examples from all available test samples")
    
    print(f"\nDataset summary:")
    print(f"Each example contains {num_questions} questions")
    print(f"Train set: {len(train_dataset)} examples, {len(train_dataset) * num_questions} total questions used")
    if val_dataset is not None:
        print(f"Validation set: {len(val_dataset)} examples, {len(val_dataset) * num_questions} total questions used")
    print(f"Test set: {len(test_dataset)} examples, {len(test_dataset) * num_questions} total questions used")
    
    print("\nSample example:")
    print("Question:", train_dataset[0]["messages"][0]["content"][:300] + "...")
    print("Ground truth:", train_dataset[0]["ground_truth"])
    print("Number of questions:", train_dataset[0]["num_questions"])
    print("Individual questions:", [q[:50] + "..." for q in train_dataset[0]["individual_questions"]])
    print("Individual ground truths:", train_dataset[0]["individual_ground_truths"])
    
    # Optionally push to hub
    if push_to_hub and hub_id:
        print(f"Pushing dataset to {hub_id}...")
        # Create DatasetDict with train, validation, and test splits
        from datasets import DatasetDict
        dataset_splits = {'train': train_dataset, 'test': test_dataset}
        if val_dataset is not None:
            dataset_splits['validation'] = val_dataset
        
        dataset_dict = DatasetDict(dataset_splits)
        dataset_dict.push_to_hub(hub_id)
        
        splits_info = f"train ({len(train_dataset)}), test ({len(test_dataset)})"
        if val_dataset is not None:
            splits_info = f"train ({len(train_dataset)}), validation ({len(val_dataset)}), test ({len(test_dataset)})"
        print(f"Dataset pushed successfully with {splits_info} splits!")
    
    result = {'train': train_dataset, 'test': test_dataset}
    if val_dataset is not None:
        result['validation'] = val_dataset
    
    return result


def create_multi_question_dataset_from_2wiki(num_examples: int = 1000, seed: int = 42, num_questions: int = 2, push_to_hub: bool = False, hub_id: str = None, dataset_name: str = "rl_rag_toy_case_multi_dataset_single_source_2wiki", val_size: int = 300):
    """
    Create a multi-question dataset from 2Wiki dataset (single source).
    
    Args:
        num_examples: Number of examples to generate. If -1, generate maximum possible examples.
        seed: Random seed for reproducibility
        num_questions: Number of questions to concatenate in each example
        push_to_hub: Whether to push to Hugging Face Hub
        hub_id: Hub ID for pushing (e.g., "username/dataset_name")
        dataset_name: Name to assign to the dataset
        val_size: Number of examples to reserve for validation set
    """
    # Load 2Wiki train dataset
    wiki2_dataset = load_2wiki_dataset()
    
    # Create multi-question examples from train set
    examples = create_multi_question_examples_from_2wiki(wiki2_dataset, num_examples, seed, num_questions)
    
    # Create full dataset from train examples
    full_dataset = Dataset.from_list(examples)
    
    # Split into train and validation sets
    if len(full_dataset) > val_size:
        # Use a different seed for splitting to ensure reproducibility
        split_seed = seed + 1000
        dataset_dict = full_dataset.train_test_split(test_size=val_size, seed=split_seed)
        train_dataset = dataset_dict['train']
        val_dataset = dataset_dict['test']
        
        print(f"Created 2Wiki single-source dataset with {len(full_dataset)} total examples from train set")
        print(f"Split into: {len(train_dataset)} train examples, {len(val_dataset)} validation examples")
    else:
        print(f"Warning: Dataset has only {len(full_dataset)} examples, which is <= val_size ({val_size})")
        print("Using all examples as training set, no validation split created.")
        train_dataset = full_dataset
        val_dataset = None
    
    # Load 2Wiki test dataset and create test examples using all test samples
    print("\nCreating test set from 2Wiki test dataset...")
    wiki2_test_dataset = load_2wiki_test_dataset()
    
    # Use all test samples (set num_examples to -1 for maximum)
    test_examples = create_multi_question_examples_from_2wiki(wiki2_test_dataset, -1, seed, num_questions)
    test_dataset = Dataset.from_list(test_examples)
    
    print(f"Created test set with {len(test_dataset)} examples from all available test samples")
    
    print(f"\nDataset summary:")
    print(f"Each example contains {num_questions} questions")
    print(f"Train set: {len(train_dataset)} examples, {len(train_dataset) * num_questions} total questions used")
    if val_dataset is not None:
        print(f"Validation set: {len(val_dataset)} examples, {len(val_dataset) * num_questions} total questions used")
    print(f"Test set: {len(test_dataset)} examples, {len(test_dataset) * num_questions} total questions used")
    
    print("\nSample example:")
    print("Question:", train_dataset[0]["messages"][0]["content"][:300] + "...")
    print("Ground truth:", train_dataset[0]["ground_truth"])
    print("Number of questions:", train_dataset[0]["num_questions"])
    print("Individual questions:", [q[:50] + "..." for q in train_dataset[0]["individual_questions"]])
    print("Individual ground truths:", train_dataset[0]["individual_ground_truths"])
    
    # Optionally push to hub
    if push_to_hub and hub_id:
        print(f"Pushing dataset to {hub_id}...")
        # Create DatasetDict with train, validation, and test splits
        from datasets import DatasetDict
        dataset_splits = {'train': train_dataset, 'test': test_dataset}
        if val_dataset is not None:
            dataset_splits['validation'] = val_dataset
        
        dataset_dict = DatasetDict(dataset_splits)
        dataset_dict.push_to_hub(hub_id)
        
        splits_info = f"train ({len(train_dataset)}), test ({len(test_dataset)})"
        if val_dataset is not None:
            splits_info = f"train ({len(train_dataset)}), validation ({len(val_dataset)}), test ({len(test_dataset)})"
        print(f"Dataset pushed successfully with {splits_info} splits!")
    
    result = {'train': train_dataset, 'test': test_dataset}
    if val_dataset is not None:
        result['validation'] = val_dataset
    
    return result


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
    examples = create_multi_question_examples_from_2wiki_and_hotpotqa(hotpotqa, wiki2, num_examples, seed)
    
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
    # # Example 1: Create a balanced dataset with 2 different datasets (existing functionality)
    # print("=== Creating balanced multi-question dataset from 2 datasets ===")
    # dataset = create_balanced_multi_question_dataset(
    #     num_examples=1000,
    #     seed=42,
    #     push_to_hub=False,  # Set to True to push to hub
    #     hub_id="rulins/multi_question_synthetic_concat_of_2wiki_and_hotpotqa_finegrained",
    #     dataset_name="rl_rag_toy_case_multi_dataset_finegrained",
    # )
    
    # Example 2: Create multi-question dataset from single TQA source (NEW functionality)
    print("\n=== Creating multi-question dataset from single TQA source ===")
    
    # Create dataset with 2 questions per example from TQA
    tqa_2q_datasets = create_multi_question_dataset_from_tqa(
        num_examples=-1,
        seed=42,
        num_questions=2,  # 2 questions per example
        push_to_hub=True,  # Set to True to push to hub
        hub_id="rulins/multi_question_synthetic_single_source_tqa_2q",
        dataset_name="rl_rag_toy_case_multi_dataset_finegrained",
        val_size=300  # 300 examples for validation
    )
    
    # Create dataset with 5 questions per example from TQA (maximum possible examples)
    print("\n=== Creating 5-question dataset from TQA (maximum examples) ===")
    tqa_5q_datasets = create_multi_question_dataset_from_tqa(
        num_examples=-1,  # Generate maximum possible examples until no more samples
        seed=42,
        num_questions=5,  # 5 questions per example
        push_to_hub=True,  # Set to True to push to hub
        hub_id="rulins/multi_question_synthetic_single_source_tqa_5q",
        dataset_name="rl_rag_toy_case_multi_dataset_finegrained",
        val_size=300  # 300 examples for validation
    )
    
    print(f"\n=== TQA Dataset Summary ===")
    tqa_2q_train_size = len(tqa_2q_datasets['train'])
    tqa_2q_val_size = len(tqa_2q_datasets['validation']) if 'validation' in tqa_2q_datasets else 0
    tqa_2q_test_size = len(tqa_2q_datasets['test'])
    tqa_2q_total = tqa_2q_train_size + tqa_2q_val_size + tqa_2q_test_size
    
    tqa_5q_train_size = len(tqa_5q_datasets['train'])
    tqa_5q_val_size = len(tqa_5q_datasets['validation']) if 'validation' in tqa_5q_datasets else 0
    tqa_5q_test_size = len(tqa_5q_datasets['test'])
    tqa_5q_total = tqa_5q_train_size + tqa_5q_val_size + tqa_5q_test_size
    
    print(f"Single source TQA (2Q): {tqa_2q_total} total examples ({tqa_2q_train_size} train, {tqa_2q_val_size} val, {tqa_2q_test_size} test), {tqa_2q_total * 2} total questions used")
    print(f"Single source TQA (5Q): {tqa_5q_total} total examples ({tqa_5q_train_size} train, {tqa_5q_val_size} val, {tqa_5q_test_size} test), {tqa_5q_total * 5} total questions used")
    print(f"Total TQA questions used: {tqa_2q_total * 2 + tqa_5q_total * 5}")
    
    # Example 3: Create multi-question dataset from single 2Wiki source (NEW functionality)
    print("\n=== Creating multi-question dataset from single 2Wiki source ===")
    
    # Create dataset with 2 questions per example from 2Wiki
    wiki2_2q_datasets = create_multi_question_dataset_from_2wiki(
        num_examples=-1,
        seed=42,
        num_questions=2,  # 2 questions per example
        push_to_hub=True,  # Set to True to push to hub
        hub_id="rulins/multi_question_synthetic_single_source_2wiki_2q",
        dataset_name="rl_rag_toy_case_multi_dataset_finegrained",
        val_size=300  # 300 examples for validation
    )
    
    # Create dataset with 5 questions per example from 2Wiki (maximum possible examples)
    print("\n=== Creating 5-question dataset from 2Wiki (maximum examples) ===")
    wiki2_5q_datasets = create_multi_question_dataset_from_2wiki(
        num_examples=-1,  # Generate maximum possible examples until no more samples
        seed=42,
        num_questions=5,  # 5 questions per example
        push_to_hub=True,  # Set to True to push to hub
        hub_id="rulins/multi_question_synthetic_single_source_2wiki_5q",
        dataset_name="rl_rag_toy_case_multi_dataset_finegrained",
        val_size=300  # 300 examples for validation
    )
    
    print(f"\n=== 2Wiki Dataset Summary ===")
    wiki2_2q_train_size = len(wiki2_2q_datasets['train'])
    wiki2_2q_val_size = len(wiki2_2q_datasets['validation']) if 'validation' in wiki2_2q_datasets else 0
    wiki2_2q_test_size = len(wiki2_2q_datasets['test'])
    wiki2_2q_total = wiki2_2q_train_size + wiki2_2q_val_size + wiki2_2q_test_size
    
    wiki2_5q_train_size = len(wiki2_5q_datasets['train'])
    wiki2_5q_val_size = len(wiki2_5q_datasets['validation']) if 'validation' in wiki2_5q_datasets else 0
    wiki2_5q_test_size = len(wiki2_5q_datasets['test'])
    wiki2_5q_total = wiki2_5q_train_size + wiki2_5q_val_size + wiki2_5q_test_size
    
    print(f"Single source 2Wiki (2Q): {wiki2_2q_total} total examples ({wiki2_2q_train_size} train, {wiki2_2q_val_size} val, {wiki2_2q_test_size} test), {wiki2_2q_total * 2} total questions used")
    print(f"Single source 2Wiki (5Q): {wiki2_5q_total} total examples ({wiki2_5q_train_size} train, {wiki2_5q_val_size} val, {wiki2_5q_test_size} test), {wiki2_5q_total * 5} total questions used")
    print(f"Total 2Wiki questions used: {wiki2_2q_total * 2 + wiki2_5q_total * 5}")
    
    print(f"\n=== Overall Summary ===")
    print(f"Total TQA questions used: {tqa_2q_total * 2 + tqa_5q_total * 5}")
    print(f"Total 2Wiki questions used: {wiki2_2q_total * 2 + wiki2_5q_total * 5}")
    print(f"Grand total questions used: {(tqa_2q_total * 2 + tqa_5q_total * 5) + (wiki2_2q_total * 2 + wiki2_5q_total * 5)}")
    
    print("\nTo push datasets to hub, set push_to_hub=True in the function calls above.")
    print("\nNote: Setting num_examples=-1 will generate the maximum possible number of examples")
    print("until there are no more training samples to unpack from the source dataset.")
    print(f"Each dataset is automatically split into train/validation with {300} examples reserved for validation.")
    print("Test sets are created using ALL available test samples from the original datasets.")
