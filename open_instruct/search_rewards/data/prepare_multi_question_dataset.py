"""
Generate synthetic multi-question dataset by combining questions from seed datasets.
Each example contains two questions that must both be answered correctly.
"""

import os
import json
import random
import pandas as pd
from datasets import Dataset, load_dataset
from typing import List, Dict, Any


def generate_multi_question_prompt(num_questions: int = 2):
    """Generate a multi-question prompt template for variable number of questions."""
    _ = num_questions  # kept for backward compatibility but no longer used dynamically

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
        "- You will be given multiple questions (Question 1, Question 2, ..., Question N)",
        "- Answer each question completely before moving to the next",
        "- For each question k: Use <answerk>\\boxed{{your_answer_k}}</answerk>",
        "- Each answer should be concise and placed within the \\boxed{} format",
        "- All answers must be correct for the response to be considered successful",
        "",
        "WORKFLOW EXAMPLE:",
        "<think>I need to research Question k</think>",
        "<search>relevant query for Question k</search>",
        "[results provided]",
        "<answerk>\\boxed{{example_answer_k}}</answerk>",
        "...",
        "",
        "REQUIREMENTS:",
        "- Answer questions in order: Question 1 → Question 2 → ... → Question N",
        "- Think and search as needed before providing each answer",
        "- Ensure every <answerk> tag contains exactly one boxed answer",
        "- Verify that all answers are complete and correct",
        "",
        "{question}"
    ]

    return "\n".join(prompt_parts)


MULTI_QUESTION_PROMPT = generate_multi_question_prompt(2)  # Default 2-question prompt for backward compatibility


def load_asearcher_base_dataset(cache_dir: str = None):
    """Load the ASearcher Base dataset with short-form QA pairs."""

    data_file = "hf://datasets/inclusionAI/ASearcher-train-data/ASearcher-Base-35k.jsonl"

    dataset_kwargs = {"data_files": data_file}
    if cache_dir is not None:
        dataset_kwargs["cache_dir"] = cache_dir

    print("Loading ASearcher Base dataset (ASearcherBase35k split)...")
    ds = load_dataset("json", split="train", **dataset_kwargs)
    print(f"ASearcher Base loaded: {len(ds)} examples")
    return ds


def _extract_asearcher_answers(sample: Dict[str, Any]) -> List[str]:
    """Normalize answer aliases from an ASearcher sample."""

    def _normalize(values) -> List[str]:
        if values is None:
            return []
        if isinstance(values, str):
            values = [values]

        normalized = []
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                normalized.append(text)

        seen = set()
        deduped = []
        for text in normalized:
            if text not in seen:
                seen.add(text)
                deduped.append(text)
        return deduped

    aug_answers = _normalize(sample.get("aug_answer"))
    if aug_answers:
        return aug_answers

    return _normalize(sample.get("answer"))


def _prepare_asearcher_samples(asearcher_dataset) -> List[Dict[str, Any]]:
    """Convert raw ASearcher records into a cleaned list of question/answer dicts."""

    processed_samples: List[Dict[str, Any]] = []

    for raw_sample in asearcher_dataset:
        question_text = str(raw_sample.get("question", "")).strip()
        if not question_text:
            continue

        answers = _extract_asearcher_answers(raw_sample)
        if not answers:
            continue

        processed_samples.append(
            {
                "question": question_text,
                "answers": answers,
                "qid": raw_sample.get("qid"),
                "source": raw_sample.get("source"),
                "id": raw_sample.get("id"),
                "idx": raw_sample.get("idx"),
            }
        )

    if not processed_samples:
        raise ValueError(
            "No valid samples found in ASearcher dataset with non-empty question and aug_answer."
        )

    return processed_samples


def create_multi_question_examples_from_asearcher_base(
    samples: List[Dict[str, Any]],
    num_examples: int = 1000,
    seed: int = 42,
    num_questions: int = 2,
    dataset_name: str = "asearcher_multi_question_base",
    include_metadata: bool = True,
):
    """
    Create multi-question examples from the ASearcher Base split.

    Args:
        samples: List of cleaned ASearcher samples to draw questions from
        num_examples: Number of multi-question examples to generate. If -1, generate a full pass over the dataset.
        seed: Random seed for reproducibility
        num_questions: Number of questions to concatenate per example (sampled with replacement)
        dataset_name: Name recorded in the generated examples
    """

    if not samples:
        raise ValueError("No samples provided for ASearcher multi-question generation.")

    if num_questions <= 0:
        raise ValueError("num_questions must be a positive integer")

    if num_examples == -1:
        num_examples = len(samples)
        print(f"num_examples set to -1: generating {num_examples} examples (one pass with replacement)")

    if num_examples <= 0:
        return []

    rng = random.Random(seed)

    prompt_template = generate_multi_question_prompt(num_questions)
    multi_question_examples = []

    print(f"Generating {num_examples} multi-question examples with {num_questions} questions each from ASearcher Base...")

    for i in range(num_examples):
        selected_samples = rng.choices(samples, k=num_questions)

        questions = []
        answer_aliases = []
        primary_answers = []
        question_metadata = []

        for sample in selected_samples:
            questions.append(sample["question"])
            answer_aliases.append(sample["answers"])
            primary_answers.append(sample["answers"][0] if sample["answers"] else "")
            question_metadata.append(
                {
                    "qid": sample.get("qid"),
                    "source": sample.get("source"),
                    "id": sample.get("id"),
                    "idx": sample.get("idx"),
                }
            )

        combined_question_parts = [f"Question {idx}: {text}" for idx, text in enumerate(questions, 1)]
        combined_question = "\n".join(combined_question_parts)

        formatted_prompt = prompt_template.replace("{question}", combined_question)
        combined_ground_truth = json.dumps(answer_aliases)

        example: Dict[str, Any] = {
            "messages": [{"role": "user", "content": formatted_prompt}],
            "ground_truth": combined_ground_truth,
            "dataset": dataset_name,
            "source_datasets": ["ASearcherBase35k"],
            "individual_ground_truths": primary_answers,
            "individual_answer_aliases": answer_aliases,
            "individual_questions": questions,
            "num_questions": num_questions,
        }

        if include_metadata:
            example["question_metadata"] = question_metadata

        multi_question_examples.append(example)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_examples} examples")

    return multi_question_examples


def create_multi_question_dataset_from_asearcher_base(
    num_examples: int = 1000,
    seed: int = 42,
    num_questions: int = 2,
    include_metadata: bool = True,
    push_to_hub: bool = False,
    hub_id: str = None,
    dataset_name: str = "asearcher_multi_question_base",
    val_size: int = 1000,
    cache_dir: str = None,
):
    """
    Create a multi-question dataset from the ASearcher Base split.

    Args:
        num_examples: Number of examples to generate. If -1, generate maximum possible examples.
        seed: Random seed for reproducibility.
        num_questions: Number of questions to concatenate in each example.
        include_metadata: Whether to keep per-question metadata (qid, source, etc.) in each example.
        push_to_hub: Whether to push the resulting dataset to the Hugging Face Hub.
        hub_id: Hub ID for pushing (e.g., "username/dataset_name").
        dataset_name: Name recorded inside each generated example.
        val_size: Number of examples to reserve for validation. If the dataset is too small, no split is created.
        cache_dir: Optional cache directory passed to dataset loader.
    """

    base_dataset = load_asearcher_base_dataset(cache_dir=cache_dir)
    processed_samples = _prepare_asearcher_samples(base_dataset)

    # Deterministic shuffle for splitting pools
    shuffle_rng = random.Random(seed)
    samples_copy = processed_samples.copy()
    shuffle_rng.shuffle(samples_copy)

    dev_examples = []
    val_dataset = None

    if val_size is not None and val_size > 0 and len(samples_copy) > 1:
        total_requested = max(num_examples, 0) + val_size
        if total_requested <= 0:
            total_requested = len(samples_copy)

        dev_fraction = min(0.5, max(0.1, val_size / total_requested))
        dev_pool_size = max(1, int(round(len(samples_copy) * dev_fraction)))
        if dev_pool_size >= len(samples_copy):
            dev_pool_size = len(samples_copy) - 1

        dev_samples = samples_copy[:dev_pool_size]
        train_samples = samples_copy[dev_pool_size:]

        if not train_samples:
            train_samples = dev_samples

        if val_size > 0:
            dev_examples = create_multi_question_examples_from_asearcher_base(
                dev_samples,
                num_examples=val_size,
                seed=seed + 1,
                num_questions=num_questions,
                dataset_name=dataset_name,
                include_metadata=include_metadata,
            )
    else:
        train_samples = samples_copy
        dev_samples = []

    train_examples = create_multi_question_examples_from_asearcher_base(
        train_samples,
        num_examples=num_examples,
        seed=seed,
        num_questions=num_questions,
        dataset_name=dataset_name,
        include_metadata=include_metadata,
    )

    train_dataset = Dataset.from_list(train_examples)

    if dev_examples:
        val_dataset = Dataset.from_list(dev_examples)
        print(
            f"Created ASearcher Base multi-question dataset with {len(train_dataset)} train examples "
            f"and {len(val_dataset)} validation examples"
        )
    else:
        print(
            f"Created ASearcher Base multi-question dataset with {len(train_dataset)} train examples. "
            "Validation split not created."
        )

    print("\nSample example:")
    preview = train_dataset[0]
    print("Question:", preview["messages"][0]["content"][:300] + "...")
    print("Ground truth (aliases):", preview["ground_truth"])
    print("Individual primary answers:", preview["individual_ground_truths"])
    if include_metadata and "question_metadata" in preview:
        print("Metadata for first question:", preview["question_metadata"][0])

    print("\nFull training example before upload:")
    print(json.dumps(preview, indent=2))

    if val_dataset is not None and len(val_dataset) > 0:
        val_preview = val_dataset[0]
        print("\nValidation sample:")
        print("Question:", val_preview["messages"][0]["content"][:300] + "...")
        print("Individual primary answers:", val_preview["individual_ground_truths"])

    if push_to_hub and hub_id:
        from datasets import DatasetDict

        dataset_splits = {"train": train_dataset}
        if val_dataset is not None:
            dataset_splits["validation"] = val_dataset

        dataset_dict = DatasetDict(dataset_splits)
        dataset_dict.push_to_hub(hub_id)
        splits = ", ".join(f"{name} ({len(ds)})" for name, ds in dataset_splits.items())
        print(f"Dataset pushed to {hub_id} with splits: {splits}")

    result = {"train": train_dataset}
    if val_dataset is not None:
        result["validation"] = val_dataset

    return result


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
    
    # Calculate maximum possible examples
    max_possible_examples = len(tqa_list) // num_questions
    
    # Handle num_examples = -1 (generate maximum possible)
    if num_examples == -1:
        num_examples = max_possible_examples
        print(f"num_examples set to -1: generating maximum possible examples = {num_examples}")
    else:
        # Check if we have enough questions after filtering
        total_questions_needed = num_examples * num_questions
        if len(tqa_list) < total_questions_needed:
            raise ValueError(f"Not enough questions in TQA dataset after filtering. Need {total_questions_needed} but only have {len(filtered_tqa_list)}. Maximum possible examples: {max_possible_examples}")
    
    # Shuffle the filtered dataset to ensure randomness
    random.shuffle(tqa_list)
    
    multi_question_examples = []
    
    print(f"Generating {num_examples} multi-question examples with {num_questions} questions each from TQA dataset...")
    
    # Generate the prompt template for this number of questions
    prompt_template = generate_multi_question_prompt(num_questions)
    
    for i in range(num_examples):
        # Sample num_questions without overlap
        start_idx = i * num_questions
        end_idx = start_idx + num_questions
        selected_questions = tqa_list[start_idx:end_idx]
        
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
        
        # Create combined ground truth (json-separated)
        combined_ground_truth = json.dumps(answers)
        
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
    
    # Calculate maximum possible examples
    max_possible_examples = len(wiki2_list) // num_questions
    
    # Handle num_examples = -1 (generate maximum possible)
    if num_examples == -1:
        num_examples = max_possible_examples
        print(f"num_examples set to -1: generating maximum possible examples = {num_examples}")
    else:
        # Check if we have enough questions after filtering
        total_questions_needed = num_examples * num_questions
        if len(wiki2_list) < total_questions_needed:
            raise ValueError(f"Not enough questions in 2Wiki dataset after filtering. Need {total_questions_needed} but only have {len(filtered_wiki2_list)}. Maximum possible examples: {max_possible_examples}")
    
    # Shuffle the filtered dataset to ensure randomness
    random.shuffle(wiki2_list)
    
    multi_question_examples = []
    
    print(f"Generating {num_examples} multi-question examples with {num_questions} questions each from 2Wiki dataset...")
    
    # Generate the prompt template for this number of questions
    prompt_template = generate_multi_question_prompt(num_questions)
    
    for i in range(num_examples):
        # Sample num_questions without overlap
        start_idx = i * num_questions
        end_idx = start_idx + num_questions
        selected_questions = wiki2_list[start_idx:end_idx]
        
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
        
        # Create combined ground truth (json-separated)
        combined_ground_truth = json.dumps(answers)
        
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


# if __name__ == "__main__":
    # # Example 1: Create a balanced dataset with 2 different datasets (existing functionality)
    # print("=== Creating balanced multi-question dataset from 2 datasets ===")
    # dataset = create_balanced_multi_question_dataset(
    #     num_examples=1000,
    #     seed=42,
    #     push_to_hub=False,  # Set to True to push to hub
    #     hub_id="rulins/multi_question_synthetic_concat_of_2wiki_and_hotpotqa_finegrained",
    #     dataset_name="rl_rag_toy_case_multi_dataset_finegrained",
    # )
    
    # # Example 2: Create multi-question dataset from single TQA source (NEW functionality)
    # print("\n=== Creating multi-question dataset from single TQA source ===")
    
    # # Create dataset with 2 questions per example from TQA
    # tqa_2q_datasets = create_multi_question_dataset_from_tqa(
    #     num_examples=-1,
    #     seed=42,
    #     num_questions=2,  # 2 questions per example
    #     push_to_hub=True,  # Set to True to push to hub
    #     hub_id="rulins/multi_question_synthetic_single_source_tqa_2q",
    #     dataset_name="rl_rag_toy_case_multi_dataset_finegrained",
    #     val_size=100  # 300 examples for validation
    # )
    
    # # Create dataset with 5 questions per example from TQA (maximum possible examples)
    # print("\n=== Creating 5-question dataset from TQA (maximum examples) ===")
    # tqa_5q_datasets = create_multi_question_dataset_from_tqa(
    #     num_examples=-1,  # Generate maximum possible examples until no more samples
    #     seed=42,
    #     num_questions=5,  # 5 questions per example
    #     push_to_hub=True,  # Set to True to push to hub
    #     hub_id="rulins/multi_question_synthetic_single_source_tqa_5q",
    #     dataset_name="rl_rag_toy_case_multi_dataset_finegrained",
    #     val_size=100  # 300 examples for validation
    # )
    
    # print(f"\n=== TQA Dataset Summary ===")
    # tqa_2q_train_size = len(tqa_2q_datasets['train'])
    # tqa_2q_val_size = len(tqa_2q_datasets['validation']) if 'validation' in tqa_2q_datasets else 0
    # tqa_2q_test_size = len(tqa_2q_datasets['test'])
    # tqa_2q_total = tqa_2q_train_size + tqa_2q_val_size + tqa_2q_test_size
    
    # tqa_5q_train_size = len(tqa_5q_datasets['train'])
    # tqa_5q_val_size = len(tqa_5q_datasets['validation']) if 'validation' in tqa_5q_datasets else 0
    # tqa_5q_test_size = len(tqa_5q_datasets['test'])
    # tqa_5q_total = tqa_5q_train_size + tqa_5q_val_size + tqa_5q_test_size
    
    # print(f"Single source TQA (2Q): {tqa_2q_total} total examples ({tqa_2q_train_size} train, {tqa_2q_val_size} val, {tqa_2q_test_size} test), {tqa_2q_total * 2} total questions used")
    # print(f"Single source TQA (5Q): {tqa_5q_total} total examples ({tqa_5q_train_size} train, {tqa_5q_val_size} val, {tqa_5q_test_size} test), {tqa_5q_total * 5} total questions used")
    # print(f"Total TQA questions used: {tqa_2q_total * 2 + tqa_5q_total * 5}")
    
    # # Example 3: Create multi-question dataset from single 2Wiki source (NEW functionality)
    # print("\n=== Creating multi-question dataset from single 2Wiki source ===")
    
    # # Create dataset with 2 questions per example from 2Wiki
    # wiki2_2q_datasets = create_multi_question_dataset_from_2wiki(
    #     num_examples=-1,
    #     seed=42,
    #     num_questions=2,  # 2 questions per example
    #     push_to_hub=True,  # Set to True to push to hub
    #     hub_id="rulins/multi_question_synthetic_single_source_2wiki_2q",
    #     dataset_name="rl_rag_toy_case_multi_dataset_finegrained",
    #     val_size=100  # 300 examples for validation
    # )
    
    # # Create dataset with 5 questions per example from 2Wiki (maximum possible examples)
    # print("\n=== Creating 5-question dataset from 2Wiki (maximum examples) ===")
    # wiki2_5q_datasets = create_multi_question_dataset_from_2wiki(
    #     num_examples=-1,  # Generate maximum possible examples until no more samples
    #     seed=42,
    #     num_questions=5,  # 5 questions per example
    #     push_to_hub=True,  # Set to True to push to hub
    #     hub_id="rulins/multi_question_synthetic_single_source_2wiki_5q",
    #     dataset_name="rl_rag_toy_case_multi_dataset_finegrained",
    #     val_size=100  # 300 examples for validation
    # )
    
    # print(f"\n=== 2Wiki Dataset Summary ===")
    # wiki2_2q_train_size = len(wiki2_2q_datasets['train'])
    # wiki2_2q_val_size = len(wiki2_2q_datasets['validation']) if 'validation' in wiki2_2q_datasets else 0
    # wiki2_2q_test_size = len(wiki2_2q_datasets['test'])
    # wiki2_2q_total = wiki2_2q_train_size + wiki2_2q_val_size + wiki2_2q_test_size
    
    # wiki2_5q_train_size = len(wiki2_5q_datasets['train'])
    # wiki2_5q_val_size = len(wiki2_5q_datasets['validation']) if 'validation' in wiki2_5q_datasets else 0
    # wiki2_5q_test_size = len(wiki2_5q_datasets['test'])
    # wiki2_5q_total = wiki2_5q_train_size + wiki2_5q_val_size + wiki2_5q_test_size
    
    # print(f"Single source 2Wiki (2Q): {wiki2_2q_total} total examples ({wiki2_2q_train_size} train, {wiki2_2q_val_size} val, {wiki2_2q_test_size} test), {wiki2_2q_total * 2} total questions used")
    # print(f"Single source 2Wiki (5Q): {wiki2_5q_total} total examples ({wiki2_5q_train_size} train, {wiki2_5q_val_size} val, {wiki2_5q_test_size} test), {wiki2_5q_total * 5} total questions used")
    # print(f"Total 2Wiki questions used: {wiki2_2q_total * 2 + wiki2_5q_total * 5}")
    
    # print(f"\n=== Overall Summary ===")
    # print(f"Total TQA questions used: {tqa_2q_total * 2 + tqa_5q_total * 5}")
    # print(f"Total 2Wiki questions used: {wiki2_2q_total * 2 + wiki2_5q_total * 5}")
    # print(f"Grand total questions used: {(tqa_2q_total * 2 + tqa_5q_total * 5) + (wiki2_2q_total * 2 + wiki2_5q_total * 5)}")
    
    # print("\nTo push datasets to hub, set push_to_hub=True in the function calls above.")
    # print("\nNote: Setting num_examples=-1 will generate the maximum possible number of examples")
    # print("until there are no more training samples to unpack from the source dataset.")
    # print(f"Each dataset is automatically split into train/validation with {300} examples reserved for validation.")
    # print("Test sets are created using ALL available test samples from the original datasets.")


if __name__ == "__main__":
    num_questions = 5
    hub_id = f"rulins/multi_question_synthetic_single_source_asearcher_base_{num_questions}q"
    asearcher_base_dataset = create_multi_question_dataset_from_asearcher_base(
        num_examples=1000,
        seed=42,
        num_questions=num_questions,
        push_to_hub=True,
        hub_id=hub_id,
    )
