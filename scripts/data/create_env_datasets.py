#!/usr/bin/env python3
"""
Create and push environment training datasets to HuggingFace.

Usage:
    python scripts/data/create_env_datasets.py --push
"""

import argparse

from datasets import Dataset


def create_wordle_dataset(num_samples: int = 100) -> Dataset:
    """Create Wordle environment training dataset.
    
    The verifiers library handles word selection from its internal dataset,
    so we don't pass target_word. Each episode samples a new word.
    """
    examples = []
    for i in range(num_samples):
        examples.append({
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are playing Wordle. Guess a 5-letter word. After each guess, "
                        "you'll receive feedback: green (correct position), yellow (wrong position), "
                        "gray (not in word). Use the wordle tool to submit guesses."
                    ),
                },
                {
                    "role": "user", 
                    "content": "Start a new game of Wordle. You have 6 guesses to find the secret word.",
                },
            ],
            "tools": ["wordle"],
            "env_info": {
                "env_config": {},  # verifiers samples from its internal dataset
            },
            "ground_truth": "",  # Reward comes from env, not ground_truth matching
            "dataset": "env",  # verifier source - "env" means use EnvVerifier
        })
    
    return Dataset.from_list(examples)


def create_wiki_search_dataset(num_samples: int = 100) -> Dataset:
    """Create Wiki-Search environment training dataset."""
    # Sample questions that require Wikipedia search
    questions = [
        ("What year was the Eiffel Tower completed?", "1889"),
        ("Who wrote the novel '1984'?", "George Orwell"),
        ("What is the capital of Australia?", "Canberra"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("Who discovered penicillin?", "Alexander Fleming"),
        ("What year did World War II end?", "1945"),
        ("Who was the first person to walk on the moon?", "Neil Armstrong"),
        ("What is the chemical symbol for gold?", "Au"),
        ("Who wrote 'Romeo and Juliet'?", "William Shakespeare"),
        ("What is the tallest mountain in the world?", "Mount Everest"),
        ("Who invented the telephone?", "Alexander Graham Bell"),
        ("What is the largest ocean on Earth?", "Pacific Ocean"),
        ("Who was the first President of the United States?", "George Washington"),
        ("What year was the Declaration of Independence signed?", "1776"),
    ]
    
    examples = []
    for i in range(num_samples):
        question, answer = questions[i % len(questions)]
        examples.append({
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant with access to Wikipedia search. "
                        "Use the wiki_search tool to find information and answer questions accurately. "
                        "Search for relevant articles, then provide your answer."
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            "tools": ["wiki_search"],
            "env_info": {
                "env_config": {"question": question, "answer": answer},
            },
            "ground_truth": answer,
            "dataset": "env",  # verifier source - "env" means use EnvVerifier
        })
    
    return Dataset.from_list(examples)


def create_appworld_dataset(num_samples: int = 50) -> Dataset:
    """Create AppWorld environment training dataset."""
    # Sample AppWorld task descriptions (simplified)
    tasks = [
        {
            "task_id": "task_001",
            "description": "Create a new contact named John Smith with email john@example.com",
            "goal": "Add contact to address book",
        },
        {
            "task_id": "task_002", 
            "description": "Send an email to alice@example.com with subject 'Meeting Tomorrow'",
            "goal": "Send email successfully",
        },
        {
            "task_id": "task_003",
            "description": "Create a calendar event for tomorrow at 2pm titled 'Team Standup'",
            "goal": "Create calendar event",
        },
        {
            "task_id": "task_004",
            "description": "Add a todo item: 'Buy groceries' with high priority",
            "goal": "Create todo item",
        },
        {
            "task_id": "task_005",
            "description": "Search for restaurants nearby and save the top result",
            "goal": "Find and save restaurant",
        },
    ]
    
    examples = []
    for i in range(num_samples):
        task = tasks[i % len(tasks)]
        examples.append({
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that can interact with apps through API calls. "
                        "Use the appworld tool to execute Python code that calls app APIs. "
                        "Complete the task by making the appropriate API calls."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Task: {task['description']}",
                },
            ],
            "tools": ["appworld"],
            "env_info": {
                "env_config": {
                    "task_id": task["task_id"],
                },
            },
            "ground_truth": task["goal"],
            "dataset": "env",  # verifier source - "env" means use EnvVerifier
        })
    
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser(description="Create and push environment datasets")
    parser.add_argument("--push", action="store_true", help="Push datasets to HuggingFace Hub")
    parser.add_argument("--namespace", default="hamishivi", help="HuggingFace namespace")
    args = parser.parse_args()
    
    print("Creating Wordle dataset...")
    wordle_ds = create_wordle_dataset(100)
    print(f"  Created {len(wordle_ds)} examples")
    
    print("Creating Wiki-Search dataset...")
    wiki_ds = create_wiki_search_dataset(100)
    print(f"  Created {len(wiki_ds)} examples")
    
    print("Creating AppWorld dataset...")
    appworld_ds = create_appworld_dataset(50)
    print(f"  Created {len(appworld_ds)} examples")
    
    if args.push:
        print(f"\nPushing to HuggingFace Hub ({args.namespace})...")
        
        wordle_name = f"{args.namespace}/wordle_env_train"
        print(f"  Pushing {wordle_name}...")
        wordle_ds.push_to_hub(wordle_name, private=False)
        
        wiki_name = f"{args.namespace}/wiki_search_env_train"
        print(f"  Pushing {wiki_name}...")
        wiki_ds.push_to_hub(wiki_name, private=False)
        
        appworld_name = f"{args.namespace}/appworld_env_train"
        print(f"  Pushing {appworld_name}...")
        appworld_ds.push_to_hub(appworld_name, private=False)
        
        print("\nDone! Datasets pushed successfully.")
    else:
        print("\nDatasets created locally. Use --push to upload to HuggingFace Hub.")
        print("\nSample Wordle example:")
        print(wordle_ds[0])


if __name__ == "__main__":
    main()
