#!/usr/bin/env python3
"""
Advanced script to generate answers based on evidence from test_configs_snippets.json.
For each entry, generates N responses where the i-th response discusses evidence up to the i-th item.
This version includes file output options and more configuration.
"""

import json
import os
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
import datetime
import sys

# Add the parent directory to the path to import run_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from open_instruct.search_rewards.utils.run_utils import run_litellm


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from the JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing the data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_evidence_items(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract evidence items from a data entry.
    
    Args:
        entry: A single entry from the loaded data
        
    Returns:
        List of evidence items with their criteria and evidence
    """
    evidence_items = []
    
    # Navigate to the evidence items
    metric_config = entry.get('metric_config', {})
    config = metric_config.get('config', {})
    other_properties = config.get('other_properties', [])
    
    for item in other_properties:
        evidence_items.append({
            'name': item.get('name', ''),
            'criterion': item.get('criterion', ''),
            'evidence': item.get('evidence', [])
        })
    
    return evidence_items


def generate_context_for_evidence_count(
    question: str, 
    evidence_items: List[Dict[str, Any]], 
    num_evidence_items: int,
    format_type: str = "detailed"
) -> str:
    """
    Generate a context that discusses evidence up to the specified count.
    
    Args:
        question: The original question
        evidence_items: List of all evidence items
        num_evidence_items: Number of evidence items to include (1-based)
        format_type: Format type ("detailed", "summary", "evidence_only")
        
    Returns:
        Generated answer as a string
    """
    if num_evidence_items <= 0 or num_evidence_items > len(evidence_items):
        return f"Invalid evidence count: {num_evidence_items}. Must be between 1 and {len(evidence_items)}."
    
    # Get the evidence items to include
    included_items = evidence_items[:num_evidence_items]
    
    if format_type == "evidence_only":
        # Only include evidence snippets
        answer_parts = []
        for i, item in enumerate(included_items, 1):
            for j, evidence in enumerate(item['evidence'], 1):
                answer_parts.append(f"Evidence {i}.{j}: {evidence}")
        return "\n\n".join(answer_parts)
    else:  # detailed format (default)
        # Start building the detailed answer
        answer_parts = [f"Answer the question based on {num_evidence_items} rubrics provided below:\n"]
        
        for i, item in enumerate(included_items, 1):
            answer_parts.append(f"\nThe {i}-th rubric is: {item['criterion']}\nRelevant evidence:\n")
            
            # Add the evidence snippets
            for j, evidence in enumerate(item['evidence'], 1):
                answer_parts.append(f"   Evidence {j}: {evidence}")
        
        answer_parts.append("\nYour answer should satisfy all the provided rubrics using their corresponding evidence when answering the question. Be comprehensive and detailed.")
        answer_parts.append(f"\nQuestion: {question}")
        return "\n".join(answer_parts)


def generate_answer_for_evidence_count(
    question: str, 
    evidence_items: List[Dict[str, Any]], 
    num_evidence_items: int, 
    format_type: str = "detailed"
) -> str:
    """
    Generate an answer that discusses evidence up to the specified count.
    
    Args:
        question: The original question
        evidence_items: List of all evidence items
        num_evidence_items: Number of evidence items to include (1-based)
        format_type: Format type ("detailed", "summary", "evidence_only")
        
    Returns:
        Generated answer as a string
    """
    context = generate_context_for_evidence_count(question, evidence_items, num_evidence_items, format_type)

    # Generate the answer
    answer = run_litellm(system_prompt=None, user_prompt=context, model_name="gpt-4.1")
    return answer


def generate_all_answers_for_entry(
    entry: Dict[str, Any], 
    format_type: str = "detailed"
) -> List[str]:
    """
    Generate all possible answers for a single entry.
    
    Args:
        entry: A single entry from the loaded data
        format_type: Format type for the answers
        
    Returns:
        List of answers, where the i-th answer discusses evidence up to the i-th item
    """
    question = entry.get('initial_prompt', 'Unknown question')
    evidence_items = extract_evidence_items(entry)
    
    if not evidence_items:
        return [f"No evidence items found for question: {question}"]
    
    answers = []
    for i in range(1, len(evidence_items) + 1):
        answer = generate_answer_for_evidence_count(question, evidence_items, i, format_type)
        answers.append(answer)
    
    return answers


def check_entry_exists(question: str, output_file: str) -> bool:
    """
    Check if an entry with the given question already exists in the output file.
    
    Args:
        question: The question to check for
        output_file: Path to the JSONL output file
        
    Returns:
        True if the entry exists, False otherwise
    """
    if not os.path.exists(output_file):
        return False
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get('question') == question:
                        return True
    except (json.JSONDecodeError, FileNotFoundError):
        # If there's an error reading the file, assume it doesn't exist
        return False
    
    return False


def save_answers_to_jsonl(
    entry: Dict[str, Any], 
    answers: List[str], 
    output_file: str,
    format_type: str
) -> str:
    """
    Save generated answers to a JSONL file.
    
    Args:
        entry: The data entry
        answers: List of generated answers
        output_file: Path to the JSONL file
        format_type: Format type used
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    question = entry.get('initial_prompt', 'Unknown question')
    
    # Append to JSONL file
    with open(output_file, 'a', encoding='utf-8') as f:
        data = {
            "question": question,
            "all_answers": [
                {
                    "num_rubrics_provided": i,
                    "answer": answer
                }
                for i, answer in enumerate(answers, 1)
            ]
        }
        f.write(json.dumps(data) + '\n')
    
    return output_file


def main():
    """Main function to process the data and generate answers."""
    parser = argparse.ArgumentParser(description="Generate answers based on evidence from JSON file")
    parser.add_argument("--input", "-i", 
                       default="/fsx-comem/rulin/open-instruct/open_instruct/search_rewards/data/test_configs_snippets.json",
                       help="Path to input JSON file")
    parser.add_argument("--output-jsonl", "-o", 
                       default="./generated_answers/generated_answers.jsonl",
                       help="Path to JSONL output file")
    parser.add_argument("--format", "-f", 
                       choices=["detailed", "evidence_only"],
                       default="detailed",
                       help="Format type for answers")
    parser.add_argument("--max-entries", "-m", 
                       type=int, 
                       help="Maximum number of entries to process")
    parser.add_argument("--interactive", "-t", 
                       action="store_true",
                       help="Interactive mode (ask before processing each entry)")
    parser.add_argument("--overwrite", "-w", 
                       action="store_true",
                       help="Overwrite existing entries instead of skipping them")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.input):
        print(f"Error: File not found at {args.input}")
        return
    
    try:
        # Load the data
        print("Loading data from JSON file...")
        data = load_data(args.input)
        print(f"Loaded {len(data)} entries from the JSON file.")
        
        # Limit entries if specified
        if args.max_entries:
            data = data[:args.max_entries]
            print(f"Processing first {len(data)} entries.")
        
        # Initialize JSONL file
        print(f"Will save results to JSONL file: {args.output_jsonl}")
        
        # Process each entry
        processed_count = 0
        skipped_count = 0
        
        for entry_idx, entry in enumerate(data, 1):
            print(f"\n{'='*80}")
            print(f"Processing entry {entry_idx}/{len(data)}")
            question = entry.get('initial_prompt', 'Unknown')
            print(f"Question: {question[:100]}...")
            
            # Check if entry already exists in output file
            if check_entry_exists(question, args.output_jsonl):
                if args.overwrite:
                    print(f"🔄 Overwriting existing entry {entry_idx} (--overwrite flag used)")
                else:
                    print(f"⏭️  Skipping entry {entry_idx} - already exists in output file")
                    skipped_count += 1
                    continue
            
            # Generate all answers for this entry
            answers = generate_all_answers_for_entry(entry, args.format)
            
            print(f"Generated {len(answers)} answers for this entry:")
            for i, answer in enumerate(answers, 1):
                print(f"\n--- Answer {i} (using {i} evidence items) ---")
                print(answer[:300] + "..." if len(answer) > 300 else answer)
            
            # Save to JSONL file
            save_answers_to_jsonl(entry, answers, args.output_jsonl, args.format)
            print(f"Saved answers to JSONL file: {args.output_jsonl}")
            
            processed_count += 1
            
            # Interactive mode
            if args.interactive and entry_idx < len(data):
                user_input = input(f"\nPress Enter to continue to next entry, or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break
        
        print(f"\nAll results saved to: {args.output_jsonl}")
        print(f"Processed {processed_count} entries, skipped {skipped_count} entries.")
    
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 