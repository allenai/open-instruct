#!/usr/bin/env python3
"""
Script to generate comprehensive reference answers based on rubric-annotated data.
For each entry, generates a single comprehensive answer that satisfies ALL provided rubrics
using their corresponding evidence. This creates ideal reference answers for evaluation.
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
            'evidence': item.get('evidence', []),
            'weight': item.get('weight', 0.0)
        })
    
    return evidence_items


def generate_comprehensive_prompt(
    question: str, 
    evidence_items: List[Dict[str, Any]]
) -> str:
    """
    Generate a comprehensive prompt that includes ALL rubrics and evidence.
    
    Args:
        question: The original question
        evidence_items: List of all evidence items with criteria and evidence
        
    Returns:
        Generated comprehensive prompt as a string
    """
    if not evidence_items:
        return f"Answer the following question: {question}"
    
    # Sort evidence items by weight (most important first)
    sorted_items = sorted(evidence_items, key=lambda x: x.get('weight', 0.0), reverse=True)
    
    prompt_parts = [
        "You are an expert answering a question based on comprehensive rubrics and evidence.",
        f"Question: {question}",
        "",
        "Your answer MUST satisfy ALL of the following rubrics using their corresponding evidence:",
        ""
    ]
    
    for i, item in enumerate(sorted_items, 1):
        prompt_parts.append(f"Rubric {i}: {item['criterion']}")
        prompt_parts.append("Supporting evidence:")
        
        for j, evidence in enumerate(item['evidence'], 1):
            prompt_parts.append(f"  • {evidence}")
        
        prompt_parts.append("")
    
    prompt_parts.extend([
        "Instructions:",
        "1. Write a comprehensive, well-structured answer that addresses ALL rubrics above",
        "2. Use the provided evidence to support your points",
        "3. Ensure your answer flows naturally and covers all criteria thoroughly",
        "4. Aim for a length that allows you to address all rubrics adequately",
        "5. Make sure each rubric is clearly addressed in your response",
        "",
        "Your comprehensive answer:"
    ])
    
    return "\n".join(prompt_parts)


def generate_comprehensive_answer(
    question: str, 
    evidence_items: List[Dict[str, Any]],
    model_name: str = "gpt-4.1"
) -> str:
    """
    Generate a comprehensive answer that satisfies ALL rubrics using their evidence.
    
    Args:
        question: The original question
        evidence_items: List of all evidence items
        model_name: Model to use for generation
        
    Returns:
        Generated comprehensive answer as a string
    """
    prompt = generate_comprehensive_prompt(question, evidence_items)
    
    # Generate the answer
    answer = run_litellm(system_prompt=None, user_prompt=prompt, model_name=model_name)
    return answer


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


def save_answer_to_jsonl(
    entry: Dict[str, Any], 
    answer: str, 
    output_file: str,
    evidence_items: List[Dict[str, Any]]
) -> str:
    """
    Save generated answer to a JSONL file.
    
    Args:
        entry: The data entry
        answer: Generated comprehensive answer
        output_file: Path to the JSONL file
        evidence_items: List of evidence items used
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    question = entry.get('initial_prompt', 'Unknown question')
    case_id = entry.get('case_id', 'unknown')
    
    # Append to JSONL file
    with open(output_file, 'a', encoding='utf-8') as f:
        data = {
            "question": question,
            "case_id": case_id,
            "comprehensive_answer": answer,
            "num_rubrics_satisfied": len(evidence_items),
            "rubrics_used": [
                {
                    "name": item['name'],
                    "criterion": item['criterion'],
                    "weight": item['weight'],
                    "evidence_count": len(item['evidence'])
                }
                for item in evidence_items
            ],
            "timestamp": datetime.datetime.now().isoformat()
        }
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    return output_file


def main():
    """Main function to process the data and generate comprehensive answers."""
    parser = argparse.ArgumentParser(description="Generate comprehensive reference answers based on rubric-annotated data")
    parser.add_argument("--input", "-i", 
                       default="/fsx-comem/rulin/open-instruct/open_instruct/search_rewards/data/test_configs_snippets.json",
                       help="Path to input JSON file")
    parser.add_argument("--output-jsonl", "-o", 
                       default="./generated_reference_answers/reference_answers.jsonl",
                       help="Path to JSONL output file")
    parser.add_argument("--model", "-m", 
                       default="gpt-4.1",
                       help="Model to use for generation")
    parser.add_argument("--max-entries", "-e", 
                       type=int, 
                       help="Maximum number of entries to process")
    parser.add_argument("--interactive", "-t", 
                       action="store_true",
                       help="Interactive mode (ask before processing each entry)")
    parser.add_argument("--overwrite", "-w", 
                       action="store_true",
                       help="Overwrite existing entries instead of skipping them")
    parser.add_argument("--start-index", "-s",
                       type=int,
                       default=0,
                       help="Start processing from this index (0-based)")
    
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
        
        # Apply start index
        if args.start_index > 0:
            data = data[args.start_index:]
            print(f"Starting from index {args.start_index}, processing {len(data)} entries.")
        
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
            
            # Extract evidence items
            evidence_items = extract_evidence_items(entry)
            
            if not evidence_items:
                print(f"⚠️  No evidence items found for entry {entry_idx}, skipping...")
                skipped_count += 1
                continue
            
            print(f"Found {len(evidence_items)} rubrics to satisfy:")
            for i, item in enumerate(evidence_items, 1):
                print(f"  {i}. {item['criterion'][:80]}...")
            
            # Generate comprehensive answer
            print(f"Generating comprehensive answer using {args.model}...")
            answer = generate_comprehensive_answer(question, evidence_items, args.model)
            
            print(f"Generated answer length: {len(answer)} characters")
            print(f"Answer preview: {answer[:200]}...")
            
            # Save to JSONL file
            save_answer_to_jsonl(entry, answer, args.output_jsonl, evidence_items)
            print(f"✅ Saved comprehensive answer to JSONL file")
            
            processed_count += 1
            
            # Interactive mode
            if args.interactive and entry_idx < len(data):
                user_input = input(f"\nPress Enter to continue to next entry, or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break
        
        print(f"\n{'='*80}")
        print(f"Processing complete!")
        print(f"Results saved to: {args.output_jsonl}")
        print(f"Processed {processed_count} entries, skipped {skipped_count} entries.")
        
        if processed_count > 0:
            print(f"\nGenerated comprehensive reference answers that satisfy ALL rubrics for each question.")
            print(f"These answers can be used as ground truth for evaluation.")
    
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
