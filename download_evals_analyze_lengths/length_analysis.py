"""
Analyze token lengths of model responses in JSONL files using dolma2-tokenizer.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from transformers import AutoTokenizer
import argparse


def load_jsonl_file(filepath: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of JSON objects."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in {filepath}: {e}")
    return data


def extract_model_responses(data: List[Dict[str, Any]]) -> List[str]:
    """Extract model responses from the data."""
    responses = []
    
    for entry in data:
        # Extract model_output field which contains list of response dicts
        if 'model_output' in entry:
            for output in entry['model_output']:
                if 'continuation' in output:
                    response = output['continuation']
                    # print(len(output['original_output']))
                    if 'reasoning_content' in output and output['reasoning_content'] is not None and len(output['reasoning_content']) > 0:
                        response = '<think>\n' + output['reasoning_content'] + '</think>\n' + response
                    responses.append(response)
    
    return responses


def tokenize_responses(responses: List[str], tokenizer) -> List[int]:
    """Tokenize responses and return list of token counts."""
    token_counts = []
    
    for response in responses:
        # Tokenize and count tokens
        tokens = tokenizer.encode(response)
        token_counts.append(len(tokens))
    
    return token_counts


def calculate_statistics(token_counts: List[int]) -> Dict[str, float]:
    """Calculate comprehensive statistics for token counts."""
    if not token_counts:
        return {}
    
    counts = np.array(token_counts)
    
    stats = {
        'count': len(counts),
        'mean': np.mean(counts),
        'median': np.median(counts),
        'std': np.std(counts),
        'min': np.min(counts),
        'max': np.max(counts),
        'p25': np.percentile(counts, 25),
        'p75': np.percentile(counts, 75),
        'p90': np.percentile(counts, 90),
        'p95': np.percentile(counts, 95),
        'p99': np.percentile(counts, 99),
    }
    
    return stats


def print_statistics(filename: str, stats: Dict[str, float]):
    """Print statistics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Statistics for {filename}")
    print(f"{'='*60}")
    
    if not stats:
        print("No model responses found in this file.")
        return
    
    print(f"Total responses: {stats['count']:.0f}")
    print(f"\nToken Length Statistics:")
    print(f"  Mean:     {stats['mean']:.1f} tokens")
    print(f"  Median:   {stats['median']:.1f} tokens")
    print(f"  Std Dev:  {stats['std']:.1f} tokens")
    print(f"  Min:      {stats['min']:.0f} tokens")
    print(f"  Max:      {stats['max']:.0f} tokens")
    print(f"\nPercentiles:")
    print(f"  25th:     {stats['p25']:.0f} tokens")
    print(f"  75th:     {stats['p75']:.0f} tokens")
    print(f"  90th:     {stats['p90']:.0f} tokens")
    print(f"  95th:     {stats['p95']:.0f} tokens")
    print(f"  99th:     {stats['p99']:.0f} tokens")


def main():
    parser = argparse.ArgumentParser(description='Analyze token lengths in JSONL files')
    parser.add_argument('directory', nargs='?', default='.',
                       help='Directory containing JSONL files (default: current directory)')
    parser.add_argument('--pattern', default='*.jsonl',
                       help='File pattern to match (default: *.jsonl)')
    args = parser.parse_args()
    
    # Initialize tokenizer
    print("Loading dolma2 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Make sure you have installed: pip install transformers")
        sys.exit(1)
    
    # Find all JSONL files in directory
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
    
    jsonl_files = sorted(directory.glob(args.pattern))
    
    if not jsonl_files:
        print(f"No files matching '{args.pattern}' found in {directory}")
        sys.exit(1)
    
    print(f"Found {len(jsonl_files)} JSONL file(s)")
    
    # Store all statistics for summary
    all_stats = {}
    all_token_counts = []
    
    # Process each file
    for filepath in jsonl_files:
        print(f"\nProcessing: {filepath.name}")
        
        # Load data
        data = load_jsonl_file(filepath)
        print(f"  Loaded {len(data)} entries")
        
        # Extract model responses
        responses = extract_model_responses(data)
        print(f"  Found {len(responses)} model responses")
        
        if responses:
            # Tokenize responses
            token_counts = tokenize_responses(responses, tokenizer)
            
            # Calculate statistics
            stats = calculate_statistics(token_counts)
            
            # Store for summary
            all_stats[filepath.name] = stats
            all_token_counts.extend(token_counts)
            
            # Print statistics for this file
            print_statistics(filepath.name, stats)
    
    # Print overall summary if multiple files
    if len(jsonl_files) > 1 and all_token_counts:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY ACROSS ALL FILES")
        print(f"{'='*60}")
        overall_stats = calculate_statistics(all_token_counts)
        print(f"Total responses across all files: {overall_stats['count']:.0f}")
        print(f"\nOverall Token Length Statistics:")
        print(f"  Mean:     {overall_stats['mean']:.1f} tokens")
        print(f"  Median:   {overall_stats['median']:.1f} tokens")
        print(f"  Std Dev:  {overall_stats['std']:.1f} tokens")
        print(f"  Min:      {overall_stats['min']:.0f} tokens")
        print(f"  Max:      {overall_stats['max']:.0f} tokens")
        
        # Comparison table
        print(f"\n{'='*60}")
        print("FILE COMPARISON")
        print(f"{'='*60}")
        print(f"{'File':<30} {'Mean':<10} {'Median':<10} {'Std Dev':<10}")
        print(f"{'-'*60}")
        for filename, stats in all_stats.items():
            truncated_name = filename[:27] + "..." if len(filename) > 30 else filename
            print(f"{truncated_name:<30} {stats['mean']:<10.1f} {stats['median']:<10.1f} {stats['std']:<10.1f}")


if __name__ == "__main__":
    main()