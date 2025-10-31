#!/usr/bin/env python3
"""
Extract median values from statistics.json files and format for Google Sheets.
"""
import json
import os
import csv
import sys
from pathlib import Path
from collections import defaultdict
import re
import argparse

# Add the download_evals_analyze_lengths directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'download_evals_analyze_lengths'))

try:
    from length_analysis import (
        load_jsonl_file,
        extract_model_responses,
        tokenize_responses,
        calculate_statistics
    )
    HAS_LENGTH_ANALYSIS = True
except ImportError:
    HAS_LENGTH_ANALYSIS = False

def extract_step_number(text):
    """Extract step number from text containing _step_XXX pattern"""
    match = re.search(r'_step_(\d+)', text)
    if match:
        return match.group(1)
    return None

def extract_model_name(folder_name):
    """Extract model name from folder like results_nvidia-Llama-3.1-Nemotron-Nano-8B-v1-deepseek-configs"""
    # Remove 'results_' prefix and '-deepseek-configs' suffix
    name = folder_name.replace('results_', '')
    name = name.replace('-deepseek-configs', '')
    return name

def extract_eval_name(experiment_key):
    """Extract eval name from experiment key like 'lmeval-MODEL-on-aime-HASH (ID)'"""
    # Pattern: lmeval-*-on-EVALNAME-hash
    # Use non-greedy match to capture eval names that may contain dashes
    match = re.search(r'-on-(.+?)-[a-f0-9]+ \(', experiment_key)
    if match:
        return match.group(1)
    return None

def extract_eval_from_dirname(dirname):
    """Extract eval name from subdirectory name like 'lmeval-MODEL-on-EVAL-HASH'"""
    match = re.search(r'-on-(.+?)-[a-f0-9]+-[A-Z0-9]+$', dirname)
    if match:
        return match.group(1)
    return None

def analyze_subdirectories(model_dir, tokenizer=None):
    """Analyze subdirectories to compute per-experiment statistics."""
    if not HAS_LENGTH_ANALYSIS:
        print(f"Warning: Cannot analyze subdirectories for {model_dir.name} - length_analysis module not available")
        return {}
    
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
            print(f"  Loading tokenizer for {model_dir.name}...")
            tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
        except Exception as e:
            print(f"  Warning: Failed to load tokenizer: {e}")
            return {}
    
    per_experiment = {}
    
    # Find all experiment subdirectories
    subdirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('lmeval-')]
    
    if not subdirs:
        return {}
    
    print(f"  Found {len(subdirs)} experiment subdirectories, analyzing...")
    
    for subdir in subdirs:
        eval_name = extract_eval_from_dirname(subdir.name)
        if not eval_name:
            continue
        
        # Find prediction files
        pred_files = list(subdir.glob('*predictions.jsonl'))
        if not pred_files:
            continue
        
        # Collect token counts for this experiment
        token_counts = []
        for pred_file in pred_files:
            try:
                data = load_jsonl_file(pred_file)
                responses = extract_model_responses(data)
                if responses:
                    counts = tokenize_responses(responses, tokenizer)
                    token_counts.extend(counts)
            except Exception as e:
                print(f"    Warning: Failed to process {pred_file.name}: {e}")
                continue
        
        if token_counts:
            stats = calculate_statistics(token_counts)
            # Convert numpy types to Python types for JSON serialization
            stats = {k: (v.item() if hasattr(v, 'item') else v) for k, v in stats.items()}
            exp_key = f"{subdir.name.rsplit('-', 1)[0]} ({subdir.name.rsplit('-', 1)[1]})"
            per_experiment[exp_key] = stats
            print(f"    {eval_name}: median={stats.get('median', 0):.1f}")
    
    return per_experiment

def main():
    parser = argparse.ArgumentParser(
        description='Extract median values from statistics.json files and export for Google Sheets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        default='/weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/length_analyses/new_faeze',
        help='Directory containing subdirectories with statistics.json files'
    )
    parser.add_argument(
        '--output',
        default='medians_export.tsv',
        help='Output file path (TSV format for Google Sheets import)'
    )
    
    args = parser.parse_args()
    base_path = Path(args.input_dir)
    
    # Dictionary to store data: model -> eval -> list of medians
    data = defaultdict(lambda: defaultdict(list))
    
    # Load tokenizer once if needed (for models without per_experiment statistics)
    tokenizer = None
    
    # Find all statistics.json files
    for stats_file in base_path.glob('*/statistics.json'):
        folder_name = stats_file.parent.name
        model_name = extract_model_name(folder_name)
        
        # Extract step number from folder name
        folder_step_num = extract_step_number(folder_name)
        
        # Read the JSON file
        with open(stats_file, 'r') as f:
            content = json.load(f)
        
        # Process each experiment
        per_experiment = content.get('per_experiment', {})
        
        if not per_experiment:
            # No per_experiment data, try to analyze subdirectories
            print(f"\nModel {folder_name} has no per_experiment data, analyzing subdirectories...")
            if tokenizer is None and HAS_LENGTH_ANALYSIS:
                try:
                    from transformers import AutoTokenizer
                    print("Loading tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
                except Exception as e:
                    print(f"Warning: Failed to load tokenizer: {e}")
            
            per_experiment = analyze_subdirectories(stats_file.parent, tokenizer)
            if not per_experiment:
                print(f"  No per-experiment data could be extracted for {folder_name}")
                continue
        
        for exp_key, exp_data in per_experiment.items():
            # Check for step number in experiment key, fallback to folder step
            step_num = extract_step_number(exp_key)
            if step_num is None:
                step_num = folder_step_num
            
            # Create full model identifier with step if present
            if step_num:
                full_model_name = f"{model_name} (step_{step_num})"
            else:
                full_model_name = model_name
            
            eval_name = extract_eval_name(exp_key)
            if eval_name:
                median = exp_data.get('median')
                if median is not None:
                    # Collect all median values for this model/eval combination
                    data[full_model_name][eval_name].append(median)
    
    # Get all unique eval names and sort them
    all_evals = sorted(set(eval_name for model_data in data.values() for eval_name in model_data.keys()))
    
    # Get all model names and sort them by step number
    def sort_by_step(model_name):
        """Sort key function: extracts step number for sorting, or uses -1 if no step"""
        # Look for step number in format (step_XXX)
        match = re.search(r'step_(\d+)', model_name)
        if match:
            base_name = model_name.split(' (step_')[0]
            step_num = int(match.group(1))
            return (base_name, step_num)
        else:
            return (model_name, -1)
    
    all_models = sorted(data.keys(), key=sort_by_step)
    
    # Write to TSV file
    output_path = Path(args.output)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Write header
        writer.writerow(['Model'] + all_evals)
        
        # Write data rows
        for model in all_models:
            row = [model]
            for eval_name in all_evals:
                median_list = data[model].get(eval_name, [])
                if median_list:
                    # Average all medians for this eval
                    avg_median = sum(median_list) / len(median_list)
                    # Round to 1 decimal place
                    row.append(f"{avg_median:.1f}")
                else:
                    row.append('')
            writer.writerow(row)
    
    print(f"Exported data to {output_path}")
    print(f"Found {len(all_models)} models and {len(all_evals)} evaluations")
    print(f"\nTo import into Google Sheets:")
    print(f"  1. Open Google Sheets")
    print(f"  2. Go to File > Import")
    print(f"  3. Upload {output_path}")
    print(f"  4. Choose 'Tab' as the separator")

if __name__ == '__main__':
    main()

