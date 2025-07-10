#!/usr/bin/env python3
"""
Script to run benchmarks in a loop with different batch sizes and response lengths.
Tracks results in a CSV file with columns: git_commit_hash, batch_size, mfu, response_length, tokens_per_second, wall_clock_time, mbu
"""

import subprocess
import csv
import os
import sys
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# Import the benchmark generator functions
from benchmark_generators import run_benchmark_programmatic


def get_git_commit_hash() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def run_benchmark(
    num_unique_prompts: int,
    num_samples_per_prompt: int,
    num_engines: int,
    response_length: int,
    model_name_or_path: str = "hamishivi/qwen2_5_openthoughts2"
) -> Optional[Dict[str, float]]:
    """
    Run the benchmark with specified parameters and return results.
    
    Returns:
        Dictionary with mfu, tokens_per_second, wall_clock_time, and mbu, or None if failed
    """
    print(f"\nRunning benchmark with:")
    print(f"  num_unique_prompts_rollout: {num_unique_prompts}")
    print(f"  num_samples_per_prompt_rollout: {num_samples_per_prompt}")
    print(f"  vllm_num_engines: {num_engines}")
    print(f"  response_length: {response_length}")
    print(f"  Effective batch size: {num_unique_prompts * num_samples_per_prompt / num_engines}")
    
    try:
        # Call the programmatic interface
        results = run_benchmark_programmatic(
            model_name_or_path=model_name_or_path,
            num_unique_prompts_rollout=num_unique_prompts,
            num_samples_per_prompt_rollout=num_samples_per_prompt,
            vllm_num_engines=num_engines,
            response_length=response_length
        )
        
        return results
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None


def write_to_csv(
    filename: str,
    git_commit: str,
    batch_size: float,
    response_length: int,
    results: Dict[str, float]
):
    """Write results to CSV file."""
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'git_commit_hash', 'batch_size', 'response_length',
            'mfu', 'mbu', 'tokens_per_second', 'wall_clock_time'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        row = {
            'timestamp': datetime.now().isoformat(),
            'git_commit_hash': git_commit,
            'batch_size': batch_size,
            'response_length': response_length,
            'mfu': results['mfu'],
            'mbu': results['mbu'],
            'tokens_per_second': results['tokens_per_second'],
            'wall_clock_time': results['wall_clock_time']
        }
        writer.writerow(row)
        print(f"Results written to {filename}")


def main():
    """Main function to run benchmark loop."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run benchmarks in a loop with different configurations")
    parser.add_argument("--model", type=str, default="hamishivi/qwen2_5_openthoughts2",
                        help="Model name or path to benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.csv",
                        help="Output CSV filename")
    args = parser.parse_args()
    
    # Configuration ranges
    # Format: (num_unique_prompts, num_samples_per_prompt, num_engines)
    batch_size_configs = [
        (8, 1, 1),    # batch_size = 8
        (16, 1, 1),   # batch_size = 16
        (32, 1, 1),   # batch_size = 32
        (64, 1, 1),   # batch_size = 64
        (128, 1, 1),  # batch_size = 128
    ]
    
    response_lengths = [1024, 2048, 4096, 8192, 16384]
    
    # Get git commit hash once
    git_commit = get_git_commit_hash()
    print(f"Git commit hash: {git_commit}")
    print(f"Model: {args.model}")
    
    # Run benchmarks
    total_runs = len(batch_size_configs) * len(response_lengths)
    current_run = 0
    
    for num_unique, num_samples, num_engines in batch_size_configs:
        batch_size = num_unique * num_samples / num_engines
        
        for response_length in response_lengths:
            current_run += 1
            print(f"\n{'='*60}")
            print(f"Run {current_run}/{total_runs}")
            print(f"{'='*60}")
            
            results = run_benchmark(num_unique, num_samples, num_engines, response_length, args.model)
            
            if results:
                write_to_csv(args.output, git_commit, batch_size, response_length, results)
            else:
                print(f"Skipping CSV write due to benchmark failure")
    
    print(f"\n{'='*60}")
    print(f"Benchmark loop completed. Results saved to {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()