#!/usr/bin/env python3
"""
Script to run benchmarks in a loop with different batch sizes and response lengths.
Tracks results in a CSV file with columns: git_commit_hash, batch_size, mfu, response_length, tokens_per_second, wall_clock_time
"""

import subprocess
import csv
import re
import os
import sys
from datetime import datetime
from typing import List, Tuple, Dict, Optional


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
    response_length: int
) -> Optional[Dict[str, float]]:
    """
    Run the benchmark with specified parameters and return parsed results.
    
    Returns:
        Dictionary with mfu, tokens_per_second, and wall_clock_time, or None if failed
    """
    cmd = [
        "./run_benchmark.sh",
        "--num_unique_prompts_rollout", str(num_unique_prompts),
        "--num_samples_per_prompt_rollout", str(num_samples_per_prompt),
        "--vllm_num_engines", str(num_engines),
        "--response_length", str(response_length)
    ]
    
    print(f"\nRunning benchmark with:")
    print(f"  num_unique_prompts_rollout: {num_unique_prompts}")
    print(f"  num_samples_per_prompt_rollout: {num_samples_per_prompt}")
    print(f"  vllm_num_engines: {num_engines}")
    print(f"  response_length: {response_length}")
    print(f"  Effective batch size: {num_unique_prompts * num_samples_per_prompt / num_engines}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"Benchmark failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None
            
        return parse_benchmark_output(result.stdout)
        
    except subprocess.TimeoutExpired:
        print("Benchmark timed out after 1 hour")
        return None
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None


def parse_benchmark_output(output: str) -> Optional[Dict[str, float]]:
    """
    Parse the benchmark output to extract MFU, tokens/second, and wall clock time.
    
    Returns:
        Dictionary with mfu, tokens_per_second, and wall_clock_time
    """
    # Look for LAST N-1 BATCHES section
    last_n_minus_1_section = False
    lines = output.split('\n')
    
    mfu = None
    tokens_per_second = None
    wall_clock_time = None
    
    for i, line in enumerate(lines):
        if "LAST N-1 BATCHES" in line:
            last_n_minus_1_section = True
            continue
            
        if last_n_minus_1_section:
            # Extract MFU
            mfu_match = re.search(r"Average MFU:\s*([\d.]+)%", line)
            if mfu_match:
                mfu = float(mfu_match.group(1))
                
            # Extract tokens per second
            tokens_match = re.search(r"Average new tokens/second:\s*([\d.]+)", line)
            if tokens_match:
                tokens_per_second = float(tokens_match.group(1))
                
            # Extract wall clock time
            time_match = re.search(r"Average generation time per batch:\s*([\d.]+)s", line)
            if time_match:
                wall_clock_time = float(time_match.group(1))
    
    # If LAST N-1 BATCHES section not found, look in regular results
    if not last_n_minus_1_section:
        for line in lines:
            if "RESULTS:" in line:
                last_n_minus_1_section = True  # Use as flag to parse following lines
                continue
                
            if last_n_minus_1_section:
                # Extract MFU
                mfu_match = re.search(r"Average MFU:\s*([\d.]+)%", line)
                if mfu_match:
                    mfu = float(mfu_match.group(1))
                    
                # Extract tokens per second
                tokens_match = re.search(r"Average new tokens/second:\s*([\d.]+)", line)
                if tokens_match:
                    tokens_per_second = float(tokens_match.group(1))
                    
                # Extract wall clock time
                time_match = re.search(r"Average generation time per batch:\s*([\d.]+)s", line)
                if time_match:
                    wall_clock_time = float(time_match.group(1))
    
    if mfu is not None and tokens_per_second is not None and wall_clock_time is not None:
        return {
            "mfu": mfu,
            "tokens_per_second": tokens_per_second,
            "wall_clock_time": wall_clock_time
        }
    else:
        print("Failed to parse all required metrics from output")
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
            'mfu', 'tokens_per_second', 'wall_clock_time'
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
            'tokens_per_second': results['tokens_per_second'],
            'wall_clock_time': results['wall_clock_time']
        }
        writer.writerow(row)
        print(f"Results written to {filename}")


def main():
    """Main function to run benchmark loop."""
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
    
    # Output CSV filename
    csv_filename = "benchmark_results.csv"
    
    # Get git commit hash once
    git_commit = get_git_commit_hash()
    print(f"Git commit hash: {git_commit}")
    
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
            
            results = run_benchmark(num_unique, num_samples, num_engines, response_length)
            
            if results:
                write_to_csv(csv_filename, git_commit, batch_size, response_length, results)
            else:
                print(f"Skipping CSV write due to benchmark failure")
    
    print(f"\n{'='*60}")
    print(f"Benchmark loop completed. Results saved to {csv_filename}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()