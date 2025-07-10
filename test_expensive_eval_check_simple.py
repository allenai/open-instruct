#!/usr/bin/env python3
"""
Simple test script to verify expensive eval detection functionality without dependencies.
"""

import re
from typing import List

# Copy the expensive eval patterns and detection function
EXPENSIVE_EVAL_PATTERNS = [
    r"alpaca_eval",  # AlpacaEval uses OpenAI API
    r"alpaca_farm",  # AlpacaFarm evaluation
    r"openai",       # Any OpenAI API usage
    r"anthropic",    # Any Anthropic API usage
    r"claude",       # Claude models
    r"gpt-4",        # GPT-4 models
    r"gpt-3\.5",     # GPT-3.5 models
    r"azure",        # Azure OpenAI
    r"text-davinci", # Davinci models
]

def check_for_expensive_evals(command: List[str]) -> List[str]:
    """
    Check if the command contains any expensive evals that require external API calls.
    Returns a list of detected expensive eval patterns.
    """
    command_str = " ".join(command).lower()
    detected_evals = []
    
    for pattern in EXPENSIVE_EVAL_PATTERNS:
        if re.search(pattern, command_str):
            detected_evals.append(pattern)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_evals = []
    for eval_pattern in detected_evals:
        if eval_pattern not in seen:
            seen.add(eval_pattern)
            unique_evals.append(eval_pattern)
    
    return unique_evals

def test_expensive_eval_detection():
    """Test the expensive eval detection function with various commands."""
    
    test_cases = [
        # Test cases with expensive evals
        {
            "command": ["python", "eval/alpaca_farm/run_eval.py", "--model_name_or_path", "test"],
            "expected": ["alpaca_farm"],
            "description": "AlpacaFarm eval"
        },
        {
            "command": ["python", "eval/alpaca_farm/run_eval.py", "--openai_engine", "gpt-4"],
            "expected": ["alpaca_farm", "gpt-4", "openai"],
            "description": "AlpacaFarm with GPT-4"
        },
        {
            "command": ["python", "eval/alpaca_farm/run_eval.py", "--openai_engine", "text-davinci-003"],
            "expected": ["alpaca_farm", "text-davinci", "openai"],
            "description": "AlpacaFarm with Davinci"
        },
        {
            "command": ["python", "eval/alpaca_farm/run_eval.py", "--openai_engine", "gpt-3.5-turbo"],
            "expected": ["alpaca_farm", "gpt-3.5", "openai"],
            "description": "AlpacaFarm with GPT-3.5"
        },
        {
            "command": ["python", "eval/alpaca_farm/run_eval.py", "--openai_engine", "claude-3-sonnet"],
            "expected": ["alpaca_farm", "claude", "openai"],
            "description": "AlpacaFarm with Claude"
        },
        {
            "command": ["python", "eval/alpaca_farm/run_eval.py", "--openai_engine", "azure-gpt-4"],
            "expected": ["alpaca_farm", "azure", "gpt-4", "openai"],
            "description": "AlpacaFarm with Azure"
        },
        {
            "command": ["python", "eval/alpaca_farm/run_eval.py", "--openai_engine", "anthropic-claude"],
            "expected": ["alpaca_farm", "anthropic", "claude", "openai"],
            "description": "AlpacaFarm with Anthropic Claude"
        },
        # Test cases without expensive evals
        {
            "command": ["python", "eval/mmlu/run_eval.py", "--model_name_or_path", "test"],
            "expected": [],
            "description": "MMLU eval (not expensive)"
        },
        {
            "command": ["python", "open_instruct/finetune.py", "--model_name_or_path", "test"],
            "expected": [],
            "description": "Training command (not expensive)"
        },
    ]
    
    print("Testing expensive eval detection...")
    print("=" * 50)
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        command = test_case["command"]
        expected = test_case["expected"]
        description = test_case["description"]
        
        detected = check_for_expensive_evals(command)
        
        # Sort both lists for comparison
        detected.sort()
        expected.sort()
        
        passed = detected == expected
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"Test {i}: {description}")
        print(f"  Command: {' '.join(command)}")
        print(f"  Expected: {expected}")
        print(f"  Detected: {detected}")
        print(f"  Status: {status}")
        print()
        
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("💥 Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    success = test_expensive_eval_detection()
    exit(0 if success else 1)