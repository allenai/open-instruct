#!/usr/bin/env python3
"""
Test script to verify expensive eval detection functionality.
"""

import sys
import os

# Add the current directory to the path so we can import from mason.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the functions we want to test
from mason import check_for_expensive_evals, EXPENSIVE_EVAL_PATTERNS

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
            "expected": ["alpaca_farm", "gpt-4"],
            "description": "AlpacaFarm with GPT-4"
        },
        {
            "command": ["python", "eval/alpaca_farm/run_eval.py", "--openai_engine", "text-davinci-003"],
            "expected": ["alpaca_farm", "text-davinci"],
            "description": "AlpacaFarm with Davinci"
        },
        {
            "command": ["python", "eval/alpaca_farm/run_eval.py", "--openai_engine", "gpt-3.5-turbo"],
            "expected": ["alpaca_farm", "gpt-3.5"],
            "description": "AlpacaFarm with GPT-3.5"
        },
        {
            "command": ["python", "eval/alpaca_farm/run_eval.py", "--openai_engine", "claude-3-sonnet"],
            "expected": ["alpaca_farm", "claude"],
            "description": "AlpacaFarm with Claude"
        },
        {
            "command": ["python", "eval/alpaca_farm/run_eval.py", "--openai_engine", "azure-gpt-4"],
            "expected": ["alpaca_farm", "azure"],
            "description": "AlpacaFarm with Azure"
        },
        {
            "command": ["python", "eval/alpaca_farm/run_eval.py", "--openai_engine", "anthropic-claude"],
            "expected": ["alpaca_farm", "anthropic", "claude"],
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
    sys.exit(0 if success else 1)