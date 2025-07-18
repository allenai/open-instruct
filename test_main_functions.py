#!/usr/bin/env python3
"""
Test the main script with a mock dataset
"""

import sys
import os
sys.path.append('/weka/oe-adapt-default/nathanl/open-instruct/scripts/data/filtering_and_updates')

from datasets import Dataset
from filter_ngram_repetitions import process_example, should_be_filtered_by_repetition

# Create mock dataset examples
test_examples = [
    {
        "id": "normal_example",
        "messages": [
            {"role": "user", "content": "What is the weather like today?"},
            {"role": "assistant", "content": "I don't have access to real-time weather data, but I can help you find weather information through various methods."}
        ],
        "source": "normal_conversation"
    },
    {
        "id": "repetitive_example", 
        "messages": [
            {"role": "user", "content": "Help me with this issue."},
            {"role": "assistant", "content": """I'll help you solve this problem step by step.

First, let's check your configuration settings.

I'll help you solve this problem step by step.

First, let's check your configuration settings.

I'll help you solve this problem step by step.

First, let's check your configuration settings."""}
        ],
        "source": "problematic_dataset"
    },
    {
        "id": "consecutive_repetition",
        "messages": [
            {"role": "user", "content": "Explain this concept."},
            {"role": "assistant", "content": """Here's the explanation you requested about this important concept:

The same line repeats here with detailed information.
The same line repeats here with detailed information.
The same line repeats here with detailed information.

I hope this comprehensive explanation helps you understand the concept better!"""}
        ],
        "source": "another_dataset"
    }
]

def test_main_functions():
    print("Testing main filtering functions...")
    print("=" * 80)
    
    for i, example in enumerate(test_examples):
        print(f"\n🧪 Test {i+1}: {example['id']}")
        
        # Test the filtering function
        should_filter, reason, details = should_be_filtered_by_repetition(
            example, "messages", filter_user_turns=False
        )
        
        print(f"Should filter: {should_filter}")
        print(f"Reason: {reason}")
        if details:
            print(f"Block type: {details.get('block_type', 'N/A')}")
            print(f"Total count: {details.get('total_count', 'N/A')}")
            print(f"Consecutive count: {details.get('consecutive_count', 'N/A')}")
        
        # Test the process function
        processed = process_example(example, "messages", index=i, filter_user_turns=False)
        print(f"Has repetition: {processed['has_repetition']}")
        print(f"Repetition reason: {processed['repetition_reason']}")

if __name__ == "__main__":
    test_main_functions()
