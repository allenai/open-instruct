#!/usr/bin/env python3
"""
Test consecutive repetitions specifically 
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filter_ngram_repetitions import detect_exact_block_repetition

# Test case: consecutive line repetitions
consecutive_text = """This is a normal line.

This repetition happens here.
This repetition happens here.
This repetition happens here.
This repetition happens here.

And then we continue with normal text."""

# Test case: non-consecutive repetitions
non_consecutive_text = """This repetition happens here.

Some other text in between.

This repetition happens here.

More different text.

This repetition happens here."""

def test_consecutive():
    print("Testing consecutive vs non-consecutive repetitions...")
    print("=" * 80)
    
    # Test 1: Consecutive repetitions
    print("\n🧪 Test 1: Consecutive line repetitions")
    has_rep, reason, details = detect_exact_block_repetition(consecutive_text)
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")
    if details:
        print(f"Total count: {details['total_count']}")
        print(f"Consecutive count: {details['consecutive_count']}")
        print(f"Positions: {details['positions']}")
    
    # Test 2: Non-consecutive repetitions (should still be caught)
    print("\n🧪 Test 2: Non-consecutive repetitions")  
    has_rep, reason, details = detect_exact_block_repetition(non_consecutive_text)
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")
    if details:
        print(f"Total count: {details['total_count']}")
        print(f"Consecutive count: {details['consecutive_count']}")
        print(f"Positions: {details['positions']}")

if __name__ == "__main__":
    test_consecutive()
