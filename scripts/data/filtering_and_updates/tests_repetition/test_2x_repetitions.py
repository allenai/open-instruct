#!/usr/bin/env python3
"""
Test script for 2x repetitions (minimum threshold)
"""

import sys
import os
sys.path.append('/weka/oe-adapt-default/nathanl/open-instruct/scripts/data/filtering_and_updates')

from filter_ngram_repetitions import detect_exact_block_repetition

# Test case 1: Exactly 2x paragraph repetition
two_paragraph_text = """This is the first paragraph that will be repeated.

Some other content in between.

This is the first paragraph that will be repeated.

More different content at the end."""

# Test case 2: Exactly 2x consecutive line repetition
two_line_text = """This is a normal line.

This line will be repeated twice.
This line will be repeated twice.

And then we continue with normal text."""

# Test case 3: Large chunks repeated 2x (like your examples)
large_chunk_2x = """<|user|>
I have a complex question about machine learning algorithms and their applications in natural language processing. Can you help me understand the differences?
<|assistant|>
<think>This is a comprehensive question that requires me to explain several concepts clearly and systematically.</think>

<answer>I'd be happy to help you understand the differences between various machine learning algorithms used in NLP! Here are the key distinctions:

**Supervised Learning:**
- Uses labeled training data
- Examples: Classification (sentiment analysis), Named Entity Recognition
- Algorithms: SVM, Random Forest, Neural Networks

**Unsupervised Learning:**
- No labeled data required
- Examples: Topic modeling, clustering, word embeddings
- Algorithms: K-means, LDA, Word2Vec

**Deep Learning Approaches:**
- Multi-layered neural networks
- Examples: BERT, GPT, Transformers
- Excellent for complex pattern recognition

Would you like me to dive deeper into any specific area?
</answer>

<|user|>
I have a complex question about machine learning algorithms and their applications in natural language processing. Can you help me understand the differences?
<|assistant|>
<think>This is a comprehensive question that requires me to explain several concepts clearly and systematically.</think>

<answer>I'd be happy to help you understand the differences between various machine learning algorithms used in NLP! Here are the key distinctions:

**Supervised Learning:**
- Uses labeled training data
- Examples: Classification (sentiment analysis), Named Entity Recognition
- Algorithms: SVM, Random Forest, Neural Networks

**Unsupervised Learning:**
- No labeled data required
- Examples: Topic modeling, clustering, word embeddings
- Algorithms: K-means, LDA, Word2Vec

**Deep Learning Approaches:**
- Multi-layered neural networks
- Examples: BERT, GPT, Transformers
- Excellent for complex pattern recognition

Would you like me to dive deeper into any specific area?
</answer>"""

def test_2x_repetitions():
    print("Testing 2x repetitions (minimum threshold)...")
    print("=" * 80)
    
    # Test 1: 2x paragraph repetition (should be caught with min_repetitions=2)
    print("\n🧪 Test 1: Exactly 2x paragraph repetition")
    has_rep, reason, details = detect_exact_block_repetition(two_paragraph_text, min_repetitions=2)
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")
    if details:
        print(f"Total count: {details['total_count']}")
        print(f"Consecutive count: {details['consecutive_count']}")
        print(f"Block type: {details['block_type']}")
        print(f"Positions: {details['positions']}")
        print(f"Repeated block: '{details['repeated_block'][:100]}...'")
    
    # Test 2: 2x consecutive line repetition
    print("\n🧪 Test 2: Exactly 2x consecutive line repetition")
    has_rep, reason, details = detect_exact_block_repetition(two_line_text, min_repetitions=2)
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")
    if details:
        print(f"Total count: {details['total_count']}")
        print(f"Consecutive count: {details['consecutive_count']}")
        print(f"Block type: {details['block_type']}")
        print(f"Positions: {details['positions']}")
    
    # Test 3: Large chunk repeated exactly 2x (like real examples)
    print("\n🧪 Test 3: Large conversation chunk repeated 2x")
    has_rep, reason, details = detect_exact_block_repetition(large_chunk_2x, min_repetitions=2)
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")
    if details:
        print(f"Total count: {details['total_count']}")
        print(f"Consecutive count: {details['consecutive_count']}")
        print(f"Block type: {details['block_type']}")
        print(f"Positions: {details['positions']}")
        print(f"Repeated block: '{details['repeated_block'][:200]}...'")
    
    print("\n" + "=" * 80)
    print("Testing with default min_repetitions=3 (should NOT catch 2x repetitions)")
    print("=" * 80)
    
    # Test same examples with default threshold (should not be caught)
    print("\n🧪 Test 4: 2x repetition with default threshold (should be FALSE)")
    has_rep, reason, details = detect_exact_block_repetition(two_paragraph_text, min_repetitions=3)
    print(f"Result: {has_rep} (should be False)")
    print(f"Reason: {reason}")

if __name__ == "__main__":
    test_2x_repetitions()
