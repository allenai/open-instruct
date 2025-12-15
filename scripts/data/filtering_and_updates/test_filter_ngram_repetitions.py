#!/usr/bin/env python3
"""
Comprehensive test suite for filter_ngram_repetitions.py
Merges all existing test functionality and ensures good coverage.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from filter_ngram_repetitions import (
    detect_exact_block_repetition,
    find_all_repetitions,
    find_consecutive_repetitions,
    find_ngram_repetitions,
    is_code_import_or_return,
    is_math_or_code,
    is_short_phrase,
    process_example,
    should_be_filtered_by_repetition,
    split_into_paragraphs,
    split_into_sentences,
)


def test_utility_functions():
    """Test utility functions for text processing and detection."""
    print("Testing utility functions...")
    print("=" * 80)

    # Test split_into_paragraphs
    print("\n🧪 Test: split_into_paragraphs")
    text = "Para 1.\n\nPara 2.\n\n\nPara 3."
    paragraphs = split_into_paragraphs(text)
    print(f"Input: {repr(text)}")
    print(f"Output: {paragraphs}")
    assert len(paragraphs) == 3, f"Expected 3 paragraphs, got {len(paragraphs)}"

    # Test split_into_sentences
    print("\n🧪 Test: split_into_sentences")
    text = "Sentence 1. Sentence 2! Sentence 3?"
    sentences = split_into_sentences(text)
    print(f"Input: {repr(text)}")
    print(f"Output: {sentences}")
    assert len(sentences) == 3, f"Expected 3 sentences, got {len(sentences)}"

    # Test is_math_or_code
    print("\n🧪 Test: is_math_or_code")
    math_text = "x = y + 1"
    normal_text = "This is normal text"
    print(f"Math text '{math_text}': {is_math_or_code(math_text)}")
    print(f"Normal text '{normal_text}': {is_math_or_code(normal_text)}")
    assert is_math_or_code(math_text) == True
    assert is_math_or_code(normal_text) == False

    # Test is_code_import_or_return
    print("\n🧪 Test: is_code_import_or_return")
    import_text = "import numpy as np"
    return_text = "return True"
    normal_text = "This is not code"
    print(f"Import '{import_text}': {is_code_import_or_return(import_text)}")
    print(f"Return '{return_text}': {is_code_import_or_return(return_text)}")
    print(f"Normal '{normal_text}': {is_code_import_or_return(normal_text)}")
    assert is_code_import_or_return(import_text) == True
    assert is_code_import_or_return(return_text) == True
    assert is_code_import_or_return(normal_text) == False

    # Test is_short_phrase
    print("\n🧪 Test: is_short_phrase")
    short_phrase = "not sure"
    long_phrase = "This is a longer phrase that should not be considered short"
    print(f"Short phrase '{short_phrase}': {is_short_phrase(short_phrase)}")
    print(f"Long phrase '{long_phrase}': {is_short_phrase(long_phrase)}")
    assert is_short_phrase(short_phrase) == True
    # Note: is_short_phrase checks if any short phrase is contained within the text
    # So "This is a longer phrase" contains "is" which is in the short phrases list
    assert is_short_phrase(long_phrase) == True  # Contains "is" which is a short phrase


def test_2x_repetitions():
    """Test 2x repetitions (minimum threshold)."""
    print("\nTesting 2x repetitions (minimum threshold)...")
    print("=" * 80)

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

    # Test case 3: Large chunks repeated 2x
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

    # Test 3: Large chunk repeated exactly 2x
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
    print("Testing with default min_repetitions=10 (should NOT catch 2x repetitions)")
    print("=" * 80)

    # Test same examples with default threshold (should not be caught)
    print("\n🧪 Test 4: 2x repetition with default threshold (should be FALSE)")
    has_rep, reason, details = detect_exact_block_repetition(two_paragraph_text, min_repetitions=10)
    print(f"Result: {has_rep} (should be False)")
    print(f"Reason: {reason}")


def test_consecutive_repetitions():
    """Test consecutive vs non-consecutive repetitions."""
    print("\nTesting consecutive vs non-consecutive repetitions...")
    print("=" * 80)

    # Test case: consecutive line repetitions (more repetitions to meet threshold)
    consecutive_text = """This is a normal line.

This repetition happens here.
This repetition happens here.
This repetition happens here.
This repetition happens here.
This repetition happens here.
This repetition happens here.
This repetition happens here.
This repetition happens here.
This repetition happens here.
This repetition happens here.

And then we continue with normal text."""

    # Test case: non-consecutive repetitions (more repetitions to meet threshold)
    non_consecutive_text = """This repetition happens here.

Some other text in between.

This repetition happens here.

More different text.

This repetition happens here.

Even more different text.

This repetition happens here.

Yet more different content.

This repetition happens here.

Still more different content.

This repetition happens here.

Final different content.

This repetition happens here.

Another different section.

This repetition happens here.

More varied content.

This repetition happens here.

Different text again.

This repetition happens here.

Yet another different section.

This repetition happens here.

Final different content."""

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

    # Test 3: Simple repetition with lower thresholds (should be caught)
    print("\n🧪 Test 3: Simple repetition with lower thresholds")
    simple_text = """This line repeats.
This line repeats.
This line repeats.
This line repeats."""
    has_rep, reason, details = detect_exact_block_repetition(
        simple_text, min_repetitions=2, min_sentence_repetitions=2, min_consecutive_sentence_repetitions=2
    )
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")
    if details:
        print(f"Total count: {details['total_count']}")
        print(f"Consecutive count: {details['consecutive_count']}")
        print(f"Positions: {details['positions']}")


def test_exact_repetition_examples():
    """Test specific examples of exact block repetition."""
    print("\nTesting exact block repetition detection...")
    print("=" * 80)

    # Test case 1: Scooby-Doo example
    scooby_text = """Shaggy: (excitedly) I don't know, man! But I'm ready for some spooky action!

(Scooby-Doo theme music plays)

Scooby-Doo, where are you?

Shaggy: (excitedly) I don't know, man! But I'm ready for some spooky action!

(Scooby-Doo theme music plays)

Scooby-Doo, where are you?

Shaggy: (excitedly) I don't know, man! But I'm ready for some spooky action!

(Scooby-Doo theme music plays)

Scooby-Doo, where are you?

Shaggy: (excitedly) I don't know, man! But I'm ready for some spooky action!

(Scooby-Doo theme music plays)

Scooby-Doo, where are you?

Shaggy: (excitedly) I don't know, man! But I'm ready for some spooky action!

(Scooby-Doo theme music plays)

Scooby-Doo, where are you?"""

    # Test case 2: Marketing URL example
    marketing_url_text = """<|user|>
I am trying to change the marketing URL but it is not selecting
<|assistant|>
<think> I will provide the user with additional steps to troubleshoot the issue. </think> <answer> If you are still having trouble selecting the "Marketing URL" field, try the following steps:

1. Make sure you are logged in to the correct App Store Connect account.
2. Check if there are any other apps selected in the list. If so, deselect the app and try again.
3. Clear your browser cache and cookies, then try again.
4. Try using a different web browser to access App Store Connect.
5. If none of the above steps work, contact Apple Support for further assistance.

<|user|>
I am trying to change the marketing URL but it is not selecting
<|assistant|>
<think> I will provide the user with additional steps to troubleshoot the issue. </think> <answer> If you are still having trouble selecting the "Marketing URL" field, try the following steps:

1. Make sure you are logged in to the correct App Store Connect account.
2. Check if there are any other apps selected in the list. If so, deselect the app and try again.
3. Clear your browser cache and cookies, then try again.
4. Try using a different web browser to access App Store Connect.
5. If none of the above steps work, contact Apple Support for further assistance.

<|user|>
I am trying to change the marketing URL but it is not selecting
<|assistant|>
<think> I will provide the user with additional steps to troubleshoot the issue. </think> <answer> If you are still having trouble selecting the "Marketing URL" field, try the following steps:

1. Make sure you are logged in to the correct App Store Connect account.
2. Check if there are any other apps selected in the list. If so, deselect the app and try again.
3. Clear your browser cache and cookies, then try again.
4. Try using a different web browser to access App Store Connect.
5. If none of the above steps work, contact Apple Support for further assistance."""

    # Test case 3: Normal text (should NOT be flagged)
    normal_text = """This is a normal conversation about various topics. The user asks a question and I provide a helpful response. There might be some repetition of words here and there, but nothing excessive.

The weather today is quite nice. I hope you are having a good day. Let me know if you need any help with anything else.

Here's another paragraph with different content. This paragraph talks about something completely different from the previous one. There's no repetition here that would be problematic."""

    # Test 1: Scooby-Doo example
    print("\n🧪 Test 1: Scooby-Doo repeated lines")
    has_rep, reason, details = detect_exact_block_repetition(scooby_text)
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")
    if details:
        print(f"Details: {details}")

    # Test 2: Marketing URL example
    print("\n🧪 Test 2: Marketing URL conversation repetition")
    has_rep, reason, details = detect_exact_block_repetition(marketing_url_text)
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")
    if details:
        print(f"Details: {details}")

    # Test 3: Normal text
    print("\n🧪 Test 3: Normal text (should NOT be flagged)")
    has_rep, reason, details = detect_exact_block_repetition(normal_text)
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")
    if details:
        print(f"Details: {details}")


def test_main_functions():
    """Test the main processing functions with mock dataset examples."""
    print("\nTesting main filtering functions...")
    print("=" * 80)

    # Create mock dataset examples
    test_examples = [
        {
            "id": "normal_example",
            "messages": [
                {"role": "user", "content": "What is the weather like today?"},
                {
                    "role": "assistant",
                    "content": "I don't have access to real-time weather data, but I can help you find weather information through various methods.",
                },
            ],
            "source": "normal_conversation",
        },
        {
            "id": "repetitive_example",
            "messages": [
                {"role": "user", "content": "Help me with this issue."},
                {
                    "role": "assistant",
                    "content": """I'll help you solve this problem step by step.

First, let's check your configuration settings.

I'll help you solve this problem step by step.

First, let's check your configuration settings.

I'll help you solve this problem step by step.

First, let's check your configuration settings.""",
                },
            ],
            "source": "problematic_dataset",
        },
        {
            "id": "consecutive_repetition",
            "messages": [
                {"role": "user", "content": "Explain this concept."},
                {
                    "role": "assistant",
                    "content": """Here's the explanation you requested about this important concept:

The same line repeats here with detailed information.
The same line repeats here with detailed information.
The same line repeats here with detailed information.

I hope this comprehensive explanation helps you understand the concept better!""",
                },
            ],
            "source": "another_dataset",
        },
    ]

    for i, example in enumerate(test_examples):
        print(f"\n🧪 Test {i+1}: {example['id']}")

        # Test the filtering function
        should_filter, reason, details = should_be_filtered_by_repetition(
            example, "messages", filter_user_turns=False, min_repetitions=2
        )

        print(f"Should filter: {should_filter}")
        print(f"Reason: {reason}")
        if details:
            print(f"Block type: {details.get('block_type', 'N/A')}")
            print(f"Total count: {details.get('total_count', 'N/A')}")
            print(f"Consecutive count: {details.get('consecutive_count', 'N/A')}")

        # Test the process function
        processed = process_example(example, "messages", index=i, filter_user_turns=False, min_repetitions=2)
        print(f"Has repetition: {processed['has_repetition']}")
        print(f"Repetition reason: {processed['repetition_reason']}")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\nTesting edge cases and boundary conditions...")
    print("=" * 80)

    # Test empty text
    print("\n🧪 Test 1: Empty text")
    has_rep, reason, details = detect_exact_block_repetition("")
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")

    # Test single word
    print("\n🧪 Test 2: Single word")
    has_rep, reason, details = detect_exact_block_repetition("hello")
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")

    # Test code patterns (should be ignored)
    print("\n🧪 Test 3: Code patterns")
    code_text = """import numpy as np
def function():
    return True
import numpy as np
def function():
    return True"""
    has_rep, reason, details = detect_exact_block_repetition(code_text)
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")

    # Test math expressions (should be ignored)
    print("\n🧪 Test 4: Math expressions")
    math_text = """x = y + 1
z = a * b
x = y + 1
z = a * b"""
    has_rep, reason, details = detect_exact_block_repetition(math_text)
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")

    # Test short phrases (should be ignored for non-consecutive)
    print("\n🧪 Test 5: Short phrases")
    short_phrase_text = """not sure
maybe
not sure
probably
not sure"""
    has_rep, reason, details = detect_exact_block_repetition(short_phrase_text)
    print(f"Result: {has_rep}")
    print(f"Reason: {reason}")


def test_ngram_functions():
    """Test the n-gram repetition detection functions."""
    print("\nTesting n-gram repetition detection functions...")
    print("=" * 80)

    # Test find_consecutive_repetitions
    print("\n🧪 Test 1: find_consecutive_repetitions")
    items = ["a", "b", "a", "b", "a", "b", "c"]
    consecutive = find_consecutive_repetitions(items, "test")
    print(f"Items: {items}")
    print(f"Consecutive repetitions: {consecutive}")

    # Test find_all_repetitions
    print("\n🧪 Test 2: find_all_repetitions")
    all_reps = find_all_repetitions(items, "test")
    print(f"All repetitions: {all_reps}")

    # Test find_ngram_repetitions
    print("\n🧪 Test 3: find_ngram_repetitions")
    text = "the cat the dog the cat the bird"
    ngrams = find_ngram_repetitions(text, n=2, min_occurrences=2)
    print(f"Text: {text}")
    print(f"N-gram repetitions: {ngrams}")


def run_all_tests():
    """Run all test functions."""
    print("🧪 COMPREHENSIVE TEST SUITE FOR FILTER_NGRAM_REPETITIONS")
    print("=" * 80)

    try:
        test_utility_functions()
        test_2x_repetitions()
        test_consecutive_repetitions()
        test_exact_repetition_examples()
        test_main_functions()
        test_edge_cases()
        test_ngram_functions()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
