#!/usr/bin/env python3
"""
Test suite for filter_thinking_repetitions.py

Tests all six detection strategies, length scaling, and the combined detection
logic using inline test data that simulates repetitive and clean thinking traces.
"""

import os
import sys
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from filter_thinking_repetitions import (
    classify_thinking_segments,
    detect_thinking_repetition,
    extract_thinking_trace,
    length_scaled_min_repeats,
    marker_phrase_density,
    near_duplicate_segments,
    paragraph_start_repetition,
    process_example,
    topk_vocab_combined,
    topk_vocab_ngram_repetition,
    word_class,
    word_class_ngrams,
)

# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

# 1. Extreme "Wait" looping -- should be flagged
WAIT_LOOP_EXAMPLE = {
    "messages": [
        {"role": "user", "content": "What is 15 * 23?"},
        {
            "role": "assistant",
            "content": "<think>"
            + "\n\n".join(
                [
                    f"Wait, let me reconsider this. Maybe the answer is {345 + i}. "
                    f"Let me check: 15 * 23... hmm, let me try again. "
                    f"Hold on, perhaps I should use a different approach."
                    for i in range(30)
                ]
            )
            + "</think>\nThe answer is 345.",
        },
    ]
}

# 2. "Let me" self-directed loop -- should be flagged
LET_ME_LOOP_EXAMPLE = {
    "messages": [
        {"role": "user", "content": "Solve this equation: 2x + 3 = 7"},
        {
            "role": "assistant",
            "content": "<think>"
            + "\n\n".join(
                [
                    "Let me "
                    + ["try", "check", "verify"][i % 3]
                    + " "
                    + [
                        "a different approach to this problem",
                        "this calculation once more carefully",
                        "my work on this equation step by step",
                        "using substitution to solve this",
                    ][i % 4]
                    + ". "
                    + ["If 2x + 3 = 7", "So 2x = 4"][i % 2]
                    + ", then "
                    + ["x = 2", "x should be 2"][i % 2]
                    + ". "
                    + "Let me reconsider whether this is correct."
                    for i in range(40)
                ]
            )
            + "</think>\nx = 2",
        },
    ]
}

# 3. Strategy cycling -- should be flagged
STRATEGY_CYCLING_EXAMPLE = {
    "messages": [
        {"role": "user", "content": "Find the integral of sin(x)*cos(x)"},
        {
            "role": "assistant",
            "content": "<think>"
            + "\n\n".join(
                [
                    "Let me try substitution. If u = sin(x), then du = cos(x)dx. "
                    "So the integral becomes integral of u du.",
                    "Wait, maybe integration by parts would be better here. Let me reconsider the approach entirely.",
                    "Actually, let me go back to substitution. Setting u = sin(x)... but wait, I already tried this.",
                    "Hmm, let me reconsider. Perhaps I should use the double angle formula. sin(2x)/2 = sin(x)cos(x).",
                    "Wait, let me try substitution again. With u = cos(x), du = -sin(x)dx. "
                    "Let me check if this works.",
                    "No, let me think about integration by parts once more. Let me verify: u = sin(x), dv = cos(x)dx.",
                    "Actually, substitution is simpler. Let me try u = sin(x) again. "
                    "Let me reconsider this whole thing.",
                    "Hold on, maybe the double angle identity works. Let me check: sin(2x)/2 = sin(x)cos(x).",
                    "Wait, let me go back and reconsider substitution. Perhaps I made an error earlier.",
                    "Let me try yet another approach. What about using trig identities first? "
                    "Let me think about this differently.",
                ]
                * 3
            )
            + "</think>\nThe integral is sin^2(x)/2 + C.",
        },
    ]
}

# 4. Hedging overload -- should be flagged
HEDGING_EXAMPLE = {
    "messages": [
        {"role": "user", "content": "What is the capital of Atlantis?"},
        {
            "role": "assistant",
            "content": "<think>"
            + "\n\n".join(
                [
                    "Perhaps the answer involves looking at historical records. "
                    + "Maybe I should consider what Plato wrote. "
                    + "Alternatively, this could be a trick question.",
                    "Maybe I should approach this differently. "
                    + "Perhaps the user means a fictional capital. "
                    + "Alternatively, they might be testing my knowledge.",
                    "Alternatively, let me reconsider what we know about Atlantis. "
                    + "Perhaps it was described by Plato in his dialogues. "
                    + "Maybe the capital was mentioned there.",
                    "Perhaps I need to think about this more carefully. "
                    + "Maybe the answer is that there is no known capital. "
                    + "Alternatively, some sources mention Poseidonis.",
                ]
                * 9
            )
            + "</think>\nAtlantis is a mythical place described by Plato. Its legendary capital is sometimes "
            "referred to as Poseidonis.",
        },
    ]
}

# 5. Clean short thinking trace -- should NOT be flagged
CLEAN_SHORT_EXAMPLE = {
    "messages": [
        {"role": "user", "content": "What is the derivative of x^3 + 2x?"},
        {
            "role": "assistant",
            "content": "<think>I need to find the derivative of x^3 + 2x. "
            "Using the power rule, d/dx(x^3) = 3x^2 and d/dx(2x) = 2. "
            "So the derivative is 3x^2 + 2. Let me verify: if x=1, f(1)=3, f'(1)=5. "
            "If x=2, f(2)=12, f'(2)=14. The slopes increase as expected.</think>\n"
            "The derivative is 3x^2 + 2.",
        },
    ]
}

# 6. Moderate thinking with some verification -- should NOT be flagged
MODERATE_EXAMPLE = {
    "messages": [
        {"role": "user", "content": "Is 97 a prime number?"},
        {
            "role": "assistant",
            "content": "<think>Let me check if 97 is prime. I need to test divisibility by primes "
            "up to sqrt(97) which is about 9.8. So I need to check 2, 3, 5, 7.\n\n"
            "97 / 2 = 48.5 (not divisible)\n"
            "97 / 3 = 32.33... (not divisible)\n"
            "97 / 5 = 19.4 (not divisible)\n"
            "97 / 7 = 13.86... (not divisible)\n\n"
            "Since 97 is not divisible by any prime up to its square root, it is indeed prime. "
            "Let me double-check: 7 * 14 = 98, so 7 doesn't divide 97. Correct.</think>\n"
            "Yes, 97 is a prime number.",
        },
    ]
}

# 7. Code-heavy trace -- should NOT be flagged
CODE_EXAMPLE = {
    "messages": [
        {"role": "user", "content": "Write a function to check if a number is prime."},
        {
            "role": "assistant",
            "content": "<think>I'll write a function to check if a number is prime.\n\n"
            "def is_prime(n):\n    if n < 2:\n        return False\n"
            "    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n"
            "            return False\n    return True\n\n"
            "This works by checking divisibility from 2 up to the square root of n. "
            "For each potential divisor i, if n is divisible by i, it's not prime.\n\n"
            "def is_prime_optimized(n):\n    if n < 2:\n        return False\n"
            "    if n < 4:\n        return True\n"
            "    if n % 2 == 0 or n % 3 == 0:\n        return False\n"
            "    i = 5\n    while i * i <= n:\n"
            "        if n % i == 0 or n % (i + 2) == 0:\n"
            "            return False\n        i += 6\n    return True\n\n"
            "The optimized version skips even numbers and multiples of 3.</think>\n"
            "Here is the function:\n```python\ndef is_prime(n):\n    ...\n```",
        },
    ]
}

# 8. No thinking trace -- should skip gracefully
NO_THINK_EXAMPLE = {
    "messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
    ]
}

# 9. Empty assistant message
EMPTY_ASSISTANT_EXAMPLE = {"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": ""}]}

# 10. Only user message
NO_ASSISTANT_EXAMPLE = {"messages": [{"role": "user", "content": "Hello!"}]}


# 11. Near-duplicate segments -- should be flagged by Strategy 6
# Each segment paraphrases the same approach with slightly different wording
_NEAR_DUP_SEGMENT_TEMPLATE = [
    (
        "Let me try solving this using the substitution method to find the unknown variable. "
        "First I will substitute the known values into the equation and then simplify "
        "the resulting expression step by step until I arrive at the final answer. "
        "This approach should work because substitution preserves equality throughout."
    ),
    (
        "Wait, let me attempt solving this problem with the substitution technique instead. "
        "I will substitute the given values into our equation and carefully simplify "
        "the expression that results from this operation step by step to get the answer. "
        "The substitution approach maintains equality at every stage of the process."
    ),
    (
        "Actually, let me try the substitution method to solve for the unknown variable here. "
        "Substituting the known values into the original equation gives an expression "
        "that I need to simplify step by step until I reach the final answer value. "
        "Substitution is valid because it preserves the equality relationship throughout."
    ),
    (
        "Hmm, let me reconsider and solve this using substitution to find the variable. "
        "By substituting known values into the equation I can simplify the resulting "
        "expression step by step and eventually arrive at the correct final answer. "
        "This substitution approach works because equality is preserved at each step."
    ),
    (
        "Hold on, I should try the substitution method to solve for the unknown here. "
        "When I substitute the known values into the equation the expression simplifies "
        "step by step and I should be able to determine the answer from the result. "
        "Substitution keeps the equality relationship intact throughout the process."
    ),
    (
        "Let me reconsider this problem and apply substitution to find the unknown value. "
        "Taking the known values and substituting them into the equation gives me "
        "an expression to simplify step by step until I can determine the final answer. "
        "The substitution method preserves equality so this approach should be correct."
    ),
]

NEAR_DUPLICATE_EXAMPLE = {
    "messages": [
        {"role": "user", "content": "Solve for x: 3x + 5 = 20"},
        {
            "role": "assistant",
            "content": "<think>"
            + "\n\n".join(_NEAR_DUP_SEGMENT_TEMPLATE * 4)
            + "</think>\nx = 5",
        },
    ]
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_length_scaled_min_repeats():
    """Test the length scaling function for min_repeats thresholds."""
    print("Testing length_scaled_min_repeats...")
    print("=" * 80)

    # Short texts: should return base unchanged
    assert length_scaled_min_repeats(8, 1000) == 8, "1K words should keep base=8"
    assert length_scaled_min_repeats(8, 5000) == 8, "5K words (reference) should keep base=8"
    print("  Short texts: base threshold preserved")

    # Long texts: should scale up
    assert length_scaled_min_repeats(8, 10000) == 16, "10K words should give 16"
    assert length_scaled_min_repeats(8, 20000) == 32, "20K words should give 32"
    assert length_scaled_min_repeats(8, 40000) == 64, "40K words should give 64"
    print("  Long texts: threshold scales linearly")

    # Custom reference
    assert length_scaled_min_repeats(4, 2000, reference_words=1000) == 8, "2x reference should double"
    print("  Custom reference: correct scaling")

    # Edge cases
    assert length_scaled_min_repeats(8, 0) == 8, "0 words should keep base"
    assert length_scaled_min_repeats(1, 100000) == 20, "Very long text with base=1"
    print("  Edge cases: correct")

    print("  PASSED\n")


def test_extract_thinking_trace():
    """Test extraction of thinking traces from examples."""
    print("Testing extract_thinking_trace...")
    print("=" * 80)

    # Should extract thinking trace
    trace = extract_thinking_trace(CLEAN_SHORT_EXAMPLE, "messages")
    assert trace is not None, "Should extract thinking trace from clean example"
    assert "derivative" in trace, "Trace should contain reasoning about derivatives"
    print(f"  Clean example: extracted {len(trace)} chars")

    # Should return None for no-think example
    trace = extract_thinking_trace(NO_THINK_EXAMPLE, "messages")
    assert trace is None, "Should return None when no <think> tags present"
    print("  No-think example: correctly returned None")

    # Should return None for empty assistant
    trace = extract_thinking_trace(EMPTY_ASSISTANT_EXAMPLE, "messages")
    assert trace is None, "Should return None for empty content"
    print("  Empty assistant: correctly returned None")

    # Should return None for no assistant
    trace = extract_thinking_trace(NO_ASSISTANT_EXAMPLE, "messages")
    assert trace is None, "Should return None when no assistant message"
    print("  No assistant: correctly returned None")

    # Should extract from wait loop example
    trace = extract_thinking_trace(WAIT_LOOP_EXAMPLE, "messages")
    assert trace is not None, "Should extract thinking trace from wait loop example"
    assert "Wait" in trace, "Trace should contain Wait markers"
    print(f"  Wait loop example: extracted {len(trace)} chars")

    print("  PASSED\n")


def test_marker_phrase_density():
    """Test Strategy 1: Marker phrase density detection."""
    print("Testing marker_phrase_density (Strategy 1)...")
    print("=" * 80)

    # Wait loop should have high marker density
    trace = extract_thinking_trace(WAIT_LOOP_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = marker_phrase_density(trace)
    print(
        f"  Wait loop: flagged={flagged}, counts={details['category_counts']}, density={details['total_density']:.2f}"
    )
    assert flagged, "Wait loop should be flagged by marker density"

    # Let me loop should have high marker density
    trace = extract_thinking_trace(LET_ME_LOOP_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = marker_phrase_density(trace)
    print(
        f"  Let me loop: flagged={flagged}, counts={details['category_counts']}, "
        f"density={details['total_density']:.2f}"
    )
    assert flagged, "Let me loop should be flagged by marker density"

    # Clean trace should NOT be flagged
    trace = extract_thinking_trace(CLEAN_SHORT_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = marker_phrase_density(trace)
    print(f"  Clean: flagged={flagged}, counts={details['category_counts']}, density={details['total_density']:.2f}")
    assert not flagged, "Clean trace should NOT be flagged by marker density"

    # Code trace should NOT be flagged
    trace = extract_thinking_trace(CODE_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = marker_phrase_density(trace)
    print(f"  Code: flagged={flagged}, counts={details['category_counts']}, density={details['total_density']:.2f}")
    assert not flagged, "Code trace should NOT be flagged by marker density"

    print("  PASSED\n")


def test_topk_vocab_ngrams():
    """Test Strategy 2: Top-K vocabulary n-gram detection."""
    print("Testing topk_vocab_ngram_repetition (Strategy 2)...")
    print("=" * 80)

    # Wait loop: highly repetitive at low K
    trace = extract_thinking_trace(WAIT_LOOP_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = topk_vocab_ngram_repetition(trace, k=10, n=4, min_repeats=8)
    print(f"  Wait loop (K=10, n=4): flagged={flagged}")
    if details.get("repeated_ngrams"):
        for ng, count in list(details["repeated_ngrams"].items())[:3]:
            print(f"    '{ng}': {count}x")
    assert flagged, "Wait loop should be flagged by top-K vocab n-grams"

    # Combined configs
    flagged, details = topk_vocab_combined(trace)
    print(f"  Wait loop (combined): flagged={flagged}")
    assert flagged, "Wait loop should be flagged by combined top-K configs"

    # Clean trace should NOT be flagged
    trace = extract_thinking_trace(CLEAN_SHORT_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = topk_vocab_combined(trace)
    print(f"  Clean (combined): flagged={flagged}")
    assert not flagged, "Clean trace should NOT be flagged by top-K vocab n-grams"

    # Short text should return False gracefully
    flagged, details = topk_vocab_ngram_repetition("short", k=10, n=4, min_repeats=5)
    assert not flagged, "Short text should not be flagged"
    print("  Short text: correctly not flagged")

    print("  PASSED\n")


def test_paragraph_start_repetition():
    """Test Strategy 3: Paragraph start pattern detection."""
    print("Testing paragraph_start_repetition (Strategy 3)...")
    print("=" * 80)

    # Wait loop: most paragraphs start with "Wait,"
    trace = extract_thinking_trace(WAIT_LOOP_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = paragraph_start_repetition(trace)
    print(f"  Wait loop: flagged={flagged}, details={details.get('prefix_results', {}).get(2, {})}")
    assert flagged, "Wait loop should be flagged by paragraph start repetition"

    # Let me loop: most paragraphs start with "Let me"
    trace = extract_thinking_trace(LET_ME_LOOP_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = paragraph_start_repetition(trace)
    print(f"  Let me loop: flagged={flagged}, details={details.get('prefix_results', {}).get(2, {})}")
    assert flagged, "Let me loop should be flagged by paragraph start repetition"

    # Clean trace: too few paragraphs to analyze
    trace = extract_thinking_trace(CLEAN_SHORT_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = paragraph_start_repetition(trace)
    print(f"  Clean: flagged={flagged}, skipped={details.get('skipped', False)}")
    assert not flagged, "Clean short trace should NOT be flagged"

    print("  PASSED\n")


def test_word_class_ngrams():
    """Test Strategy 4: Word class n-gram detection."""
    print("Testing word_class_ngrams (Strategy 4)...")
    print("=" * 80)

    # Test word class assignments
    assert word_class("wait") == "M", "wait should be Marker"
    assert word_class("the") == "F", "the should be Function"
    assert word_class("42") == "N", "42 should be Number"
    assert word_class("integral") == "C", "integral should be Content"
    print("  Word class assignments: correct")

    # Wait loop should have repeated marker n-grams
    trace = extract_thinking_trace(WAIT_LOOP_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = word_class_ngrams(trace)
    print(f"  Wait loop: flagged={flagged}, repeats={details.get('repeated_class_ngrams', {})}")
    assert flagged, "Wait loop should be flagged by word class n-grams"

    # Clean trace should NOT be flagged
    trace = extract_thinking_trace(CLEAN_SHORT_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = word_class_ngrams(trace)
    print(f"  Clean: flagged={flagged}")
    assert not flagged, "Clean trace should NOT be flagged by word class n-grams"

    print("  PASSED\n")


def test_segment_classification():
    """Test Strategy 5: Thinking segment action classification."""
    print("Testing classify_thinking_segments (Strategy 5)...")
    print("=" * 80)

    # Strategy cycling should show loop patterns
    trace = extract_thinking_trace(STRATEGY_CYCLING_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = classify_thinking_segments(trace)
    print(f"  Strategy cycling: flagged={flagged}, details={details}")
    assert flagged, "Strategy cycling should be flagged by segment classification"

    # Clean trace: too few segments
    trace = extract_thinking_trace(CLEAN_SHORT_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = classify_thinking_segments(trace)
    print(f"  Clean: flagged={flagged}, skipped={details.get('skipped', False)}")
    assert not flagged, "Clean trace should NOT be flagged by segment classification"

    # Moderate trace should not be flagged
    trace = extract_thinking_trace(MODERATE_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = classify_thinking_segments(trace)
    print(f"  Moderate: flagged={flagged}, details={details}")
    assert not flagged, "Moderate trace should NOT be flagged by segment classification"

    print("  PASSED\n")


def test_near_duplicate_segments():
    """Test Strategy 6: Near-duplicate segment detection."""
    print("Testing near_duplicate_segments (Strategy 6)...")
    print("=" * 80)

    # Near-duplicate example should be flagged
    trace = extract_thinking_trace(NEAR_DUPLICATE_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = near_duplicate_segments(trace)
    print(f"  Near-duplicate: flagged={flagged}, details={details}")
    assert flagged, "Near-duplicate segments should be flagged"

    # Clean trace: too few qualifying segments (short text)
    trace = extract_thinking_trace(CLEAN_SHORT_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = near_duplicate_segments(trace)
    print(f"  Clean: flagged={flagged}, skipped={details.get('skipped', False)}")
    assert not flagged, "Clean trace should NOT be flagged by near-duplicate detection"

    # Code example: segments are structurally different
    trace = extract_thinking_trace(CODE_EXAMPLE, "messages")
    assert trace is not None

    flagged, details = near_duplicate_segments(trace)
    print(f"  Code: flagged={flagged}")
    assert not flagged, "Code trace should NOT be flagged by near-duplicate detection"

    # Diverse paragraphs should not flag
    diverse = "\n\n".join(
        [
            f"Let me think about topic {chr(65 + i)}. "
            + f"This involves understanding concept {i} which relates to "
            + f"theorem number {i * 7} in the textbook. The proof requires "
            + f"applying lemma {i + 100} and corollary {i + 200} sequentially."
            for i in range(10)
        ]
    )
    flagged, details = near_duplicate_segments(diverse)
    print(f"  Diverse segments: flagged={flagged}")
    assert not flagged, "Diverse segments should NOT be flagged"

    print("  PASSED\n")


def test_topk_length_scaling():
    """Test that length scaling raises thresholds and is correctly applied."""
    print("Testing top-K length scaling behavior...")
    print("=" * 80)

    import re

    # Test 1: Verify effective_min_repeats is reported correctly in details
    # Build text where a pattern repeats exactly 6 times in ~10K words of filler
    filler_words = ["the", "quick", "brown", "fox", "jumped", "over", "lazy", "dog"]
    import random

    rng = random.Random(42)

    # Create ~10K words of random filler
    filler_parts = []
    for _ in range(1250):
        rng.shuffle(filler_words)
        filler_parts.append(" ".join(filler_words))
    filler_text = " ".join(filler_parts)
    num_words = len(re.findall(r"[a-z]+", filler_text.lower()))
    print(f"  Filler text has {num_words} words")

    # A single config: k=50, n=5, base min_repeats=4
    # With ~10K words, scale = max(1.0, 10000/5000) = 2.0
    # effective = max(4, int(4*2)) = 8
    flagged, details = topk_vocab_ngram_repetition(filler_text, k=50, n=5, min_repeats=4)
    effective = details.get("effective_min_repeats", details.get("min_repeats"))
    print(f"  Effective min_repeats for {num_words} words, base=4: {effective}")
    assert effective > 4, f"Length scaling should raise min_repeats above 4 for {num_words} words"

    # Test 2: Verify word_class_ngrams also scales
    flagged, details = word_class_ngrams(filler_text)
    wc_effective = details.get("effective_min_repeats", details.get("min_repeats"))
    print(f"  word_class_ngrams effective min_repeats: {wc_effective} (base=6)")
    assert wc_effective > 6, f"word_class_ngrams should scale min_repeats above 6 for long text"

    # Test 3: Short text should NOT scale (keeps base threshold)
    short_text = "wait let me try this approach again " * 10  # ~70 words
    flagged, details = topk_vocab_ngram_repetition(short_text, k=10, n=4, min_repeats=8)
    short_effective = details.get("effective_min_repeats", details.get("min_repeats"))
    print(f"  Short text effective min_repeats: {short_effective} (base=8)")
    assert short_effective == 8, "Short text should keep base min_repeats"

    print("  PASSED\n")


def test_combined_detection():
    """Test combined detection with min_strategies threshold."""
    print("Testing detect_thinking_repetition (combined)...")
    print("=" * 80)

    # Wait loop: should trigger multiple strategies
    trace = extract_thinking_trace(WAIT_LOOP_EXAMPLE, "messages")
    assert trace is not None

    has_rep, reason, scores = detect_thinking_repetition(trace, min_strategies=2)
    strategies_triggered = reason.split(", ") if reason else []
    print(f"  Wait loop: flagged={has_rep}, strategies={strategies_triggered}")
    assert has_rep, "Wait loop should be flagged with min_strategies=2"
    assert len(strategies_triggered) >= 2, f"Should trigger 2+ strategies, got {len(strategies_triggered)}"

    # Let me loop: should trigger multiple strategies
    trace = extract_thinking_trace(LET_ME_LOOP_EXAMPLE, "messages")
    assert trace is not None

    has_rep, reason, scores = detect_thinking_repetition(trace, min_strategies=2)
    strategies_triggered = reason.split(", ") if reason else []
    print(f"  Let me loop: flagged={has_rep}, strategies={strategies_triggered}")
    assert has_rep, "Let me loop should be flagged with min_strategies=2"

    # Strategy cycling: should trigger multiple strategies
    trace = extract_thinking_trace(STRATEGY_CYCLING_EXAMPLE, "messages")
    assert trace is not None

    has_rep, reason, scores = detect_thinking_repetition(trace, min_strategies=2)
    strategies_triggered = reason.split(", ") if reason else []
    print(f"  Strategy cycling: flagged={has_rep}, strategies={strategies_triggered}")
    assert has_rep, "Strategy cycling should be flagged with min_strategies=2"

    # Hedging: should trigger multiple strategies
    trace = extract_thinking_trace(HEDGING_EXAMPLE, "messages")
    assert trace is not None

    has_rep, reason, scores = detect_thinking_repetition(trace, min_strategies=2)
    strategies_triggered = reason.split(", ") if reason else []
    print(f"  Hedging: flagged={has_rep}, strategies={strategies_triggered}")
    assert has_rep, "Hedging example should be flagged with min_strategies=2"

    # Clean trace: should NOT be flagged
    trace = extract_thinking_trace(CLEAN_SHORT_EXAMPLE, "messages")
    assert trace is not None

    has_rep, reason, scores = detect_thinking_repetition(trace, min_strategies=2)
    print(f"  Clean: flagged={has_rep}, reason='{reason}'")
    assert not has_rep, "Clean trace should NOT be flagged"

    # Moderate trace: should NOT be flagged
    trace = extract_thinking_trace(MODERATE_EXAMPLE, "messages")
    assert trace is not None

    has_rep, reason, scores = detect_thinking_repetition(trace, min_strategies=2)
    print(f"  Moderate: flagged={has_rep}, reason='{reason}'")
    assert not has_rep, "Moderate trace should NOT be flagged"

    # Code trace: should NOT be flagged
    trace = extract_thinking_trace(CODE_EXAMPLE, "messages")
    assert trace is not None

    has_rep, reason, scores = detect_thinking_repetition(trace, min_strategies=2)
    print(f"  Code: flagged={has_rep}, reason='{reason}'")
    assert not has_rep, "Code trace should NOT be flagged"

    # No-think trace: should skip
    trace = extract_thinking_trace(NO_THINK_EXAMPLE, "messages")
    assert trace is None

    # The combined function should handle None by being called with short/empty text
    has_rep, reason, scores = detect_thinking_repetition("", min_strategies=2)
    print(f"  Empty: flagged={has_rep}, skipped={scores.get('skipped')}")
    assert not has_rep, "Empty text should NOT be flagged"

    print("  PASSED\n")


def test_min_strategies_threshold():
    """Test that the min_strategies parameter controls sensitivity."""
    print("Testing min_strategies threshold behavior...")
    print("=" * 80)

    trace = extract_thinking_trace(WAIT_LOOP_EXAMPLE, "messages")
    assert trace is not None

    # With min_strategies=1 (aggressive): should flag more easily
    has_rep_1, reason_1, _ = detect_thinking_repetition(trace, min_strategies=1)
    assert trace is not None

    strategies_1 = reason_1.split(", ") if reason_1 else []
    print(f"  min_strategies=1: flagged={has_rep_1}, strategies={len(strategies_1)}")

    # With min_strategies=6 (very conservative): likely not flagged
    has_rep_6, reason_6, _ = detect_thinking_repetition(trace, min_strategies=6)
    assert trace is not None

    strategies_6 = reason_6.split(", ") if reason_6 else []
    print(f"  min_strategies=6: flagged={has_rep_6}, strategies={len(strategies_6)}")

    assert has_rep_1, "Should be flagged with min_strategies=1"
    # With 6 strategies available, the wait loop likely triggers most but maybe not all 6
    # The point is min_strategies=6 is stricter
    if len(strategies_1) < 6:
        assert not has_rep_6, "Should NOT be flagged with min_strategies=6 if fewer than 6 trigger"

    print("  PASSED\n")


def test_process_example():
    """Test the dataset processing function."""
    print("Testing process_example...")
    print("=" * 80)

    # Repetitive example
    result = process_example(WAIT_LOOP_EXAMPLE, column="messages", index=0, min_strategies=2, use_pos_tagging=False)
    print(f"  Wait loop: {result}")
    assert result["has_thinking_repetition"] is True, "Wait loop should be flagged"
    assert result["thinking_repetition_reason"] != "", "Should have a reason"
    assert result["thinking_repetition_strategies_triggered"] >= 2, "Should trigger 2+ strategies"
    # Check per-strategy columns exist and are booleans
    assert isinstance(result["hit_marker_density"], bool)
    assert isinstance(result["hit_topk_vocab"], bool)
    assert isinstance(result["hit_paragraph_starts"], bool)
    assert isinstance(result["hit_pos_or_wordclass"], bool)
    assert isinstance(result["hit_segment_classification"], bool)
    assert isinstance(result["hit_near_duplicate_segments"], bool)
    # At least the marker density and topk should hit
    assert result["hit_marker_density"], "Wait loop should hit marker_density"

    # Clean example
    result = process_example(CLEAN_SHORT_EXAMPLE, column="messages", index=1, min_strategies=2, use_pos_tagging=False)
    print(f"  Clean: {result}")
    assert result["has_thinking_repetition"] is False, "Clean example should not be flagged"
    assert result["thinking_repetition_reason"] == "", "Should have empty reason"
    assert result["thinking_repetition_strategies_triggered"] == 0, "Should trigger 0 strategies"
    assert not result["hit_marker_density"], "Clean should not hit marker_density"

    # No-think example
    result = process_example(NO_THINK_EXAMPLE, column="messages", index=2, min_strategies=2, use_pos_tagging=False)
    print(f"  No-think: {result}")
    assert result["has_thinking_repetition"] is False, "No-think should not be flagged"
    assert not result["hit_marker_density"], "No-think should not hit any strategy"

    print("  PASSED\n")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("Testing edge cases...")
    print("=" * 80)

    # Very short text
    has_rep, reason, scores = detect_thinking_repetition("short", min_strategies=2)
    assert not has_rep, "Very short text should not be flagged"
    print("  Very short text: correctly not flagged")

    # Single long paragraph with no repetition
    text = "This is a single long paragraph. " * 20
    has_rep, reason, scores = detect_thinking_repetition(text, min_strategies=2)
    assert not has_rep, "Single paragraph without markers should not be flagged"
    print("  Single long paragraph: correctly not flagged")

    # Text with many paragraphs but diverse content
    diverse_paragraphs = "\n\n".join(
        [f"Paragraph {i} discusses topic {chr(65 + i % 26)} in detail." for i in range(20)]
    )
    has_rep, reason, scores = detect_thinking_repetition(diverse_paragraphs, min_strategies=2)
    assert not has_rep, "Diverse paragraphs should not be flagged"
    print("  Diverse paragraphs: correctly not flagged")

    # None/empty message handling
    result = process_example({"messages": []}, column="messages", index=0, min_strategies=2)
    assert not result["has_thinking_repetition"], "Empty messages should not be flagged"
    print("  Empty messages: correctly not flagged")

    print("  PASSED\n")


def run_all_tests():
    """Run all test functions."""
    print("TEST SUITE FOR FILTER_THINKING_REPETITIONS")
    print("=" * 80)

    try:
        test_length_scaled_min_repeats()
        test_extract_thinking_trace()
        test_marker_phrase_density()
        test_topk_vocab_ngrams()
        test_paragraph_start_repetition()
        test_word_class_ngrams()
        test_segment_classification()
        test_near_duplicate_segments()
        test_topk_length_scaling()
        test_combined_detection()
        test_min_strategies_threshold()
        test_process_example()
        test_edge_cases()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
