#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import re
from collections import defaultdict

from datasets import Dataset, load_dataset

import open_instruct.utils as open_instruct_utils

"""
Script to remove examples with repetitive reasoning/text patterns in post-training datasets.
Focuses on sentence-level repetition patterns that indicate "unhinged" behavior, especially
useful for reasoning traces from models like R1.

Run with:
python scripts/data/filtering_and_updates/filter_ngram_repetitions.py --input-dataset allenai/tulu-3-sft-mixture --column messages
"""


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, removing empty ones."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using basic punctuatisi."""
    # Simple sentence splitting - can be improved with nltk/spacy
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def is_math_or_code(text: str) -> bool:
    """Detect if text contains mathematical expressions or code patterns."""
    # Common math/code symbols and patterns
    math_symbols = [
        "=",
        "+",
        "-",
        "*",
        "/",
        "^",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        "|",
        "&",
        "~",
        "!",
        "@",
        "#",
        "$",
        "%",
        "\\",
    ]

    # Check if text contains math symbols
    has_math_symbols = any(symbol in text for symbol in math_symbols)

    # Check for common code patterns
    code_patterns = [
        r"\w+\([^)]*\)",  # function calls like Array(n)
        r"\w+\s*=\s*\w+",  # assignments like s = 1/3600
        r"\d+/\d+",  # fractions like 1/3600
        r"[a-zA-Z_]\w*\s*[+\-*/]\s*[a-zA-Z_]\w*",  # variable operations
        r"[a-zA-Z_]\w*\s*[=<>!]+\s*[a-zA-Z_]\w*",  # comparisons
    ]

    has_code_patterns = any(re.search(pattern, text) for pattern in code_patterns)

    return has_math_symbols or has_code_patterns


def is_code_import_or_return(text: str) -> bool:
    """Detect if text contains import statements, return statements, or other code patterns."""
    # Common code patterns that should be ignored
    code_patterns = [
        r"^import\s+",  # import statements
        r"^from\s+\w+\s+import",  # from ... import statements
        r"^return\s+",  # return statements
        r"^return\s+\w+",  # return with value like "return None", "return True"
        r"^\w+\s*\+\s*=\s*\d+",  # increment operations like "next_node += 1"
        r"^\w+\s*-\s*=\s*\d+",  # decrement operations
        r"^\w+\s*\*\s*=\s*\d+",  # multiplication assignment
        r"^\w+\s*/\s*=\s*\d+",  # division assignment
        r"^for\s+",  # for loops
        r"^while\s+",  # while loops
        r"^if\s+",  # if statements
        r"^def\s+",  # function definitions
        r"^class\s+",  # class definitions
        r"^\w+\.\w+\(",  # method calls like "Image.open()"
        r"^\w+\s*=\s*\w+\(",  # function assignments
    ]

    # Check for single-sided brackets/parentheses (unmatched)
    single_bracket_patterns = [
        r"[\(\[\{][^\)\]\}]*$",  # Starts with opening bracket but no closing
        r"^[^\(\[\{]*[\)\]\}]",  # Ends with closing bracket but no opening
        r"[\(\[\{][^\(\[\{\)\]\}]*[\)\]\}]",  # Has both opening and closing brackets
    ]

    # If it has single-sided brackets, it's likely a code fragment
    has_single_brackets = any(re.search(pattern, text) for pattern in single_bracket_patterns[:2])
    has_matched_brackets = any(re.search(pattern, text) for pattern in single_bracket_patterns[2:])

    # If it has unmatched brackets, consider it code
    if has_single_brackets and not has_matched_brackets:
        return True

    return any(re.search(pattern, text, re.IGNORECASE) for pattern in code_patterns)


def is_short_phrase(text: str) -> bool:
    """Detect if text is a short phrase that shouldn't count for overall repetition."""
    # Short phrases that are common and shouldn't trigger repetition detection
    short_phrases = [
        "not sure",
        "wait no",
        "i think",
        "maybe",
        "probably",
        "yes",
        "no",
        "ok",
        "okay",
        "right",
        "wrong",
        "correct",
        "incorrect",
        "true",
        "false",
        "good",
        "bad",
        "nice",
        "great",
        "awesome",
        "terrible",
        "wow",
        "oh",
        "hmm",
        "um",
        "uh",
        "well",
        "so",
        "now",
        "then",
        "here",
        "there",
        "this",
        "that",
        "these",
        "those",
        "it",
        "is",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "can",
        "cannot",
        "cant",
        "dont",
        "doesnt",
        "isnt",
        "arent",
        "second < b",
        "we can use:",
        "we can use",
        "first < a",
        "third < c",
        "fourth < d",
        "a < b",
        "b < c",
        "c < d",
        "d < e",
        "e < f",
        "f < g",
        "g < h",
        "x < y",
        "y < z",
        "i < j",
        "j < k",
        "k < l",
        "l < m",
        "m < n",
        "n < o",
        "o < p",
        "p < q",
        "q < r",
        "r < s",
        "s < t",
        "t < u",
        "u < v",
        "v < w",
        "w < x",
        "x < y",
        "y < z",
        "return none",
        "return true",
        "return false",
        "return 0",
        "return 1",
        "return -1",
        "return null",
        "return undefined",
    ]

    # Check if the text (case insensitive) matches any short phrase
    text_lower = text.lower().strip()
    if any(phrase in text_lower for phrase in short_phrases):
        return True

    # Also check for simple comparison patterns (word < word or word > word)
    # But only if it's not a code pattern
    import re

    comparison_pattern = r"^\w+\s*[<>]\s*\w+$"
    if re.match(comparison_pattern, text.strip()):
        # Make sure it's not a code pattern like "import math"
        code_patterns = [
            "import",
            "from",
            "return",
            "def",
            "class",
            "if",
            "for",
            "while",
            "return none",
            "return true",
            "return false",
        ]
        if not any(code_pattern in text_lower for code_pattern in code_patterns):
            return True

    return False


def is_complex_math_expression(text: str) -> bool:
    """Detect complex mathematical expressions that should be ignored."""
    # Patterns for complex math expressions
    math_patterns = [
        r"math\.\w+\([^)]*\)",  # math functions like math.log()
        r"\w+\*\*[0-9]+",  # power operations
        r"[0-9]+\s*[+\-*/]\s*[0-9]+",  # arithmetic operations
        r"[a-zA-Z_]\w*\s*[+\-*/]\s*[a-zA-Z_]\w*",  # variable operations
        r"def\s+\w+\([^)]*\):",  # function definitions
        r"lambda\s+[^:]+:",  # lambda functions
        r"\[[^\]]*for[^\]]*\]",  # list comprehensions
        r"\{[^}]*for[^}]*\}",  # set comprehensions
        r"\([^)]*for[^)]*\)",  # generator expressions
    ]

    return any(re.search(pattern, text) for pattern in math_patterns)


def is_structured_list(text: str) -> bool:
    """Detect if text contains structured list patterns."""
    # Check for numbered lists (1., 2., 3., etc.)
    numbered_list_pattern = r"^\d+\.\s"

    # Check for bullet points (-, *, •, etc.)
    bullet_pattern = r"^[\-\*•]\s"

    # Check for structured content with consistent formatting
    lines = text.split("\n")
    if len(lines) < 2:
        return False

    # Check if multiple lines start with the same pattern
    numbered_count = sum(1 for line in lines if re.match(numbered_list_pattern, line.strip()))
    bullet_count = sum(1 for line in lines if re.match(bullet_pattern, line.strip()))

    # If more than 2 lines have the same structure, consider it a structured list
    return numbered_count >= 2 or bullet_count >= 2


def is_multi_line_paragraph(text: str) -> bool:
    """Detect if text is a multi-line paragraph (contains newlines)."""
    return "\n" in text.strip()


def is_common_transition_word(text: str) -> bool:
    """Detect if text is a common transition word that frequently appears and should be ignored."""
    # Common transition words that are frequently repeated and cause false positives
    transition_words = [
        "alternatively,",
        "therefore,",
        "however,",
        # 'furthermore,', 'moreover,',
        # 'consequently,', 'nevertheless,', 'nonetheless,', 'hence,', 'thus,',
        # 'meanwhile,', 'subsequently,', 'additionally,', 'likewise,', 'similarly,',
        # 'conversely,', 'on the other hand,', 'in contrast,', 'in conclusion,',
        # 'finally,', 'firstly,', 'secondly,', 'thirdly,', 'lastly,', 'initially,'
    ]

    # Check if the text (case insensitive) exactly matches a transition word
    text_lower = text.lower().strip()
    return text_lower in transition_words


def find_consecutive_repetitions(items: list[str], block_type: str) -> dict[str, tuple[int, list[int]]]:
    """
    Find consecutive repetitions in a list of items (sentences or paragraphs).
    Returns dict mapping repeated items to (total_count, consecutive_positions).
    """
    repetition_info = {}

    # Find consecutive repetitions
    i = 0
    while i < len(items):
        current_item = items[i].strip()
        if len(current_item) < (10 if block_type == "line" else 20):
            i += 1
            continue

        # Count consecutive occurrences
        consecutive_count = 1
        j = i + 1
        while j < len(items) and items[j].strip() == current_item:
            consecutive_count += 1
            j += 1

        # If we found consecutive repetitions
        if consecutive_count > 1:
            # Count total occurrences in the entire text
            total_count = sum(1 for item in items if item.strip() == current_item)

            key = f"consecutive_{block_type}_repeated_{consecutive_count}x"
            if consecutive_count >= total_count:
                # All repetitions are consecutive
                key = f"total_{block_type}_repeated_{total_count}x"

            if current_item not in repetition_info:
                repetition_info[current_item] = {
                    "type": key,
                    "total_count": total_count,
                    "consecutive_count": consecutive_count,
                    "positions": list(range(i, j)),
                }

        i = j if consecutive_count > 1 else i + 1

    return repetition_info


def find_all_repetitions(items: list[str], block_type: str) -> dict[str, dict]:
    """
    Find all repetitions (both consecutive and non-consecutive) in a list of items.
    Returns dict mapping repeated items to repetition info.
    """
    repetition_info = {}

    # Count occurrences of each item
    item_counts = {}
    item_positions = {}

    for i, item in enumerate(items):
        clean_item = item.strip()
        if len(clean_item) < (5 if block_type == "line" else 10):  # Minimum 5 chars for sentences, 10 for paragraphs
            continue

        if clean_item not in item_counts:
            item_counts[clean_item] = 0
            item_positions[clean_item] = []

        item_counts[clean_item] += 1
        item_positions[clean_item].append(i)

    # Find consecutive runs for each repeated item
    for item, count in item_counts.items():
        if count > 1:  # Only consider items that appear more than once
            positions = item_positions[item]

            # Find the longest consecutive run
            max_consecutive = 1
            current_consecutive = 1

            for i in range(1, len(positions)):
                if positions[i] == positions[i - 1] + 1:
                    current_consecutive += 1
                else:
                    max_consecutive = max(max_consecutive, current_consecutive)
                    current_consecutive = 1
            max_consecutive = max(max_consecutive, current_consecutive)

            repetition_info[item] = {
                "type": f"total_{block_type}_repeated_{count}x",
                "total_count": count,
                "consecutive_count": max_consecutive,
                "positions": positions,
            }

    return repetition_info


def find_ngram_repetitions(text: str, n: int = 3, min_occurrences: int = 2) -> dict[str, list[int]]:
    """
    Find n-gram repetitions in text.
    Returns dict mapping n-grams to their positions.
    """
    words = text.lower().split()
    ngrams = {}

    # Generate all n-grams and their positions
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        if ngram not in ngrams:
            ngrams[ngram] = []
        ngrams[ngram].append(i)

    # Filter to only repeated n-grams
    repeated = {ngram: positions for ngram, positions in ngrams.items() if len(positions) >= min_occurrences}

    return repeated


def detect_exact_block_repetition(
    text: str,
    min_repetitions: int = 10,
    min_sentence_repetitions: int = 40,
    min_consecutive_sentence_repetitions: int = 4,
):
    """
    Detect exact block repetitions in text (paragraphs or lines).

    This function analyzes text for repetitive patterns at both paragraph and sentence levels,
    using different thresholds based on content type and repetition pattern.

    Args:
        text (str): The input text to analyze for repetitive patterns.
        min_repetitions (int, optional): Minimum number of repetitions for paragraphs.
            Defaults to 2.
        min_sentence_repetitions (int, optional): Minimum number of repetitions for
            non-consecutive sentences. Defaults to 20.
        min_consecutive_sentence_repetitions (int, optional): Minimum number of repetitions
            for consecutive sentences. Defaults to 2.

    Returns:
        tuple: A tuple containing:
            - has_repetition (bool): True if repetitive patterns are detected, False otherwise
            - reason (str or None): Description of the repetition type if detected, None otherwise
            - details (dict or None): Detailed information about the repetition including:
                - block_type (str): Type of repeated block ('paragraph' or 'line')
                - total_count (int): Total number of repetitions
                - consecutive_count (int): Number of consecutive repetitions
                - positions (list): List of positions where repetitions occur
                - repeated_block (str): The actual repeated text block

    Thresholds Used:
        - Paragraphs: min_repetitions+ repetitions (default: 10+)
        - Multi-line paragraphs: 4x higher threshold (40+ repetitions)
        - Math/code paragraphs: 2x higher threshold (20+ repetitions)
        - Consecutive sentences: min_consecutive_sentence_repetitions+ repetitions (if ~10+ characters)
        - Non-consecutive sentences: min_sentence_repetitions+ repetitions
        - Math/code content: 2x higher thresholds
        - Structured lists: 3x higher thresholds
        - Short phrases: Ignored for non-consecutive repetitions
        - Short paragraphs: 20+ characters for non-consecutive repetitions
        - Code patterns: Completely ignored for both paragraphs and sentences
    """
    # Check if this is structured content (lists, etc.)
    is_structured = is_structured_list(text)

    # Check paragraph-level repetitions first (threshold: 2+)
    paragraphs = split_into_paragraphs(text)
    paragraph_repetitions = find_all_repetitions(paragraphs, "paragraph")

    # Check for any paragraph repeated min_repetitions or more times
    for paragraph, info in paragraph_repetitions.items():
        # Skip short paragraphs (less than 20 characters) for non-consecutive repetitions
        if info["consecutive_count"] == 1 and len(paragraph.strip()) < 20:
            continue

        # Skip code patterns entirely for paragraphs
        if is_code_import_or_return(paragraph.strip()):
            continue

        # Skip complex math expressions for paragraphs
        if is_complex_math_expression(paragraph.strip()):
            continue

        # Skip common transition words that are false positives
        if is_common_transition_word(paragraph.strip()):
            continue

        # Check if this is a multi-line paragraph
        is_multi_line = is_multi_line_paragraph(paragraph)

        # Check if this paragraph contains math/code content
        is_math_code = is_math_or_code(paragraph)

        # Use higher threshold for structured content, multi-line paragraphs, and math/code content
        structured_multiplier = 3 if is_structured else 1
        multi_line_multiplier = 4 if is_multi_line else 1  # Much higher threshold for multi-line
        math_code_multiplier = 2 if is_math_code else 1  # Higher threshold for math/code content
        effective_min_repetitions = min_repetitions * max(
            structured_multiplier, multi_line_multiplier, math_code_multiplier
        )

        # For non-consecutive paragraph repetitions, use higher threshold
        if info["consecutive_count"] == 1 and info["total_count"] > 1:
            # Non-consecutive repetitions need higher threshold, but cap it reasonably
            effective_min_repetitions = min(effective_min_repetitions * 2, 30)  # Increased cap for multi-line and code

        if info["total_count"] >= effective_min_repetitions:
            details = {
                "block_type": "paragraph",
                "total_count": info["total_count"],
                "consecutive_count": info["consecutive_count"],
                "positions": info["positions"],
                "repeated_block": paragraph,
            }
            return True, f"paragraph_repeated_{info['total_count']}x", details

    # Check line-level repetitions with different thresholds for consecutive vs non-consecutive
    sentences = split_into_sentences(text)
    sentence_repetitions = find_all_repetitions(sentences, "line")

    # Check for any sentence repeated
    for sentence, info in sentence_repetitions.items():
        # Skip single words entirely
        sentence_words = sentence.strip().split()
        if len(sentence_words) <= 1:
            continue

        # Skip short phrases for overall repetition (but not consecutive)
        if is_short_phrase(sentence.strip()):
            # For short phrases, only flag if they're consecutive
            if info["consecutive_count"] >= min_consecutive_sentence_repetitions * 2:
                details = {
                    "block_type": "line",
                    "total_count": info["total_count"],
                    "consecutive_count": info["consecutive_count"],
                    "positions": info["positions"],
                    "repeated_block": sentence,
                }
                return True, f"line_repeated_{info['total_count']}x_consecutive_short_phrase", details
            continue

        # Skip code patterns entirely
        if is_code_import_or_return(sentence.strip()):
            continue

        # Skip complex math expressions
        if is_complex_math_expression(sentence.strip()):
            continue

        # Skip common transition words that are false positives
        if is_common_transition_word(sentence.strip()):
            continue

        # Check if this sentence contains math/code content
        is_math_code = is_math_or_code(sentence)

        # Adjust thresholds for math/code content (2x higher) and structured content (3x higher)
        math_multiplier = 2 if is_math_code else 1
        structured_multiplier = 3 if is_structured else 1
        effective_min_consecutive = min_consecutive_sentence_repetitions * max(math_multiplier, structured_multiplier)
        effective_min_sentence = min_sentence_repetitions * max(math_multiplier, structured_multiplier)

        # For consecutive repetitions, use lower threshold (2+) if sentence is long enough
        if info["consecutive_count"] >= effective_min_consecutive and len(sentence.strip()) >= 5:
            details = {
                "block_type": "line",
                "total_count": info["total_count"],
                "consecutive_count": info["consecutive_count"],
                "positions": info["positions"],
                "repeated_block": sentence,
            }
            return True, f"line_repeated_{info['total_count']}x_consecutive", details

        # For non-consecutive repetitions, use higher threshold (5+) and require longer sentences
        elif info["total_count"] >= effective_min_sentence and len(sentence.strip()) >= 10:
            details = {
                "block_type": "line",
                "total_count": info["total_count"],
                "consecutive_count": info["consecutive_count"],
                "positions": info["positions"],
                "repeated_block": sentence,
            }
            return True, f"line_repeated_{info['total_count']}x", details

    return False, None, None


def detect_repetitive_patterns(example: dict, column: str = "message", sentence_level: bool = True) -> dict:
    """Detect various types of repetitive patterns in text, focusing on consecutive repetitions."""
    text = extract_assistant_content(example, column)

    # Use the new detection function with updated thresholds
    has_repetition, reason, details = detect_exact_block_repetition(
        text, min_repetitions=25, min_sentence_repetitions=40, min_consecutive_sentence_repetitions=20
    )

    # Always create repetition_examples field with consistent structure
    # This avoids type inference issues in multiprocessing
    repetition_examples = []
    if has_repetition and details and "repeated_block" in details:
        repetition_examples = [str(details["repeated_block"])]

    # Create result with consistent types
    result = example.copy()
    result["has_repetition"] = bool(has_repetition)  # Ensure boolean type
    result["repetition_reason"] = str(reason) if reason else ""  # Ensure string type
    result["repetition_examples"] = list(repetition_examples)  # Always list of strings, ensure it's a list

    return result


def collect_repetitive_examples(example: dict) -> dict:
    """Collect examples that have repetitions for analysis."""
    if example.get("has_repetition", False):
        return example
    return None


def filter_repetitive_examples(example: dict) -> bool:
    """Filter out examples with repetitions."""
    return not example.get("has_repetition", False)


def extract_assistant_content(example: dict, column: str) -> str:
    """Extract assistant content from example based on column format."""
    if column == "messages":
        messages = example.get("messages", [])
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        if assistant_messages:
            return assistant_messages[-1].get("content", "")
    elif column == "assistant":
        return example.get("assistant", "")
    return ""


def process_example(
    example: dict, column: str, index: int = 0, filter_user_turns: bool = False, min_repetitions: int = 2
) -> dict:
    """
    Process a single example to detect repetitions and add metadata.
    """
    # Extract assistant content
    assistant_content = extract_assistant_content(example, column)

    # Check for repetitions with updated thresholds
    has_repetition, reason, details = detect_exact_block_repetition(
        assistant_content, min_repetitions=10, min_sentence_repetitions=40, min_consecutive_sentence_repetitions=4
    )

    # Add repetition metadata with consistent types
    result = example.copy()
    result["has_repetition"] = bool(has_repetition)  # Ensure boolean type
    result["repetition_reason"] = str(reason) if reason else ""  # Ensure string type

    # Always create repetition_examples field with consistent structure
    # This avoids type inference issues in multiprocessing
    repetition_examples = []
    if has_repetition and details and "repeated_block" in details:
        repetition_examples = [str(details["repeated_block"])]
    result["repetition_examples"] = list(repetition_examples)  # Always list of strings, ensure it's a list

    return result


def should_be_filtered_by_repetition(
    example: dict, column: str, filter_user_turns: bool = False, min_repetitions: int = 2
):
    """
    Determine if an example should be filtered due to repetitions.
    Returns (should_filter, reason, details).
    """
    # Extract assistant content
    assistant_content = extract_assistant_content(example, column)

    # Check for repetitions with updated thresholds
    has_repetition, reason, details = detect_exact_block_repetition(
        assistant_content, min_repetitions=10, min_sentence_repetitions=40, min_consecutive_sentence_repetitions=4
    )

    return has_repetition, reason, details


def print_repetitive_examples(dataset: Dataset, column: str = "assistant", num_examples: int = 20):
    """Print examples of repetitive patterns for analysis."""
    repetitive_examples = dataset.filter(lambda x: x.get("has_repetition", False))

    print(f"\n{'='*80}")
    print(f"🔍 FIRST {min(num_examples, len(repetitive_examples))} REPETITIVE EXAMPLES")
    print(f"{'='*80}")

    for i, example in enumerate(repetitive_examples.select(range(min(num_examples, len(repetitive_examples))))):
        print(f"{'='*80}")
        print(f"🚫 FILTERED #{i+1}: {example.get('repetition_reason', 'unknown')}")
        print(f"📁 Source: {example.get('source', 'unknown')}")

        # Parse repetition reason to extract block type and count
        reason = example.get("repetition_reason", "")
        if "paragraph" in reason:
            block_type = "paragraph"
        elif "line" in reason:
            block_type = "line"
        else:
            block_type = "unknown"

        # Extract repetition count
        import re

        count_match = re.search(r"(\d+)x", reason)
        total_repetitions = int(count_match.group(1)) if count_match else 0
        consecutive_repetitions = 1 if "consecutive" not in reason else total_repetitions

        print(f"🔄 Block type: {block_type}")
        print(f"📈 Total repetitions: {total_repetitions}")
        print(f"➡️  Consecutive repetitions: {consecutive_repetitions}")

        if example.get("repetition_examples"):
            # Show the first repeated block
            repeated_block = example["repetition_examples"][0]

            # Find positions in the text
            text = extract_assistant_content(example, column)
            if block_type == "paragraph":
                items = split_into_paragraphs(text)
            else:
                items = split_into_sentences(text)

            positions = [i for i, item in enumerate(items) if item.strip() == repeated_block.strip()]

            print(f"📍 Found at positions: {positions}")
            print("🔁 Repeated block:")
            print(f"   '{repeated_block}'")
        else:
            print(f"📍 Repetition details from reason: {reason}")

        assistant_content = extract_assistant_content(example, column)
        print(f"📄 Assistant content ({len(assistant_content)} chars) [TRIGGERED FILTERING]:")
        # Show full content without truncatisi
        print(assistant_content)
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Filter out examples with n-gram repetitions")
    parser.add_argument("dataset_name", nargs="?", help="Name of the dataset to filter")
    parser.add_argument(
        "--input-dataset", type=str, help="Name of the dataset to filter (alternative to positional argument)"
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--column", type=str, default="messages", help="Column name to filter (default: messages)")
    parser.add_argument("--output-name", help="Output dataset name")
    parser.add_argument(
        "--sentence-level", action="store_true", default=True, help="Enable sentence-level repetition detection"
    )
    parser.add_argument(
        "--filter-user-turns", action="store_true", default=False, help="Also filter user turn repetitions"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--push-to-hf", action="store_true", help="Push filtered dataset to HuggingFace")
    parser.add_argument(
        "--num-proc", type=int, default=mp.cpu_count(), help="Number of processes for parallel processing"
    )
    parser.add_argument("--manual-filter", action="store_true", help="Manually review and filter flagged repetitions")

    args = parser.parse_args()

    # Handle both positional and --input-dataset argument
    dataset_name = args.dataset_name or args.input_dataset
    if not dataset_name:
        parser.error("Either dataset_name positional argument or --input-dataset must be provided")

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=args.split, num_proc=open_instruct_utils.max_num_processes())
    print(f"Dataset loaded with {len(dataset)} examples")

    if args.verbose:
        print("Filtering parameters:")
        print("  Sentence-level repetition detection enabled")
        print(f"  Filter user turns: {args.filter_user_turns}")
        print(f"  Debug mode: {args.debug}")
        print(f"  Split: {args.split}")
        print(f"  Column: {args.column}")

    # Detect repetitive patterns
    print(f"\nDetecting repetitive patterns (num_proc={args.num_proc}):")

    # Use single-threaded processing if multiprocessing causes type issues
    if args.num_proc > 1:
        try:
            # First, try to create a sample to ensure consistent types
            sample = detect_repetitive_patterns(dataset[0], column=args.column, sentence_level=args.sentence_level)

            # Define explicit features for the new columns to avoid type inference issues
            from datasets import Sequence, Value

            # Create new features for the additional columns
            new_features = {
                "has_repetition": Value("bool"),
                "repetition_reason": Value("string"),
                "repetition_examples": Sequence(Value("string")),
            }

            # Combine with existing features
            combined_features = dataset.features.copy()
            combined_features.update(new_features)

            dataset_with_flags = dataset.map(
                lambda x: detect_repetitive_patterns(x, column=args.column, sentence_level=args.sentence_level),
                num_proc=args.num_proc,
                desc="Detecting repetitive patterns",
                features=combined_features,
            )
        except Exception as e:
            print("⚠️  Multiprocessing failed due to type inference issues, falling back to single-threaded processing")
            print(f"Error: {e}")
            dataset_with_flags = dataset.map(
                lambda x: detect_repetitive_patterns(x, column=args.column, sentence_level=args.sentence_level),
                num_proc=1,
                desc="Detecting repetitive patterns (single-threaded)",
            )
    else:
        dataset_with_flags = dataset.map(
            lambda x: detect_repetitive_patterns(x, column=args.column, sentence_level=args.sentence_level),
            num_proc=args.num_proc,
            desc="Detecting repetitive patterns",
        )

    # Collect repetitive examples for analysis and store their indices in the main dataset
    repetitive_dataset = dataset_with_flags.filter(
        lambda x: x.get("has_repetition", False), desc="Collecting repetitive examples"
    )

    # Get the indices of repetitive examples by finding them in the original dataset
    repetitive_indices = []
    for i, example in enumerate(dataset_with_flags):
        if example.get("has_repetition", False):
            repetitive_indices.append(i)

    print(f"\nFound {len(repetitive_dataset)} examples with repetitive patterns")

    # Analyze repetition types
    repetition_types = defaultdict(int)
    sources = defaultdict(int)

    for example in repetitive_dataset:
        repetition_types[example.get("repetition_reason", "unknown")] += 1
        sources[example.get("source", "unknown")] += 1

    print("Repetition types breakdown:")
    for rep_type, count in sorted(repetition_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rep_type}: {count}")

    print("\nSources and counts:")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count}")

    # Print examples for manual inspectisi
    if args.debug or len(repetitive_dataset) > 0:
        print_repetitive_examples(repetitive_dataset, column=args.column)

    # Manual filtering option
    manual_keep_map = {}
    if args.manual_filter and len(repetitive_dataset) > 0:
        proceed = input("Do you want to manually filter flagged repetitions? (y/n): ").strip().lower()
        if proceed == "y":
            print("\nManual filtering enabled. For each flagged example, enter 'y' to keep or 'n' to remove.")
            for i, example in enumerate(repetitive_dataset):
                main_idx = repetitive_indices[i]
                print(f"\n{'='*80}")
                print(f"🚫 FILTERED #{i+1}: {example.get('repetition_reason', 'unknown')}")
                print(f"📁 Source: {example.get('source', 'unknown')}")
                reason = example.get("repetition_reason", "")
                if "paragraph" in reason:
                    block_type = "paragraph"
                elif "line" in reason:
                    block_type = "line"
                else:
                    block_type = "unknown"
                import re

                count_match = re.search(r"(\d+)x", reason)
                total_repetitions = int(count_match.group(1)) if count_match else 0
                consecutive_repetitions = 1 if "consecutive" not in reason else total_repetitions
                print(f"🔄 Block type: {block_type}")
                print(f"📈 Total repetitions: {total_repetitions}")
                print(f"➡️  Consecutive repetitions: {consecutive_repetitions}")
                if example.get("repetition_examples"):
                    repeated_block = example["repetition_examples"][0]
                    text = extract_assistant_content(example, args.column)
                    if block_type == "paragraph":
                        items = split_into_paragraphs(text)
                    else:
                        items = split_into_sentences(text)
                    positions = [i for i, item in enumerate(items) if item.strip() == repeated_block.strip()]
                    print(f"📍 Found at positions: {positions}")
                    print("🔁 Repeated block:")
                    print(f"   '{repeated_block}'")
                else:
                    print(f"📍 Repetition details from reason: {reason}")
                assistant_content = extract_assistant_content(example, args.column)
                print(f"📄 Assistant content ({len(assistant_content)} chars) [TRIGGERED FILTERING]:")
                print(assistant_content)
                print(f"{'='*80}")
                keep = input("Keep this example? (y/n): ").strip().lower()
                manual_keep_map[main_idx] = keep == "y"
            print(
                f"\nAfter manual filtering, {sum(manual_keep_map.values())} repetitive examples remain and will be kept."
            )
        else:
            print("Skipping manual filtering.")

    # Add manual_keep column to dataset_with_flags
    def set_manual_keep(example, idx):
        # If manual filtering was done with user input, use manual decisions
        if args.manual_filter and len(manual_keep_map) > 0:
            if example.get("has_repetition", False):
                return {"manual_keep": manual_keep_map.get(idx, False)}
            else:
                return {"manual_keep": True}
        # If manual filtering was skipped or not enabled, remove all repetitive examples
        else:
            if example.get("has_repetition", False):
                return {"manual_keep": False}
            else:
                return {"manual_keep": True}

    dataset_with_flags = dataset_with_flags.map(set_manual_keep, with_indices=True, desc="Setting manual_keep column")

    # Filter out repetitive examples
    print(f"\nRemoving repetitive examples (num_proc={args.num_proc}):")
    filtered_dataset = dataset_with_flags.filter(
        lambda x: x.get("manual_keep", True), num_proc=args.num_proc, desc="Removing repetitive examples (manual_keep)"
    )

    print(f"\nFiltered dataset size: {len(filtered_dataset)}")
    print(
        f"Removed {len(dataset) - len(filtered_dataset)} examples ({(len(dataset) - len(filtered_dataset))/len(dataset)*100:.2f}%)"
    )

    # Clean up temporary columis
    print("Removing temporary columis: ['repetition_reason', 'has_repetition', 'repetition_examples', 'manual_keep']")
    columis_to_remove = ["repetition_reason", "has_repetition", "repetition_examples", "manual_keep"]
    for col in columis_to_remove:
        if col in filtered_dataset.column_names:
            filtered_dataset = filtered_dataset.remove_columns([col])

    output_name = args.output_name or f"{dataset_name}-ngram-filtered"

    if args.push_to_hf:
        print(f"Pushing filtered dataset to HuggingFace: {output_name}")
        filtered_dataset.push_to_hub(output_name, private=True)
    else:
        print(f"Dataset ready. Use --push-to-hf to upload to: {output_name}")


if __name__ == "__main__":
    main()
