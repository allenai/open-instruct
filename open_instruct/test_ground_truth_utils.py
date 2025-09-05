#!/usr/bin/env python3
"""
Test script for PuzzleMatcherVerifier functionality in Python
"""

import re
import string
from dataclasses import dataclass
from typing import Optional


def normalize_answer(s: str) -> str:
    """
    Normalize answer: convert to lowercase, remove punctuation, articles, and extra whitespace
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


@dataclass
class VerifierConfig:
    """Base config for verifiers"""
    
    @classmethod
    def from_args(cls, args):
        """Create config from args - simplified for testing"""
        return cls()


@dataclass
class VerificationResult:
    score: float
    cost: float = 0.0
    reasoning: Optional[str] = None


class VerifierFunction:
    """Base class for verifier functions"""
    def __init__(self, name: str, weight: float = 1.0, verifier_config: Optional[VerifierConfig] = None):
        self.name = name
        self.weight = weight
        self.verifier_config = verifier_config

    @classmethod
    def get_config_class(cls):
        return VerifierConfig


class PuzzleMatcherVerifier(VerifierFunction):
    """
    Verifier for tasks that require string matching.
    It checks if the model output matches the ground truth answer.
    """

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        super().__init__("Puzzle", verifier_config=verifier_config, weight=1.0)

    def __call__(self, tokenized_prediction, prediction: str, label: str, query=None) -> VerificationResult:
        # Remove thinking section
        prediction = prediction.split("</think>")[-1]
        # Remove answer tags
        prediction = prediction.replace("<answer>", "").replace("</answer>", "")
        # Normalize and compare
        score = float(normalize_answer(prediction) == normalize_answer(label))
        return VerificationResult(score=score)


class MockArgs:
    """Mock arguments class to simulate args for build_all_verifiers"""
    def __init__(self):
        self.remap_verifier = None


def build_all_verifiers(args):
    """
    Simplified version of build_all_verifiers for testing
    """
    verifiers = {}
    verifier_config = PuzzleMatcherVerifier.get_config_class().from_args(args)
    puzzle_verifier = PuzzleMatcherVerifier(verifier_config)
    verifiers[puzzle_verifier.name] = puzzle_verifier
    return verifiers


def quick_test():
    """
    Quick test function - just call and verify results
    """
    print("=" * 50)
    print("QUICK TEST - PuzzleMatcherVerifier")
    print("=" * 50)
    
    # Setup verifier using standard pattern
    args = MockArgs()
    reward_fn_mapping = build_all_verifiers(args)
    verifier = reward_fn_mapping['Puzzle']
    
    # Quick test cases
    print("1. Simple match test:")
    result1 = verifier([], "The answer is 42", "answer is 42")
    print(f'   Input: "The answer is 42"')
    print(f'   Expected: "answer is 42"')
    print(f'   Score: {result1.score} ✅')
    
    print("\n2. With thinking tags:")
    result2 = verifier([], "<think>Let me solve this</think>Paris", "paris")
    print(f'   Input: "<think>Let me solve this</think>Paris"')
    print(f'   Expected: "paris"')
    print(f'   Score: {result2.score} ✅')
    
    print("\n3. With answer tags:")
    result3 = verifier([], "<answer>New York City!</answer>", "new york city")
    print(f'   Input: "<answer>New York City!</answer>"')
    print(f'   Expected: "new york city"')
    print(f'   Score: {result3.score} ✅')
    
    print("\n4. Should fail test:")
    result4 = verifier([], "Wrong answer", "correct answer")
    print(f'   Input: "Wrong answer"')
    print(f'   Expected: "correct answer"')
    print(f'   Score: {result4.score} ❌ (correctly fails)')
    
    print("\n5. Complex example:")
    result5 = verifier([], 
        "<think>This is about geography</think><answer>The capital of France is Paris.</answer>", 
        "capital of france is paris")
    print(f'   Input: "<think>...</think><answer>The capital of France is Paris.</answer>"')
    print(f'   Expected: "capital of france is paris"')
    print(f'   Score: {result5.score} ✅')
    
    print("\n" + "=" * 50)
    print("Quick test complete!")
    print("=" * 50)


def test_puzzle_matcher_verifier():
    """
    Comprehensive test suite with various scenarios
    """
    # Create mock args and build verifiers
    args = MockArgs()
    reward_fn_mapping = build_all_verifiers(args)
    verifier = reward_fn_mapping['Puzzle']
    
    # Test cases list
    test_cases = [
        # (prediction, label, expected_score, description)
        ("42", "42", 1.0, "Exact match with numbers"),
        ("hello world", "hello world", 1.0, "Exact match with text"),
        ("Hello World", "hello world", 1.0, "Case insensitive match"),
        
        # Tests with thinking tags
        ("<think>Let me think about this...</think>42", "42", 1.0, "With thinking tags, answer matches"),
        ("<think>This is complex</think>hello world", "hello world", 1.0, "With thinking tags, text matches"),
        ("<think>Analysis...</think>Wrong Answer", "42", 0.0, "With thinking tags, answer doesn't match"),
        
        # Tests with answer tags
        ("<answer>42</answer>", "42", 1.0, "With answer tags, matches"),
        ("<answer>hello world</answer>", "hello world", 1.0, "With answer tags, text matches"),
        ("<answer>wrong</answer>", "42", 0.0, "With answer tags, answer doesn't match"),
        
        # Combined tags tests
        ("<think>Thinking...</think><answer>42</answer>", "42", 1.0, "With both thinking and answer tags, matches"),
        ("<think>Let me solve this step by step</think><answer>hello world</answer>", "hello world", 1.0, "Combined tags with text match"),
        
        # Punctuation and articles tests
        ("The answer is 42!", "answer is 42", 1.0, "Remove articles and punctuation"),
        ("A simple test.", "simple test", 1.0, "Remove article 'A' and punctuation"),
        ("Hello, world!", "hello world", 1.0, "Remove punctuation"),
        
        # Whitespace tests
        ("  hello   world  ", "hello world", 1.0, "Normalize whitespace"),
        ("hello\tworld\n", "hello world", 1.0, "Replace tabs and newlines"),
        
        # Non-matching tests
        ("42", "43", 0.0, "Numbers don't match"),
        ("hello", "world", 0.0, "Text doesn't match"),
        ("", "42", 0.0, "Empty string vs non-empty"),
        
        # English examples
        ("<answer>London</answer>", "london", 1.0, "Capital city name"),
        ("<think>Animal question</think>The elephant", "elephant", 1.0, "Animal with article"),
        ("<answer>Albert Einstein</answer>", "albert einstein", 1.0, "Famous scientist name"),
        ("Romeo and Juliet by Shakespeare", "romeo and juliet by shakespeare", 1.0, "Literature reference"),
        ("<answer>United States of America</answer>", "united states of america", 1.0, "Country name"),
    ]
    
    print("=" * 80)
    print("PuzzleMatcherVerifier Test Results")
    print("=" * 80)
    
    passed = 0
    total = len(test_cases)
    
    for i, (prediction, label, expected_score, description) in enumerate(test_cases, 1):
        result = verifier([], prediction, label)
        actual_score = result.score
        
        status = "✅ PASS" if actual_score == expected_score else "❌ FAIL"
        
        print(f"\nTest {i:2d}: {status}")
        print(f"Description: {description}")
        print(f"Input: '{prediction}'")
        print(f"Label: '{label}'")
        print(f"Expected: {expected_score}, Actual: {actual_score}")
        
        if actual_score == expected_score:
            passed += 1
        else:
            # Show processed strings for debugging
            processed_pred = prediction.split("</think>")[-1].replace("<answer>", "").replace("</answer>", "")
            norm_pred = normalize_answer(processed_pred)
            norm_label = normalize_answer(label)
            print(f"Debug info:")
            print(f"  Processed prediction: '{processed_pred}'")
            print(f"  Normalized prediction: '{norm_pred}'")
            print(f"  Normalized label: '{norm_label}'")
    
    print("\n" + "=" * 80)
    print(f"Test Summary: {passed}/{total} passed ({passed/total*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    # Run quick test first
    quick_test()
    
    # Then run comprehensive tests
    print("\n")
    test_puzzle_matcher_verifier()
    
    print("\n" + "=" * 80)
    print("Testing completed!")
    print("=" * 80)
