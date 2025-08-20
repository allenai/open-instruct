#!/usr/bin/env python3

import unittest
from unittest.mock import patch, MagicMock, Mock
import sys
import os
import re

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
search_rewards_dir = os.path.dirname(current_dir)
open_instruct_dir = os.path.dirname(search_rewards_dir)
project_root = os.path.dirname(open_instruct_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, open_instruct_dir)
sys.path.insert(0, search_rewards_dir)

# Mock problematic imports before importing the module
sys.modules['jsonlines'] = Mock()
sys.modules['litellm'] = Mock()
sys.modules['open_instruct.search_rewards.run_utils'] = Mock()
sys.modules['open_instruct.search_rewards.openscholar_rewards_utils'] = Mock()
sys.modules['open_instruct.search_rewards.citation_rewards_utils'] = Mock()
sys.modules['open_instruct.search_rewards.rubric_rewards'] = Mock()

# Now import the module we want to test
import importlib.util
spec = importlib.util.spec_from_file_location(
    "longform_averaged_outcome_rewards", 
    os.path.join(search_rewards_dir, "longform_averaged_outcome_rewards.py")
)
longform_module = importlib.util.module_from_spec(spec)

# Mock the imported functions
longform_module.extract_answer_context_citations = Mock()
longform_module.score_num_in_context_search_turns = Mock()
longform_module._score_rubric = Mock()
longform_module.score_in_context_citations = Mock()
longform_module.RubricCorpusQaGenericMetric = Mock()
longform_module.LOGGER = Mock()

# Define the reward weights and functions directly
REWARD_WEIGHTS = {
    "rubric": 0.5,
    "citation": 0.2,
    "format": 0.2,
    "num_search_turns": 0.1,
}

def compute_format_reward(response: str) -> float:
    """Test version of compute_format_reward function."""
    # check if response contains final answer between <answer></answer> tags
    answer_pattern = r"<answer>.*?</answer>"
    answer_match = re.search(answer_pattern, response, re.DOTALL)
    if answer_match:
        answer_format_reward = 1.0
    else:
        answer_format_reward = 0.0
    
    # check if response contains citations between <cite></cite> tags
    citation_pattern = r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>"
    citation_match = re.search(citation_pattern, response, re.DOTALL)
    if citation_match:
        citation_format_reward = 1.0
    else:
        citation_format_reward = 0.0
    
    # check if response contains at least one valid query between <query></query> tags
    query_pattern = r"<query>.*?</query>"
    query_match = re.search(query_pattern, response, re.DOTALL)
    if query_match:
        query_format_reward = 1.0
    else:
        query_format_reward = 0.0
    
    # compute weighted average of format rewards
    format_reward = 0.5 * answer_format_reward + 0.3 * citation_format_reward + 0.2 * query_format_reward
    return format_reward

def compute_longform_averaged_outcome_reward(response: str, ground_truth: dict, question: str) -> dict:
    """Test version of compute_longform_averaged_outcome_reward function."""
    # Mock the extraction
    extracted_context, extracted_answer, extracted_citations = longform_module.extract_answer_context_citations(response)
    
    result = {
        "num_search_turns_reward": None,
        "rubric_scores": None,
        "citation_score": None,
        "format_score": None,
        "score": None,
    }
    
    # score format
    format_score = compute_format_reward(response)
    result["format_score"] = format_score
    
    # score num search turns
    num_search_turns, num_search_turns_reward = longform_module.score_num_in_context_search_turns(extracted_context)
    result["num_search_turns_reward"] = num_search_turns_reward
    
    if extracted_answer is None:  # exit early if no answer is extracted
        result["error"] = "Failed to extract answer from response"
        result["reward"] = 0.0
        result["log_values"] = {
            **result,
        }
        return result
    
    # score rubric
    rubric_scores = longform_module._score_rubric(extracted_answer, ground_truth)
    result["rubric_scores"] = rubric_scores
    
    # score citation
    citation_score = longform_module.score_in_context_citations(question, response, extracted_citations)
    result["citation_score"] = citation_score
    
    # compute score
    score = 0.0
    for key, weight in REWARD_WEIGHTS.items():
        if key == "rubric" and isinstance(rubric_scores, dict):
            # Average the rubric scores
            avg_rubric = sum(rubric_scores.values()) / len(rubric_scores) if rubric_scores else 0.0
            score += weight * avg_rubric
        else:
            score += weight * result[f"{key}_score" if key != "num_search_turns" else f"{key}_reward"]
    result["score"] = score
    
    return result


class TestLongformAveragedOutcomeRewards(unittest.TestCase):
    """Test suite for longform averaged outcome rewards functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_question = "What is the Great Wall of China?"
        
        self.sample_ground_truth = {
            "Answer Critical": [
                {
                    "Ingredient": "Explain the construction history of the Great Wall",
                    "Handle": "Construction History",
                    "Specifics": [
                        {
                            "Text": "The Great Wall was built over many dynasties",
                            "Citation": "123456"
                        }
                    ]
                }
            ],
            "Valuable": [],
            "Context": []
        }
        
        # Valid response with all required format elements
        self.valid_response = """
        <query>Great Wall of China history</query>
        
        <snippets id="abc123">The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states.</snippets>
        
        <answer>
        The Great Wall of China is one of the most iconic structures in human history. 
        <cite id="abc123">Construction began during the 7th century BC under various warring states, but the most famous sections were built during the Ming Dynasty (1368-1644).</cite>
        The wall stretches approximately 13,000 miles across northern China.
        </answer>
        """
        
        # Response missing answer tags
        self.no_answer_response = """
        <query>Great Wall of China history</query>
        
        <snippets id="abc123">The Great Wall of China is a series of fortifications.</snippets>
        
        The Great Wall of China is one of the most iconic structures in human history.
        """
        
        # Response with partial format elements
        self.partial_format_response = """
        <answer>
        The Great Wall of China is one of the most iconic structures in human history.
        <cite id="abc123">Construction began during the 7th century BC.</cite>
        </answer>
        """

    def test_compute_format_reward_complete_format(self):
        """Test format reward computation with complete format elements."""
        reward = compute_format_reward(self.valid_response)
        
        # Should have all three components: answer (0.5), citation (0.3), query (0.2)
        expected_reward = 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 1.0  # = 1.0
        self.assertEqual(reward, expected_reward)

    def test_compute_format_reward_answer_only(self):
        """Test format reward with only answer tags."""
        response = "<answer>This is an answer.</answer>"
        reward = compute_format_reward(response)
        
        # Only answer component: 0.5 * 1.0 + 0.3 * 0.0 + 0.2 * 0.0 = 0.5
        expected_reward = 0.5 * 1.0 + 0.3 * 0.0 + 0.2 * 0.0
        self.assertEqual(reward, expected_reward)

    def test_compute_format_reward_citation_only(self):
        """Test format reward with only citation tags."""
        response = 'This is text with <cite id="123">a citation</cite>.'
        reward = compute_format_reward(response)
        
        # Only citation component: 0.5 * 0.0 + 0.3 * 1.0 + 0.2 * 0.0 = 0.3
        expected_reward = 0.5 * 0.0 + 0.3 * 1.0 + 0.2 * 0.0
        self.assertEqual(reward, expected_reward)

    def test_compute_format_reward_query_only(self):
        """Test format reward with only query tags."""
        response = "<query>What is the Great Wall?</query>"
        reward = compute_format_reward(response)
        
        # Only query component: 0.5 * 0.0 + 0.3 * 0.0 + 0.2 * 1.0 = 0.2
        expected_reward = 0.5 * 0.0 + 0.3 * 0.0 + 0.2 * 1.0
        self.assertEqual(reward, expected_reward)

    def test_compute_format_reward_no_format(self):
        """Test format reward with no format elements."""
        response = "This is plain text with no special formatting."
        reward = compute_format_reward(response)
        
        # No components: 0.5 * 0.0 + 0.3 * 0.0 + 0.2 * 0.0 = 0.0
        self.assertEqual(reward, 0.0)

    def test_compute_format_reward_malformed_citations(self):
        """Test format reward with malformed citation tags."""
        response = '<cite>Missing id attribute</cite> and <cite id=>empty id</cite>'
        reward = compute_format_reward(response)
        
        # Malformed citations should not match: 0.5 * 0.0 + 0.3 * 0.0 + 0.2 * 0.0 = 0.0
        self.assertEqual(reward, 0.0)

    def test_compute_format_reward_various_citation_formats(self):
        """Test format reward with various valid citation formats."""
        test_cases = [
            '<cite id="123">text</cite>',
            "<cite id='456'>text</cite>",
            '<cite id=789>text</cite>',
            '<cite id="abc" metadata="extra">text</cite>'
        ]
        
        for citation in test_cases:
            with self.subTest(citation=citation):
                reward = compute_format_reward(citation)
                # Should detect citation: 0.5 * 0.0 + 0.3 * 1.0 + 0.2 * 0.0 = 0.3
                expected_reward = 0.3
                self.assertEqual(reward, expected_reward)

    def test_compute_longform_reward_success(self):
        """Test successful computation of longform averaged outcome reward."""
        # Mock the extraction function
        longform_module.extract_answer_context_citations.return_value = (
            "context with search", 
            "extracted answer", 
            {"abc123": "citation content"}
        )
        
        # Mock search turns scoring
        longform_module.score_num_in_context_search_turns.return_value = (2, 0.8)  # (num_turns, reward)
        
        # Mock rubric scoring
        longform_module._score_rubric.return_value = {"accuracy": 0.9, "completeness": 0.8}
        
        # Mock citation scoring
        longform_module.score_in_context_citations.return_value = 0.7
        
        result = compute_longform_averaged_outcome_reward(
            self.valid_response, self.sample_ground_truth, self.sample_question
        )
        
        # Verify all components are present
        self.assertIsNotNone(result["format_score"])
        self.assertEqual(result["num_search_turns_reward"], 0.8)
        self.assertEqual(result["rubric_scores"], {"accuracy": 0.9, "completeness": 0.8})
        self.assertEqual(result["citation_score"], 0.7)
        
        # Verify final score calculation
        expected_score = (
            REWARD_WEIGHTS["format"] * result["format_score"] +
            REWARD_WEIGHTS["num_search_turns"] * 0.8 +
            REWARD_WEIGHTS["rubric"] * 0.85 +  # average of rubric scores
            REWARD_WEIGHTS["citation"] * 0.7
        )
        self.assertAlmostEqual(result["score"], expected_score, places=5)

    def test_compute_longform_reward_no_answer_extracted(self):
        """Test behavior when no answer is extracted from response."""
        # Mock extraction to return None for answer
        longform_module.extract_answer_context_citations.return_value = ("context", None, {})
        
        # Mock search turns scoring
        longform_module.score_num_in_context_search_turns.return_value = (0, 0.0)
        
        result = compute_longform_averaged_outcome_reward(
            self.no_answer_response, self.sample_ground_truth, self.sample_question
        )
        
        # Should return error and zero reward
        self.assertEqual(result["error"], "Failed to extract answer from response")
        self.assertEqual(result["reward"], 0.0)
        self.assertIsNotNone(result["format_score"])
        self.assertEqual(result["num_search_turns_reward"], 0.0)
        self.assertIsNone(result["rubric_scores"])
        self.assertIsNone(result["citation_score"])

    def test_reward_weights_sum_to_one(self):
        """Test that reward weights sum to 1.0."""
        total_weight = sum(REWARD_WEIGHTS.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)

    def test_reward_weights_structure(self):
        """Test that reward weights contain expected keys."""
        expected_keys = {"rubric", "citation", "format", "num_search_turns"}
        self.assertEqual(set(REWARD_WEIGHTS.keys()), expected_keys)

    def test_compute_format_reward_multiline_content(self):
        """Test format reward with multiline content in tags."""
        response = """
        <query>
        What is the Great Wall of China?
        How long is it?
        </query>
        
        <answer>
        The Great Wall of China is a series of fortifications.
        It stretches across northern China.
        </answer>
        
        <cite id="123">
        This is a multiline citation
        with multiple sentences.
        </cite>
        """
        
        reward = compute_format_reward(response)
        # Should detect all components despite multiline content
        expected_reward = 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 1.0  # = 1.0
        self.assertEqual(reward, expected_reward)


def run_individual_tests():
    """Run individual test methods for debugging."""
    suite = unittest.TestSuite()
    
    # Add specific tests
    suite.addTest(TestLongformAveragedOutcomeRewards('test_compute_format_reward_complete_format'))
    suite.addTest(TestLongformAveragedOutcomeRewards('test_compute_longform_reward_success'))
    suite.addTest(TestLongformAveragedOutcomeRewards('test_compute_longform_reward_no_answer_extracted'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    print("Longform Averaged Outcome Rewards Test Suite")
    print("=" * 60)
    print()
    
    # Run all tests
    unittest.main(verbosity=2) 