#!/usr/bin/env python3

import logging
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_instruct.search_rewards.long_form_rewards import compute_paper_reward
from open_instruct.search_rewards.reasoning_model_rewards import compute_hle_reward

# Import the example from formatted_test_answer.py
from open_instruct.search_rewards.tests.formatted_test_answer import example_answer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a simple test case for the Great Wall of China example
test_case = {
    "initial_prompt": "Tell me about the Great Wall of China",
    "metric_config": {
        "name": "rubric_corpusqa_generic",
        "config": {
            "question": "Tell me about the Great Wall of China",
            "low_length": 200,
            "high_length": 800,
            "length_weight": 0.1,
            "expertise_weight": 0.1,
            "citations_weight": 0.3,
            "excerpts_weight": 0.2,
            "model_name": "gpt-4-turbo",
            "other_properties": [
                {
                    "name": "most_important_item_0",
                    "criterion": "The answer should provide historical context about when the Great Wall was built.",
                    "weight": 0.1,
                    "evidence": [
                        "Construction of the Great Wall began during the 7th century BC under various warring states, but the most famous sections were built during the Ming Dynasty (1368-1644)."
                    ],
                },
                {
                    "name": "most_important_item_1",
                    "criterion": "The answer should explain the purpose and function of the Great Wall.",
                    "weight": 0.1,
                    "evidence": [
                        "The wall was primarily constructed as a defensive fortification to protect Chinese states from invasions by nomadic groups from the north."
                    ],
                },
                {
                    "name": "nice_to_have_item_0",
                    "criterion": "The answer should mention construction materials and methods.",
                    "weight": 0.1,
                    "evidence": [
                        "The wall incorporates various materials including stone, brick, tamped earth, wood, and other materials, with different sections built using locally available resources."
                    ],
                },
            ],
        },
    },
    "case_id": "great_wall_test_001",
}

def test_formatted_example(reward_type: str = "hle"):
    """Test the reward computation with the formatted example from formatted_test_answer.py"""
    print("Testing reward computation with formatted example...")
    print("=" * 60)

    # Use the example_answer directly
    full_response = example_answer

    print("Test case info:")
    print(f"Question: {test_case['metric_config']['config']['question']}")
    print(f"Case ID: {test_case['case_id']}")
    print()

    print("Response length:", len(full_response))
    print()

    print("Computing reward...")
    if reward_type == "paper":
        result = compute_paper_reward(full_response, test_case)
    elif reward_type == "hle":
        question = test_case['metric_config']['config']['question']
        correct_answer = "The Great Wall of China, one of the most iconic structures in human history, stretches approximately 13,000 miles across northern China. Construction of the Great Wall began during the 8th century BC under various warring states, but the most famous sections were built during the Ming Dynasty (1368-1644)."
        result = compute_hle_reward(full_response, correct_answer, question)

    print("Results:")
    print("-" * 40)
    print(f"Extraction success: {result['extraction_success']}")
    print(f"Reward score: {result['reward']:.4f}")

    if result["error"]:
        print(f"Error: {result['error']}")
        print("\nFull error details:")
        print("-" * 20)
        import traceback

        traceback.print_exc()
    else:
        print("\nScoring components:")
        scoring_results = result["scoring_results"]
        for key, value in scoring_results.items():
            if key not in ["score", "ann_score"]:
                print(f"  {key}: {value:.4f}")
        print(f"\nOverall score: {scoring_results['score']:.4f}")
        if "ann_score" in scoring_results:
            print(f"Annotation score: {scoring_results['ann_score']:.4f}")
        else:
            print("No annotation score found")

        # Analyze the score
        print("\nScore Analysis:")
        print("-" * 20)
        if scoring_results["score"] >= 0.8:
            print("✅ Excellent score - Comprehensive and well-cited answer")
        elif scoring_results["score"] >= 0.6:
            print("✅ Good score - Addresses most criteria well")
        elif scoring_results["score"] >= 0.4:
            print("⚠️  Moderate score - Some criteria addressed but needs improvement")
        else:
            print("❌ Low score - Significant improvement needed")

    print("\nCitations extracted:")
    print("-" * 20)
    for citation_id, citation_text in result["citations"].items():
        print(f"ID: {citation_id}")
        print(f"Text: {citation_text[:100]}...")
        print()

    print("=" * 60)


if __name__ == "__main__":
    test_formatted_example(reward_type="hle")
