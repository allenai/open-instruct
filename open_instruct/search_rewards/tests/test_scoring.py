#!/usr/bin/env python3

import os
import sys

# Add parent directory to path to import run_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_instruct.search_rewards.openscholar_rewards_utils import (
    RubricCorpusQaGenericMetric,
)

from .test_answer import test_answer
from .test_case import test_case


def test_scoring_function():
    """
    Test the scoring function with the provided test case and draft answer.
    """
    print("Testing the scoring function with the provided test case...")
    print("=" * 80)

    # Extract the metric config from the test case
    metric_config = test_case[0]["metric_config"]["config"]

    # Initialize the scoring metric
    metric = RubricCorpusQaGenericMetric(metric_config)

    # Score the answer
    print("Scoring the draft answer...")
    print("-" * 40)
    print("Draft Answer:")
    print(test_answer)
    print("-" * 40)

    try:
        scores = metric.score_output(test_answer)

        print("\nSCORING RESULTS:")
        print("=" * 50)
        print(f"Overall Score: {scores['score']:.4f}")
        print(f"Annotation Score: {scores['ann_score']:.4f}")
        print("\nDetailed Component Scores:")
        print("-" * 30)

        # Print individual component scores
        for component, score in scores.items():
            if component not in ["score", "ann_score"]:
                print(f"{component}: {score:.4f}")

        print("\n" + "=" * 50)
        print("ANALYSIS:")
        print("-" * 20)

        # Analyze the scores
        if scores["score"] >= 0.8:
            print("✅ Excellent score - The answer comprehensively addresses the criteria")
        elif scores["score"] >= 0.6:
            print("✅ Good score - The answer addresses most criteria well")
        elif scores["score"] >= 0.4:
            print("⚠️  Moderate score - The answer addresses some criteria but needs improvement")
        else:
            print("❌ Low score - The answer needs significant improvement")

        # Check specific criteria
        print(f"\nLength score: {scores.get('length', 'N/A'):.4f}")
        print(f"Expertise score: {scores.get('expertise', 'N/A'):.4f}")

        # Check rubric criteria
        rubric_scores = {
            k: v for k, v in scores.items() if k.startswith("most_important_item") or k.startswith("nice_to_have_item")
        }
        print("\nRubric Criteria Scores:")
        for criterion, score in rubric_scores.items():
            status = "✅" if score >= 0.7 else "⚠️" if score >= 0.4 else "❌"
            print(f"  {criterion}: {score:.4f} {status}")

    except Exception as e:
        print(f"Error during scoring: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_scoring_function()
