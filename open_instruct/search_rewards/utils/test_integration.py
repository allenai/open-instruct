#!/usr/bin/env python3
"""
Integration test for the adaptive rubrics feature.

This test demonstrates the full flow:
1. Initialize rubric buffer from ground truths
2. Generate adaptive rubrics for a batch of responses
3. Update ground truths with adaptive rubrics
4. Score responses using RubricVerifier

Usage:
    # With API keys for full testing:
    OPENAI_API_KEY=sk-xxx python -m open_instruct.search_rewards.utils.test_integration

    # Without API keys (will skip live tests):
    python -m open_instruct.search_rewards.utils.test_integration
"""

import asyncio
import json
import os
import sys

# Add the repo root to path if needed
if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


async def run_integration_test():
    """Run the full integration test."""
    from open_instruct.ground_truth_utils import RubricVerifier, RubricVerifierConfig  # noqa: PLC0415
    from open_instruct.search_rewards.utils.rubric_utils import (  # noqa: PLC0415
        _generate_instance_wise_adaptive_rubrics,
        initialize_rubric_buffer,
        update_ground_truths_with_adaptive_rubrics,
    )

    print("=" * 70)
    print("ADAPTIVE RUBRICS - INTEGRATION TEST")
    print("=" * 70)

    # Check if we have API credentials
    has_api = os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")

    # =========================================================================
    # Step 1: Create sample data
    # =========================================================================
    print("\n📋 Step 1: Create sample data")
    print("-" * 50)

    # Sample ground truths (2 questions, each with initial rubrics)
    base_ground_truths = [
        {
            "query": "Explain how neural networks learn through backpropagation.",
            "rubrics": [
                {"description": "Explains the concept of gradient descent", "weight": 1.0},
                {"description": "Mentions loss function optimization", "weight": 1.0},
            ],
        },
        {
            "query": "What are the benefits of renewable energy?",
            "rubrics": [
                {"description": "Lists environmental benefits", "weight": 1.0},
                {"description": "Mentions sustainability", "weight": 0.5},
            ],
        },
    ]

    # Sample responses (2 prompts × 3 samples per prompt = 6 total)
    num_samples_per_prompt = 3
    responses = [
        # Responses for Q1 (neural networks)
        "Backpropagation is how neural networks learn by calculating gradients of the loss function with respect to weights and adjusting them using gradient descent.",
        "Neural networks learn by propagating errors backwards and updating weights to minimize loss.",
        "I think neural networks are like brains that learn things.",
        # Responses for Q2 (renewable energy)
        "Renewable energy offers environmental benefits like reduced carbon emissions, sustainability through infinite supply, and economic benefits through job creation.",
        "Solar and wind energy are good because they don't pollute as much.",
        "Renewable energy is better than fossil fuels.",
    ]

    # Expand ground truths to match response structure (each GT repeated for each sample)
    ground_truths = []
    for gt in base_ground_truths:
        for _ in range(num_samples_per_prompt):
            ground_truths.append([json.dumps(gt)])

    print(f"  Created {len(base_ground_truths)} unique prompts")
    print(f"  {num_samples_per_prompt} samples per prompt")
    print(f"  Total ground truths: {len(ground_truths)}")

    # =========================================================================
    # Step 2: Initialize rubric buffer
    # =========================================================================
    print("\n📋 Step 2: Initialize rubric buffer")
    print("-" * 50)

    rubric_buffer = initialize_rubric_buffer(
        [json.dumps(gt) for gt in base_ground_truths], use_static_rubrics_as_persistent=True
    )

    for query, buffer_entry in rubric_buffer.items():
        print(f"  Query: {query[:50]}...")
        print(f"    Persistent rubrics: {len(buffer_entry['persistent_rubrics'])}")
        print(f"    Active rubrics: {len(buffer_entry['active_rubrics'])}")

    # =========================================================================
    # Step 3: Generate adaptive rubrics (requires API)
    # =========================================================================
    print("\n📋 Step 3: Generate adaptive rubrics")
    print("-" * 50)

    if has_api:
        print("  Using live API to generate adaptive rubrics...")
        try:
            adaptive_rubrics, num_subsampled = await _generate_instance_wise_adaptive_rubrics(
                responses=responses,
                ground_truths=ground_truths,
                num_samples_per_prompt_rollout=num_samples_per_prompt,
                rubric_buffer=rubric_buffer,
                use_full_responses=True,
            )

            for i, rubric in enumerate(adaptive_rubrics):
                if rubric:
                    pos = len(rubric.get("positive_rubrics", []))
                    neg = len(rubric.get("negative_rubrics", []))
                    print(f"  Prompt {i + 1}: {pos} positive, {neg} negative rubrics")
                else:
                    print(f"  Prompt {i + 1}: Generation failed (None)")

        except Exception as e:
            print(f"  ⚠️ Error generating rubrics: {e}")
            # Create mock rubrics for testing rest of flow
            adaptive_rubrics = [
                {
                    "positive_rubrics": [{"description": "Clear explanation", "title": "Clarity"}],
                    "negative_rubrics": [],
                },
                {"positive_rubrics": [], "negative_rubrics": [{"description": "Too vague", "title": "Vagueness"}]},
            ]
    else:
        print("  ⚠️ No API credentials - using mock adaptive rubrics")
        adaptive_rubrics = [
            {
                "positive_rubrics": [{"description": "Provides technical depth", "title": "Technical Depth"}],
                "negative_rubrics": [],
            },
            {
                "positive_rubrics": [{"description": "Comprehensive coverage", "title": "Coverage"}],
                "negative_rubrics": [{"description": "Lacks specifics", "title": "Specificity"}],
            },
        ]

    # =========================================================================
    # Step 4: Update ground truths with adaptive rubrics
    # =========================================================================
    print("\n📋 Step 4: Update ground truths with adaptive rubrics")
    print("-" * 50)

    (
        updated_ground_truths,
        valid_rate,
        avg_gt_rubrics,
        avg_adaptive_rubrics,
        avg_active_buffer,
        updated_buffer,
        skipped,
    ) = update_ground_truths_with_adaptive_rubrics(
        ground_truths=ground_truths.copy(),
        all_adaptive_rubrics=adaptive_rubrics,
        num_samples_per_prompt_rollout=num_samples_per_prompt,
        rubric_buffer=rubric_buffer,
    )

    print(f"  Valid adaptive rubric rate: {valid_rate:.2%}")
    print(f"  Avg ground truth rubrics: {avg_gt_rubrics:.1f}")
    print(f"  Avg adaptive rubrics added: {avg_adaptive_rubrics:.1f}")
    print(f"  Avg active buffer rubrics: {avg_active_buffer:.1f}")
    print(f"  Skipped (failed generation): {skipped}")

    # Show updated buffer state
    print("\n  Updated buffer state:")
    for query, buffer_entry in updated_buffer.items():
        print(f"    Query: {query[:40]}...")
        print(f"      Active rubrics: {len(buffer_entry['active_rubrics'])}")

    # =========================================================================
    # Step 5: Score responses using RubricVerifier
    # =========================================================================
    print("\n📋 Step 5: Score responses using RubricVerifier")
    print("-" * 50)

    if has_api:
        config = RubricVerifierConfig(rubric_judge_model=os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1-mini"))
        verifier = RubricVerifier(config)

        print("  Scoring responses against rubrics...")
        for i, (gt, response) in enumerate(zip(updated_ground_truths, responses)):
            gt_obj = json.loads(gt[0])
            try:
                result = await verifier.async_call(
                    tokenized_prediction=[1, 2, 3], prediction=response, label=gt_obj, query=gt_obj["query"]
                )
                print(f"  Response {i + 1}: Score = {result.score:.3f}")
                print(f"    Response: {response[:60]}...")
            except Exception as e:
                print(f"  Response {i + 1}: Error - {e}")
    else:
        print("  ⚠️ Skipping scoring - no API credentials")
        print("  Set OPENAI_API_KEY or AZURE_API_KEY to test scoring")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETED")
    print("=" * 70)

    print("\n📊 Summary:")
    print(f"  - Processed {len(base_ground_truths)} unique prompts")
    print(f"  - Generated {len(adaptive_rubrics)} sets of adaptive rubrics")
    print(f"  - Updated {len(updated_ground_truths)} ground truths")
    print(
        f"  - Rubric buffer now has {sum(len(b['active_rubrics']) for b in updated_buffer.values())} total active rubrics"
    )

    if has_api:
        print("\n✅ Full integration test completed with live API calls")
    else:
        print("\n⚠️ Integration test completed with mock data (no API)")
        print("   Set OPENAI_API_KEY to run full test")

    return True


def main():
    """Main entry point."""
    try:
        result = asyncio.run(run_integration_test())
        return 0 if result else 1
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
