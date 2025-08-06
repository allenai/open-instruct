#!/usr/bin/env python3
"""
Test cases for the new FinegrainedRewardOutput class.
This demonstrates the simple output format for finegrained reward functions.
"""

import sys
import os

# Add the open_instruct module to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_instruct.ground_truth_utils import FinegrainedRewardOutput


def test_basic_finegrained_reward_output():
    """Test basic FinegrainedRewardOutput functionality."""
    print("🧪 Testing FinegrainedRewardOutput basic functionality")
    print("=" * 60)
    
    # Create sample finegrained scores
    finegrained_scores = [
        (0.85, (0, 20), 0, 0),    # Response 0, group 0: score 0.85 for chars 0-20
        (0.72, (25, 45), 1, 0),   # Response 0, group 1: score 0.72 for chars 25-45
        (0.91, (50, 85), 2, 0),   # Response 0, group 2: score 0.91 for chars 50-85
        (0.78, (0, 15), 0, 1),    # Response 1, group 0: score 0.78 for chars 0-15
        (0.83, (20, 50), 1, 1),   # Response 1, group 1: score 0.83 for chars 20-50
    ]
    
    log_values = {
        "total_rewards": 5,
        "avg_score": 0.818,
        "groups_used": [0, 1, 2]
    }
    
    # Create output
    output = FinegrainedRewardOutput(
        finegrained_scores=finegrained_scores,
        log_values=log_values,
        cost=0.05,
        reasoning="Computed rewards for methodology, correctness, and clarity"
    )
    
    print(f"✅ Created output with {len(output)} finegrained scores")
    print(f"   Log values: {output.log_values}")
    print(f"   Cost: {output.cost}")
    print(f"   Reasoning: {output.reasoning}")
    print(f"   Sample score: {output.finegrained_scores[0]}")
    
    # Test iteration
    print("   All scores:")
    for i, score in enumerate(output):
        print(f"     {i}: score={score[0]:.2f}, span={score[1]}, group={score[2]}, response={score[3]}")
    
    print()
    return True


def test_filtering_methods():
    """Test filtering methods."""
    print("🔍 Testing filtering methods")
    print("=" * 60)
    
    finegrained_scores = [
        (0.85, (0, 20), 0, 0),    # Response 0, group 0
        (0.72, (25, 45), 1, 0),   # Response 0, group 1
        (0.91, (50, 85), 2, 0),   # Response 0, group 2
        (0.78, (0, 15), 0, 1),    # Response 1, group 0
        (0.83, (20, 50), 1, 1),   # Response 1, group 1
        (0.69, (0, 30), 0, 2),    # Response 2, group 0
    ]
    
    output = FinegrainedRewardOutput(finegrained_scores=finegrained_scores)
    
    # Test filtering by response
    response_0_scores = output.get_scores_for_response(0)
    print(f"✅ Response 0 has {len(response_0_scores)} scores:")
    for score in response_0_scores:
        print(f"   - Score: {score[0]:.2f}, Group: {score[2]}")
    
    # Test filtering by group
    group_0_scores = output.get_scores_for_group(0)
    print(f"✅ Group 0 has {len(group_0_scores)} scores:")
    for score in group_0_scores:
        print(f"   - Score: {score[0]:.2f}, Response: {score[3]}")
    
    print()
    return True


def test_validation():
    """Test validation of finegrained scores."""
    print("🛡️ Testing validation")
    print("=" * 60)
    
    # Test valid scores
    try:
        valid_scores = [
            (0.85, (0, 20), 0, 0),
            (0.72, (25, 45), 1, 0),
        ]
        output = FinegrainedRewardOutput(finegrained_scores=valid_scores)
        print("✅ Valid scores accepted")
    except Exception as e:
        print(f"❌ Unexpected error for valid scores: {e}")
    
    # Test invalid tuple length
    try:
        invalid_scores = [
            (0.85, (0, 20), 0),  # Missing response_idx
        ]
        output = FinegrainedRewardOutput(finegrained_scores=invalid_scores)
        print("❌ Should have failed: invalid tuple length")
    except ValueError as e:
        print(f"✅ Correctly caught invalid tuple length: {e}")
    
    # Test invalid span format
    try:
        invalid_scores = [
            (0.85, (0, 20, 30), 0, 0),  # 3-tuple span instead of 2-tuple
        ]
        output = FinegrainedRewardOutput(finegrained_scores=invalid_scores)
        print("❌ Should have failed: invalid span format")
    except ValueError as e:
        print(f"✅ Correctly caught invalid span format: {e}")
    
    # Test invalid span values
    try:
        invalid_scores = [
            (0.85, (20, 10), 0, 0),  # end < start
        ]
        output = FinegrainedRewardOutput(finegrained_scores=invalid_scores)
        print("❌ Should have failed: invalid span values")
    except ValueError as e:
        print(f"✅ Correctly caught invalid span values: {e}")
    
    # Test negative indices
    try:
        invalid_scores = [
            (0.85, (0, 20), -1, 0),  # negative reward_group_id
        ]
        output = FinegrainedRewardOutput(finegrained_scores=invalid_scores)
        print("❌ Should have failed: negative reward_group_id")
    except ValueError as e:
        print(f"✅ Correctly caught negative reward_group_id: {e}")
    
    print()
    return True


def test_usage_example():
    """Test realistic usage example for finegrained GRPO."""
    print("🌍 Realistic usage example")
    print("=" * 60)
    
    # This is your finegrained reward function that you'll implement
    def your_finegrained_reward_function(responses, decoded_responses, ground_truths, datasets, finish_reasons, infos, queries):
        """Your custom finegrained reward function that returns FinegrainedRewardOutput."""
        
        # Example: Different reward types for different spans
        finegrained_scores = []
        
        for resp_idx, decoded_resp in enumerate(decoded_responses):
            # Methodology reward for first part of response
            methodology_score = 0.8 + resp_idx * 0.1  # Vary by response
            finegrained_scores.append((methodology_score, (0, 25), 0, resp_idx))
            
            # Correctness reward for middle part
            correctness_score = 0.9 - resp_idx * 0.1  # Vary by response
            mid_start = len(decoded_resp) // 3
            mid_end = 2 * len(decoded_resp) // 3
            finegrained_scores.append((correctness_score, (mid_start, mid_end), 1, resp_idx))
            
            # Clarity reward for last part
            clarity_score = 0.7 + resp_idx * 0.05  # Vary by response
            end_start = 2 * len(decoded_resp) // 3
            finegrained_scores.append((clarity_score, (end_start, len(decoded_resp)), 2, resp_idx))
        
        # Calculate some aggregate metrics
        scores = [score[0] for score in finegrained_scores]
        log_values = {
            "total_rewards": len(finegrained_scores),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "num_responses": len(decoded_responses),
            "num_groups": len(set(score[2] for score in finegrained_scores))
        }
        
        return FinegrainedRewardOutput(
            finegrained_scores=finegrained_scores,
            log_values=log_values,
            cost=0.05,
            reasoning="Computed rewards for methodology, correctness, and clarity"
        )
    
    # This is the reward_fn wrapper that calls your function (similar to what's in fgrpo_fast.py)
    async def reward_fn_wrapper(responses, decoded_responses, ground_truths, datasets, finish_reasons, infos, queries=None):
        """
        This is the reward_fn wrapper in fgrpo_fast.py that needs to be updated.
        It calls your finegrained reward function and unpacks the result.
        """
        
        # Call your finegrained reward function
        finegrained_output = your_finegrained_reward_function(
            responses, decoded_responses, ground_truths, datasets, finish_reasons, infos, queries
        )
        
        # Unpack for fgrpo_fast.py format
        return finegrained_output.unpack_for_fgrpo()
    
    # Test with sample data
    responses = [["dummy", "tokens"], ["more", "dummy", "tokens"]]
    decoded_responses = [
        "Let me solve this step by step. First, I identify variables. Then I calculate the result.",
        "I think the answer is 7. Actually, let me double-check this calculation carefully."
    ]
    ground_truths = ["8", "8"]
    datasets = ["math", "math"]
    finish_reasons = ["stop", "stop"]
    infos = [[1, 2], [3, 4]]
    queries = ["What is 3+5?", "What is 3+5?"]
    
    # Test the wrapper (this is what fgrpo_fast.py calls)
    import asyncio
    finegrained_scores, log_values = asyncio.run(
        reward_fn_wrapper(responses, decoded_responses, ground_truths, datasets, finish_reasons, infos, queries)
    )
    
    print(f"✅ reward_fn wrapper returned:")
    print(f"   finegrained_scores: {len(finegrained_scores)} scores")
    print(f"   log_values: {log_values}")
    
    # Show how fgrpo_fast.py would use this
    print("\n📋 How fgrpo_fast.py processes this (line 1175-1179):")
    print("   finegrained_scores, reward_metrics = asyncio.run(")
    print("       reward_fn(responses, decoded_responses, ground_truths, datasets, finish_reasons, infos, decoded_queries)")
    print("   )")
    print(f"   # finegrained_scores = {finegrained_scores[:2]}...")
    print(f"   # reward_metrics = {log_values}")
    
    # Show the exact format expected by fgrpo_fast.py
    print("\n🔄 Direct usage in fgrpo_fast.py:")
    for i, score_tuple in enumerate(finegrained_scores[:3]):  # Show first 3
        score, span, group, resp = score_tuple
        print(f"   Score {i}: {score:.3f} for chars {span} (group {group}, response {resp})")
    
    print("\n💡 What you need to do:")
    print("   1. Implement your finegrained reward function that returns FinegrainedRewardOutput")
    print("   2. Update the reward_fn wrapper in fgrpo_fast.py to call your function")
    print("   3. Use finegrained_output.unpack_for_fgrpo() to return the expected format")
    print("\n🎯 The dummy function is now available:")
    print("   from open_instruct.search_rewards.finegrained_rewards import compute_finegrained_reward")
    print("   result = compute_finegrained_reward(prediction, label, query)")
    print("   # result['finegrained_scores'] = [(score, (start, end), group_id, response_idx), ...]")
    print("   # result['log_values'] = {'first_half_score': 0.8, 'second_half_score': 0.9, ...}")
    
    print()
    return True


def main():
    """Run all tests."""
    print("🧪 FINEGRAINED REWARD OUTPUT TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_basic_finegrained_reward_output,
        test_filtering_methods,
        test_validation,
        test_usage_example,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test {test.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 80)
    print(f"🎯 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The FinegrainedRewardOutput class is working correctly.")
        print("\n💡 Usage in your reward function:")
        print("   return FinegrainedRewardOutput(")
        print("       finegrained_scores=[(score, (start, end), group_id, response_idx), ...],")
        print("       reward_metrics={'avg_score': 0.8, 'total_rewards': 10}")
        print("   )")
        return True
    else:
        print("💥 Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 