#!/usr/bin/env python3
"""
Test the dummy finegrained reward function.
This demonstrates how the compute_finegrained_reward function works.
"""

import sys
import os

# Add the open_instruct module to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_instruct.search_rewards.finegrained_rewards import compute_finegrained_reward
from open_instruct.ground_truth_utils import FinegrainedRewardOutput


def test_dummy_finegrained_reward():
    """Test the dummy finegrained reward function."""
    print("🧪 Testing dummy finegrained reward function")
    print("=" * 60)
    
    # Test cases with different types of responses
    test_cases = [
        {
            "prediction": "Let me think step by step. First, I need to solve this problem. The answer is 42.",
            "label": "42",
            "query": "What is 6 * 7?",
            "description": "Math problem with methodology and answer"
        },
        {
            "prediction": "Hello world!",
            "label": "greeting",
            "query": "Say hello",
            "description": "Short response"
        },
        {
            "prediction": "",
            "label": "empty",
            "query": "Test empty",
            "description": "Empty response"
        },
        {
            "prediction": "This is a longer response that should be split into two parts. The first part contains some setup and explanation. The second part contains the conclusion and final result with numbers like 123.",
            "label": "123",
            "query": "Solve this problem",
            "description": "Long response with setup and conclusion"
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n📝 TEST CASE {i + 1}: {case['description']}")
        print(f"   Query: '{case['query']}'")
        print(f"   Prediction: '{case['prediction'][:50]}{'...' if len(case['prediction']) > 50 else ''}'")
        print(f"   Label: '{case['label']}'")
        
        # Call the dummy reward function
        result = compute_finegrained_reward(
            prediction=case['prediction'],
            label=case['label'],
            query=case['query']
        )
        
        print(f"   ✅ Returned {len(result['finegrained_scores'])} finegrained scores:")
        
        # Show the finegrained scores
        for j, score_tuple in enumerate(result['finegrained_scores']):
            score, span, group_id, response_idx = score_tuple
            start_char, end_char = span
            span_text = case['prediction'][start_char:end_char] if case['prediction'] else ""
            group_name = "methodology" if group_id == 0 else "conclusion"
            
            print(f"     Score {j}: {score:.3f} for '{span_text}' (chars {span}, group {group_id}={group_name}, response {response_idx})")
        
        # Show log values
        print(f"   📊 Log values:")
        for key, value in result['log_values'].items():
            print(f"     {key}: {value}")
    
    print()
    return True


def test_with_finegrained_reward_output():
    """Test using the result with FinegrainedRewardOutput class."""
    print("🔄 Testing integration with FinegrainedRewardOutput")
    print("=" * 60)
    
    prediction = "Let me solve this step by step. The answer is 42 because 6 times 7 equals 42."
    label = "42"
    query = "What is 6 * 7?"
    
    # Get result from dummy function
    result = compute_finegrained_reward(prediction, label, query)
    
    # Create FinegrainedRewardOutput
    output = FinegrainedRewardOutput(
        finegrained_scores=result['finegrained_scores'],
        log_values=result['log_values'],
        cost=0.01,  # Dummy cost
        reasoning="Computed using dummy heuristics for first/second half"
    )
    
    print(f"✅ Created FinegrainedRewardOutput:")
    print(f"   Finegrained scores: {len(output)} scores")
    print(f"   Log values: {output.log_values}")
    print(f"   Cost: {output.cost}")
    print(f"   Reasoning: {output.reasoning}")
    
    # Test unpacking for fgrpo_fast.py
    finegrained_scores, log_values = output.unpack_for_fgrpo()
    
    print(f"\n🎯 Unpacked for fgrpo_fast.py:")
    print(f"   finegrained_scores: {finegrained_scores}")
    print(f"   log_values: {log_values}")
    
    # Test filtering methods
    print(f"\n🔍 Filtering:")
    response_0_scores = output.get_scores_for_response(0)
    print(f"   Response 0 scores: {len(response_0_scores)}")
    
    group_0_scores = output.get_scores_for_group(0)
    group_1_scores = output.get_scores_for_group(1)
    print(f"   Group 0 (methodology) scores: {len(group_0_scores)}")
    print(f"   Group 1 (conclusion) scores: {len(group_1_scores)}")
    
    print()
    return True


def main():
    """Run all tests."""
    print("🎯 DUMMY FINEGRAINED REWARDS TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_dummy_finegrained_reward,
        test_with_finegrained_reward_output,
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
        print("🎉 All tests passed! The dummy finegrained reward function is working correctly.")
        print("\n💡 The dummy function:")
        print("   - Splits predictions into first half (methodology) and second half (conclusion)")
        print("   - Assigns reward group 0 to first half, group 1 to second half")
        print("   - Uses simple heuristics for scoring (length, keywords, numbers)")
        print("   - Returns proper format for finegrained GRPO")
        print("   - You can now replace the dummy logic with your actual reward computation!")
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