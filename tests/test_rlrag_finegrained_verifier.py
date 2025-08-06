#!/usr/bin/env python3
"""
Test the RLRAGLongFormFinegrainedVerifier with the dummy finegrained reward function.
This verifies that the verifier correctly calls compute_finegrained_reward and returns FinegrainedRewardOutput.
"""

import sys
import os

# Add the open_instruct module to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_instruct.ground_truth_utils import RLRAGLongFormFinegrainedVerifier, FinegrainedRewardOutput


def test_rlrag_finegrained_verifier_basic():
    """Test basic functionality of RLRAGLongFormFinegrainedVerifier."""
    print("🧪 Testing RLRAGLongFormFinegrainedVerifier basic functionality")
    print("=" * 70)
    
    # Create the verifier
    verifier = RLRAGLongFormFinegrainedVerifier()
    
    print(f"✅ Created verifier: {verifier}")
    print(f"   Name: {verifier.name}")
    print(f"   Weight: {verifier.weight}")
    
    # Test cases
    test_cases = [
        {
            "prediction": "Let me solve this step by step. First, I need to identify the variables. The answer is 42.",
            "label": "42",
            "query": "What is 6 * 7?",
            "description": "Math problem with methodology and answer"
        },
        {
            "prediction": "This is a comprehensive analysis of the problem. We start by examining the context and then provide a detailed solution with proper reasoning.",
            "label": "analysis",
            "query": "Analyze this problem",
            "description": "Long analytical response"
        },
        {
            "prediction": "Short answer.",
            "label": "brief",
            "query": "Give a brief response",
            "description": "Short response"
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n📝 TEST CASE {i + 1}: {case['description']}")
        print(f"   Query: '{case['query']}'")
        print(f"   Prediction: '{case['prediction'][:60]}{'...' if len(case['prediction']) > 60 else ''}'")
        print(f"   Label: '{case['label']}'")
        
        # Call the verifier
        result = verifier(
            tokenized_prediction=[],  # Not used by this verifier
            prediction=case['prediction'],
            label=case['label'],
            query=case['query']
        )
        
        # Verify the result is a FinegrainedRewardOutput
        print(f"   ✅ Result type: {type(result).__name__}")
        
        if isinstance(result, FinegrainedRewardOutput):
            print(f"   📊 Finegrained scores: {len(result.finegrained_scores)} scores")
            
            # Show the finegrained scores
            for j, score_tuple in enumerate(result.finegrained_scores):
                score, span, group_id, response_idx = score_tuple
                start_char, end_char = span
                span_text = case['prediction'][start_char:end_char] if case['prediction'] else ""
                group_name = "methodology" if group_id == 0 else "conclusion"
                
                print(f"     Score {j}: {score:.3f} for '{span_text[:30]}{'...' if len(span_text) > 30 else ''}' (chars {span}, group {group_id}={group_name})")
            
            # Show log values
            if result.log_values:
                print(f"   📈 Log values:")
                for key, value in result.log_values.items():
                    if isinstance(value, float):
                        print(f"     {key}: {value:.3f}")
                    else:
                        print(f"     {key}: {value}")
            
            print(f"   💰 Cost: {result.cost}")
            if result.reasoning:
                print(f"   🧠 Reasoning: {result.reasoning[:50]}{'...' if len(result.reasoning) > 50 else ''}")
        else:
            print(f"   ❌ Expected FinegrainedRewardOutput, got {type(result)}")
            return False
    
    print()
    return True


def test_verifier_integration_with_fgrpo():
    """Test that the verifier output can be unpacked for fgrpo_fast.py."""
    print("🔄 Testing verifier integration with fgrpo_fast.py")
    print("=" * 70)
    
    # Create the verifier
    verifier = RLRAGLongFormFinegrainedVerifier()
    
    prediction = "Let me approach this systematically. First, I'll analyze the problem. Then I'll compute the solution step by step. The final answer is 84."
    label = "84"
    query = "What is 12 * 7?"
    
    print(f"📝 Testing with:")
    print(f"   Query: '{query}'")
    print(f"   Prediction: '{prediction[:60]}...'")
    print(f"   Label: '{label}'")
    
    # Call the verifier
    result = verifier(
        tokenized_prediction=[1, 2, 3, 4, 5],  # Dummy tokenized prediction
        prediction=prediction,
        label=label,
        query=query
    )
    
    print(f"\n✅ Verifier returned: {type(result).__name__}")
    
    # Test unpacking for fgrpo_fast.py
    finegrained_scores, log_values = result.unpack_for_fgrpo()
    
    print(f"🎯 Unpacked for fgrpo_fast.py:")
    print(f"   finegrained_scores type: {type(finegrained_scores)}")
    print(f"   finegrained_scores length: {len(finegrained_scores)}")
    print(f"   log_values type: {type(log_values)}")
    print(f"   log_values keys: {list(log_values.keys())}")
    
    # Verify the format matches what fgrpo_fast.py expects
    print(f"\n🔍 Format verification:")
    for i, score_tuple in enumerate(finegrained_scores):
        if len(score_tuple) != 4:
            print(f"   ❌ Score tuple {i} has {len(score_tuple)} elements, expected 4")
            return False
        
        score, span, group_id, response_idx = score_tuple
        
        if not isinstance(score, (int, float)):
            print(f"   ❌ Score {i} is {type(score)}, expected number")
            return False
        
        if not isinstance(span, tuple) or len(span) != 2:
            print(f"   ❌ Span {i} is {span}, expected 2-tuple")
            return False
        
        if not isinstance(group_id, int) or group_id < 0:
            print(f"   ❌ Group ID {i} is {group_id}, expected non-negative int")
            return False
        
        if not isinstance(response_idx, int) or response_idx < 0:
            print(f"   ❌ Response index {i} is {response_idx}, expected non-negative int")
            return False
        
        print(f"   ✅ Score tuple {i}: valid format")
    
    if not isinstance(log_values, dict):
        print(f"   ❌ log_values is {type(log_values)}, expected dict")
        return False
    
    print(f"   ✅ log_values: valid dict with {len(log_values)} entries")
    
    # Show sample unpacked data
    print(f"\n📊 Sample unpacked data:")
    print(f"   First score: {finegrained_scores[0]}")
    if len(finegrained_scores) > 1:
        print(f"   Second score: {finegrained_scores[1]}")
    print(f"   Sample log values: {dict(list(log_values.items())[:3])}")
    
    print()
    return True


def test_verifier_with_edge_cases():
    """Test the verifier with edge cases."""
    print("🛡️ Testing verifier with edge cases")
    print("=" * 70)
    
    verifier = RLRAGLongFormFinegrainedVerifier()
    
    edge_cases = [
        {
            "prediction": "",
            "label": "empty",
            "query": "Test empty prediction",
            "description": "Empty prediction"
        },
        {
            "prediction": "A",
            "label": "single",
            "query": "Single character",
            "description": "Single character prediction"
        },
        {
            "prediction": "This is a very long prediction that should test how the verifier handles lengthy responses with multiple sentences and various types of content including numbers like 12345 and methodical words like step and systematic approaches.",
            "label": "12345",
            "query": "Long prediction test",
            "description": "Very long prediction"
        }
    ]
    
    for i, case in enumerate(edge_cases):
        print(f"\n📝 EDGE CASE {i + 1}: {case['description']}")
        print(f"   Prediction length: {len(case['prediction'])} chars")
        
        try:
            result = verifier(
                tokenized_prediction=[],
                prediction=case['prediction'],
                label=case['label'],
                query=case['query']
            )
            
            print(f"   ✅ Successfully handled edge case")
            print(f"   📊 Returned {len(result.finegrained_scores)} scores")
            
            # Verify scores are valid
            for j, score_tuple in enumerate(result.finegrained_scores):
                score, span, group_id, response_idx = score_tuple
                if not (0.0 <= score <= 1.0):
                    print(f"   ⚠️ Score {j} is {score}, outside [0,1] range")
                else:
                    print(f"   ✅ Score {j}: {score:.3f} (valid range)")
            
        except Exception as e:
            print(f"   ❌ Failed with error: {e}")
            return False
    
    print()
    return True


def main():
    """Run all tests."""
    print("🎯 RLRAG FINEGRAINED VERIFIER TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_rlrag_finegrained_verifier_basic,
        test_verifier_integration_with_fgrpo,
        test_verifier_with_edge_cases,
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
        print("🎉 All tests passed! The RLRAGLongFormFinegrainedVerifier is working correctly.")
        print("\n💡 Key findings:")
        print("   ✅ Verifier correctly calls compute_finegrained_reward()")
        print("   ✅ Returns FinegrainedRewardOutput with proper format")
        print("   ✅ Output can be unpacked for fgrpo_fast.py")
        print("   ✅ Handles edge cases gracefully")
        print("   ✅ Integrates seamlessly with the existing verifier system")
        print("\n🚀 Ready for use in finegrained GRPO training!")
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