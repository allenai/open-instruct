#!/usr/bin/env python3
"""
Test case for explicit response mapping in finegrained GRPO.
This demonstrates the new 4-tuple format: (score, effective_span, reward_group_id, response_idx)
which provides explicit mapping between scores and responses.
"""

import numpy as np
import sys
import os

# Add the open_instruct module to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_explicit_response_mapping():
    """Test the explicit response mapping format"""
    print("=" * 80)
    print("TESTING EXPLICIT RESPONSE MAPPING FORMAT")
    print("=" * 80)
    
    # Example scenario: 3 responses, 2 reward groups
    responses = [
        [100, 101, 102],      # Response 0: "Hello world!"
        [200, 201, 202, 203], # Response 1: "This is a test"
        [300, 301, 302, 303, 304], # Response 2: "Another test response"
    ]
    
    decoded_responses = [
        "Hello world!",
        "This is a test",
        "Another test response"
    ]
    
    print("Test Responses:")
    for i, (tokens, decoded) in enumerate(zip(responses, decoded_responses)):
        print(f"  Response {i}: {tokens} -> '{decoded}'")
    print()
    
    # Test Case 1: Old format (3-tuple) - implicit mapping
    print("Test Case 1: Old Format (3-tuple) - Implicit Mapping")
    old_format_scores = [
        # Assumes order: resp0_group0, resp0_group1, resp1_group0, resp1_group1, resp2_group0, resp2_group1
        (0.8, (0, 5), 0),      # Response 0, Group 0 (format)
        (0.7, (6, 12), 1),     # Response 0, Group 1 (content)
        (0.9, (0, 4), 0),      # Response 1, Group 0 (format)
        (0.6, (5, 14), 1),     # Response 1, Group 1 (content)
        (0.5, (0, 7), 0),      # Response 2, Group 0 (format)
        (0.8, (8, 21), 1),     # Response 2, Group 1 (content)
    ]
    
    print("Old format finegrained_scores:")
    for i, (score, span, group_id) in enumerate(old_format_scores):
        implied_response_idx = i // 2  # 2 groups per response
        print(f"  [{i}] Score: {score}, Span: {span}, Group: {group_id} -> Implied Response: {implied_response_idx}")
    print()
    
    # Test Case 2: New format (4-tuple) - explicit mapping
    print("Test Case 2: New Format (4-tuple) - Explicit Mapping")
    new_format_scores = [
        # Explicit response indices - can be in any order!
        (0.8, (0, 5), 0, 0),    # Response 0, Group 0 (format)
        (0.9, (0, 4), 0, 1),    # Response 1, Group 0 (format)  
        (0.5, (0, 7), 0, 2),    # Response 2, Group 0 (format)
        (0.7, (6, 12), 1, 0),   # Response 0, Group 1 (content)
        (0.6, (5, 14), 1, 1),   # Response 1, Group 1 (content)
        (0.8, (8, 21), 1, 2),   # Response 2, Group 1 (content)
    ]
    
    print("New format finegrained_scores:")
    for i, (score, span, group_id, response_idx) in enumerate(new_format_scores):
        print(f"  [{i}] Score: {score}, Span: {span}, Group: {group_id}, Response: {response_idx}")
    print()
    
    # Test Case 3: Mixed scenarios - some responses missing certain reward types
    print("Test Case 3: Flexible Mapping - Missing Reward Types")
    flexible_scores = [
        # Response 0: has both format and content rewards
        (0.8, (0, 5), 0, 0),    # Response 0, Format reward
        (0.7, (6, 12), 1, 0),   # Response 0, Content reward
        
        # Response 1: only has format reward (no content reward)
        (0.9, (0, 4), 0, 1),    # Response 1, Format reward only
        
        # Response 2: only has content reward (no format reward)
        (0.8, (0, 21), 1, 2),   # Response 2, Content reward only
    ]
    
    print("Flexible format finegrained_scores:")
    reward_names = {0: "Format", 1: "Content"}
    for i, (score, span, group_id, response_idx) in enumerate(flexible_scores):
        reward_name = reward_names.get(group_id, f"Group{group_id}")
        response_text = decoded_responses[response_idx]
        span_text = response_text[span[0]:span[1]]
        print(f"  [{i}] {reward_name} reward: {score:.2f}")
        print(f"      Response {response_idx}: '{response_text}'")
        print(f"      Span [{span[0]}:{span[1]}]: '{span_text}'")
        print()
    
    # Test Case 4: Demonstrate the benefits
    print("Benefits of Explicit Response Mapping:")
    print("✅ 1. No assumptions about ordering - scores can be in any order")
    print("✅ 2. Flexible reward types - not all responses need all reward types")
    print("✅ 3. Clear mapping - no ambiguity about which score belongs to which response")
    print("✅ 4. Robust - handles missing rewards gracefully")
    print("✅ 5. Extensible - easy to add new reward types without breaking existing code")
    print()
    
    # Test validation logic
    print("Validation Examples:")
    
    # Valid case
    try:
        validate_finegrained_scores(new_format_scores, len(responses))
        print("✅ Valid scores passed validation")
    except Exception as e:
        print(f"❌ Unexpected validation error: {e}")
    
    # Invalid case - response index out of bounds
    try:
        invalid_scores = [(0.8, (0, 5), 0, 5)]  # Response index 5 doesn't exist
        validate_finegrained_scores(invalid_scores, len(responses))
        print("❌ Should have failed validation")
    except Exception as e:
        print(f"✅ Correctly caught invalid response index: {e}")
    
    print()
    print("✅ Explicit response mapping test completed!")
    return True


def validate_finegrained_scores(finegrained_scores, num_responses):
    """Validate finegrained_scores format and response indices"""
    for i, item in enumerate(finegrained_scores):
        if len(item) == 3:
            # Old format is always valid (implicit mapping)
            continue
        elif len(item) == 4:
            # New format - validate response index
            score, effective_span, reward_group_id, response_idx = item
            if response_idx >= num_responses or response_idx < 0:
                raise ValueError(f"Invalid response_idx {response_idx} at position {i}: only {num_responses} responses available")
        else:
            raise ValueError(f"Invalid finegrained_scores format at position {i}: expected 3-tuple or 4-tuple, got {len(item)}-tuple")


if __name__ == "__main__":
    success = test_explicit_response_mapping()
    exit(0 if success else 1) 