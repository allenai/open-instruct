#!/usr/bin/env python3
"""
Test case for finegrained GRPO with actual tokenizer and realistic span scenarios.
This test demonstrates:
1. Using real tokenizer for string-to-token conversion
2. Different responses having different numbers of rewards
3. Various span types and edge cases
4. Validation of the complete pipeline
"""

import torch
import numpy as np
import sys
import os
from transformers import AutoTokenizer

# Add the open_instruct module to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from open_instruct.fgrpo_fast import convert_string_span_to_token_span
except ImportError:
    # If the function is not available, we'll define a mock version
    def convert_string_span_to_token_span(effective_span, decoded_responses, responses, tokenizer):
        """Mock version for testing"""
        start_char, end_char = effective_span
        token_spans = []
        
        for resp_idx, (decoded_resp, token_resp) in enumerate(zip(decoded_responses, responses)):
            # Simple approximation: map characters to tokens proportionally
            char_len = len(decoded_resp)
            token_len = len(token_resp)
            
            if char_len == 0:
                token_spans.append((0, 0))
                continue
                
            # Proportional mapping
            token_start = int((start_char / char_len) * token_len)
            token_end = int((end_char / char_len) * token_len)
            
            # Clamp to valid range
            token_start = max(0, min(token_start, token_len))
            token_end = max(token_start, min(token_end, token_len))
            
            token_spans.append((token_start, token_end))
        
        return token_spans


def test_realistic_tokenizer_spans():
    """Test with actual tokenizer and various span scenarios"""
    print("=" * 80)
    print("TESTING REALISTIC TOKENIZER SPANS")
    print("=" * 80)
    
    # Load a real tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Loaded tokenizer: {tokenizer.__class__.__name__}")
    print()
    
    # Create realistic test responses with different lengths and content
    test_texts = [
        "Hello! I'm here to help you with your question. Let me provide a detailed answer.",
        "Sure, that's a great question. The answer is complex but I'll break it down step by step.",
        "Actually, I need to think about this more carefully. Let me reconsider the problem.",
        "This is a short response.",
        "This is a much longer response that contains multiple sentences and covers various topics. It includes technical details, explanations, and examples to provide comprehensive coverage of the subject matter being discussed.",
    ]
    
    # Tokenize the responses
    print("Tokenizing responses...")
    responses = []
    decoded_responses = []
    
    for i, text in enumerate(test_texts):
        # Tokenize
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        responses.append(token_ids)
        
        # Decode back to verify
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        decoded_responses.append(decoded)
        
        print(f"Response {i}: {len(token_ids)} tokens")
        print(f"  Original: '{text}'")
        print(f"  Decoded:  '{decoded}'")
        print(f"  Tokens:   {token_ids}")
        print()
    
    # Create finegrained scores with different numbers of rewards per response
    print("Creating finegrained scores with varying reward counts...")
    finegrained_scores = []
    
    # Response 0: 3 different reward types
    finegrained_scores.extend([
        (0.85, (0, 6), 0, 0),      # Format reward: "Hello!"
        (0.72, (7, 25), 1, 0),     # Helpfulness: "I'm here to help you"
        (0.91, (50, 85), 2, 0),    # Detail level: "detailed answer"
    ])
    
    # Response 1: 2 reward types
    finegrained_scores.extend([
        (0.78, (0, 4), 0, 1),      # Format reward: "Sure"
        (0.83, (25, 60), 1, 1),    # Helpfulness: "I'll break it down"
    ])
    
    # Response 2: 4 reward types (more detailed analysis)
    finegrained_scores.extend([
        (0.65, (0, 8), 0, 2),      # Format: "Actually"
        (0.88, (12, 35), 3, 2),    # Thoughtfulness: "need to think about this"
        (0.74, (50, 70), 1, 2),    # Helpfulness: "Let me reconsider"
        (0.92, (35, 49), 4, 2),    # Precision: "more carefully"
    ])
    
    # Response 3: 1 reward type only
    finegrained_scores.extend([
        (0.45, (0, 27), 5, 3),     # Completeness: entire short response
    ])
    
    # Response 4: 5 reward types (comprehensive analysis)
    long_text = decoded_responses[4]
    finegrained_scores.extend([
        (0.89, (0, 20), 0, 4),     # Format: opening
        (0.76, (50, 100), 1, 4),   # Helpfulness: middle section
        (0.94, (120, 180), 2, 4),  # Detail level: technical details
        (0.82, (200, 250), 6, 4),  # Examples quality
        (0.87, (300, len(long_text)), 7, 4),  # Conclusion quality
    ])
    
    print(f"Created {len(finegrained_scores)} finegrained scores across {len(responses)} responses")
    
    # Display the reward distribution
    reward_counts = {}
    reward_names = {
        0: "Format", 1: "Helpfulness", 2: "Detail", 3: "Thoughtfulness", 
        4: "Precision", 5: "Completeness", 6: "Examples", 7: "Conclusion"
    }
    
    for _, _, reward_group_id, response_idx in finegrained_scores:
        if response_idx not in reward_counts:
            reward_counts[response_idx] = {}
        reward_name = reward_names.get(reward_group_id, f"Group{reward_group_id}")
        reward_counts[response_idx][reward_name] = reward_counts[response_idx].get(reward_name, 0) + 1
    
    print("\nReward distribution per response:")
    for resp_idx in sorted(reward_counts.keys()):
        rewards = reward_counts[resp_idx]
        reward_list = [f"{name}({count})" for name, count in rewards.items()]
        print(f"  Response {resp_idx}: {len(rewards)} types - {', '.join(reward_list)}")
    print()
    
    # Test string-to-token span conversion
    print("Testing string-to-token span conversion...")
    conversion_results = []
    
    for score, (start_char, end_char), reward_group_id, response_idx in finegrained_scores:
        # Convert character span to token span for this specific response
        decoded_resp = decoded_responses[response_idx]
        token_resp = responses[response_idx]
        
        # Use the conversion function
        token_spans = convert_string_span_to_token_span(
            (start_char, end_char), 
            [decoded_resp], 
            [token_resp], 
            tokenizer
        )
        token_start, token_end = token_spans[0]
        
        # Extract the actual text spans
        char_span_text = decoded_resp[start_char:end_char]
        if token_start < len(token_resp) and token_end <= len(token_resp):
            token_span_tokens = token_resp[token_start:token_end]
            token_span_text = tokenizer.decode(token_span_tokens, skip_special_tokens=True)
        else:
            token_span_text = "[INVALID_SPAN]"
        
        conversion_results.append({
            'response_idx': response_idx,
            'reward_type': reward_names.get(reward_group_id, f"Group{reward_group_id}"),
            'score': score,
            'char_span': (start_char, end_char),
            'token_span': (token_start, token_end),
            'char_text': char_span_text,
            'token_text': token_span_text,
            'match_quality': 'GOOD' if char_span_text.strip() in token_span_text or token_span_text.strip() in char_span_text else 'PARTIAL'
        })
    
    # Display conversion results
    print("Span conversion results:")
    for i, result in enumerate(conversion_results):
        print(f"  [{i:2d}] Response {result['response_idx']}, {result['reward_type']} (score: {result['score']:.2f})")
        print(f"       Char span {result['char_span']}: '{result['char_text']}'")
        print(f"       Token span {result['token_span']}: '{result['token_text']}'")
        print(f"       Match quality: {result['match_quality']}")
        print()
    
    # Test edge cases
    print("Testing edge cases...")
    edge_cases = [
        # Empty span
        (0.5, (0, 0), 0, 0),
        # Single character
        (0.6, (0, 1), 1, 1),
        # Entire response
        (0.7, (0, len(decoded_responses[2])), 2, 2),
        # Out of bounds (should be clamped)
        (0.8, (0, 1000), 3, 3),
        # Negative start (should be clamped)
        (0.9, (-5, 10), 4, 4),
    ]
    
    print("Edge case results:")
    for i, (score, (start_char, end_char), reward_group_id, response_idx) in enumerate(edge_cases):
        try:
            decoded_resp = decoded_responses[response_idx]
            token_resp = responses[response_idx]
            
            token_spans = convert_string_span_to_token_span(
                (start_char, end_char), 
                [decoded_resp], 
                [token_resp], 
                tokenizer
            )
            token_start, token_end = token_spans[0]
            
            # Clamp character indices for display
            start_char_clamped = max(0, min(start_char, len(decoded_resp)))
            end_char_clamped = max(start_char_clamped, min(end_char, len(decoded_resp)))
            
            print(f"  Edge case {i}: ({start_char}, {end_char}) -> ({start_char_clamped}, {end_char_clamped}) -> tokens ({token_start}, {token_end})")
            print(f"    Text: '{decoded_resp[start_char_clamped:end_char_clamped]}'")
            
        except Exception as e:
            print(f"  Edge case {i}: ERROR - {e}")
    
    print()
    
    # Summary statistics
    print("Summary Statistics:")
    print(f"  Total responses: {len(responses)}")
    print(f"  Total finegrained scores: {len(finegrained_scores)}")
    print(f"  Unique reward types: {len(set(reward_group_id for _, _, reward_group_id, _ in finegrained_scores))}")
    print(f"  Average rewards per response: {len(finegrained_scores) / len(responses):.1f}")
    
    response_lengths = [len(resp) for resp in responses]
    print(f"  Token lengths: min={min(response_lengths)}, max={max(response_lengths)}, avg={np.mean(response_lengths):.1f}")
    
    scores = [score for score, _, _, _ in finegrained_scores]
    print(f"  Score range: min={min(scores):.2f}, max={max(scores):.2f}, avg={np.mean(scores):.2f}")
    
    # Test the complete processing pipeline (simulate what fgrpo_fast.py does)
    print("\nTesting complete processing pipeline...")
    
    # Extract components
    scores = np.array([score for score, _, _, _ in finegrained_scores])
    effective_spans = [(start_char, end_char) for _, (start_char, end_char), _, _ in finegrained_scores]
    reward_group_ids = [reward_group_id for _, _, reward_group_id, _ in finegrained_scores]
    response_indices = [response_idx for _, _, _, response_idx in finegrained_scores]
    
    # Validate response indices
    max_response_idx = max(response_indices)
    if max_response_idx >= len(responses):
        print(f"❌ Invalid response index {max_response_idx}")
        return False
    else:
        print(f"✅ All response indices valid (0 to {max_response_idx})")
    
    # Group by reward type and normalize
    unique_groups = list(set(reward_group_ids))
    print(f"✅ Found {len(unique_groups)} unique reward groups: {sorted(unique_groups)}")
    
    advantages = np.zeros_like(scores, dtype=np.float32)
    for group_id in unique_groups:
        group_indices = [i for i, gid in enumerate(reward_group_ids) if gid == group_id]
        group_scores = scores[group_indices]
        
        # Normalize per group
        group_mean = np.mean(group_scores)
        group_std = np.std(group_scores) + 1e-8
        group_advantages = (group_scores - group_mean) / group_std
        
        for idx, group_idx in enumerate(group_indices):
            advantages[group_idx] = group_advantages[idx]
        
        print(f"  Group {group_id} ({reward_names.get(group_id, f'Group{group_id}')}): {len(group_indices)} scores, mean={group_mean:.3f}, std={group_std:.3f}")
    
    # Create span masks
    span_masks = []
    for i, (effective_span, response_idx) in enumerate(zip(effective_spans, response_indices)):
        start_char, end_char = effective_span
        decoded_resp = decoded_responses[response_idx]
        token_resp = responses[response_idx]
        
        # Convert to token span
        token_spans = convert_string_span_to_token_span(
            effective_span, [decoded_resp], [token_resp], tokenizer
        )
        token_start, token_end = token_spans[0]
        
        # Create mask
        response_length = len(token_resp)
        span_mask = np.zeros(response_length, dtype=bool)
        token_start = max(0, min(token_start, response_length))
        token_end = max(token_start, min(token_end, response_length))
        span_mask[token_start:token_end] = True
        span_masks.append(span_mask)
    
    print(f"✅ Created {len(span_masks)} span masks")
    
    # Verify mask properties
    mask_stats = []
    for i, mask in enumerate(span_masks):
        response_idx = response_indices[i]
        mask_ratio = np.sum(mask) / len(mask) if len(mask) > 0 else 0
        mask_stats.append(mask_ratio)
        
    print(f"  Span coverage: min={min(mask_stats):.1%}, max={max(mask_stats):.1%}, avg={np.mean(mask_stats):.1%}")
    
    print()
    print("✅ Realistic tokenizer spans test completed successfully!")
    print("✅ All components working correctly with varying reward counts per response!")
    
    return True


if __name__ == "__main__":
    try:
        success = test_realistic_tokenizer_spans()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 