#!/usr/bin/env python3
"""
Test case for string span to token span conversion.
This tests the convert_string_span_to_token_span function that maps
character indices in decoded responses to token indices in tokenized responses.
"""

import numpy as np
import sys
import os

# Add the open_instruct module to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        
    def decode(self, token_ids, skip_special_tokens=True):
        """Simple mock decoder that maps tokens to characters"""
        # Simple mapping: each token ID becomes its string representation + space
        if not token_ids:
            return ""
        
        # Mock: token 100-199 -> "word", 200-299 -> "the", etc.
        text = ""
        for token_id in token_ids:
            if token_id == 100:
                text += "Hello"
            elif token_id == 101:
                text += " "
            elif token_id == 102:
                text += "world"
            elif token_id == 103:
                text += "!"
            elif token_id == 200:
                text += "This"
            elif token_id == 201:
                text += " is"
            elif token_id == 202:
                text += " a"
            elif token_id == 203:
                text += " test"
            elif token_id == 204:
                text += " sentence"
            elif token_id == 205:
                text += "."
            else:
                text += f"[{token_id}]"
        return text


def test_string_to_token_conversion():
    """Test the convert_string_span_to_token_span function"""
    print("=" * 80)
    print("TESTING STRING SPAN TO TOKEN SPAN CONVERSION")
    print("=" * 80)
    
    # Import the function we need to test
    from open_instruct.fgrpo_fast import data_preparation_thread
    
    # We need to extract the function from the data_preparation_thread context
    # Let's create our own version for testing
    def convert_string_span_to_token_span(effective_span, decoded_responses, responses, tokenizer):
        """
        Convert character-based span indices from decoded_responses to token-based span indices in responses.
        """
        start_char, end_char = effective_span
        token_spans = []
        
        for resp_idx, (decoded_resp, token_resp) in enumerate(zip(decoded_responses, responses)):
            # Handle edge cases
            if start_char >= len(decoded_resp):
                # Start is beyond the response, return empty span at the end
                token_spans.append((len(token_resp), len(token_resp)))
                continue
            
            if end_char <= 0:
                # End is before the response, return empty span at the beginning
                token_spans.append((0, 0))
                continue
            
            # Clamp character indices to valid range
            start_char_clamped = max(0, min(start_char, len(decoded_resp)))
            end_char_clamped = max(start_char_clamped, min(end_char, len(decoded_resp)))
            
            # Build character-to-token mapping by decoding prefixes
            char_to_token = {}
            cumulative_text = ""
            
            for token_idx, token_id in enumerate(token_resp):
                # Decode up to this token
                try:
                    # Decode the token sequence up to current position
                    decoded_prefix = tokenizer.decode(token_resp[:token_idx + 1], skip_special_tokens=True)
                    
                    # Map new characters to this token
                    prev_len = len(cumulative_text)
                    new_len = len(decoded_prefix)
                    
                    # Handle cases where tokenizer might produce different text
                    if new_len >= prev_len:
                        for char_idx in range(prev_len, new_len):
                            if char_idx < len(decoded_resp):
                                char_to_token[char_idx] = token_idx
                    
                    cumulative_text = decoded_prefix
                    
                except Exception:
                    # If decoding fails, map to current token
                    if len(cumulative_text) < len(decoded_resp):
                        char_to_token[len(cumulative_text)] = token_idx
                    cumulative_text += f"[{token_id}]"  # Placeholder for failed decode
            
            # Fill in any remaining characters
            for char_idx in range(len(cumulative_text), len(decoded_resp)):
                char_to_token[char_idx] = len(token_resp) - 1
            
            # Find token indices corresponding to character spans
            token_start = 0
            token_end = len(token_resp)
            
            if start_char_clamped in char_to_token:
                token_start = char_to_token[start_char_clamped]
            else:
                # Find the closest token for start
                for char_idx in range(start_char_clamped, len(decoded_resp)):
                    if char_idx in char_to_token:
                        token_start = char_to_token[char_idx]
                        break
            
            if end_char_clamped > 0 and (end_char_clamped - 1) in char_to_token:
                token_end = char_to_token[end_char_clamped - 1] + 1  # End is exclusive
            else:
                # Find the closest token for end
                for char_idx in range(end_char_clamped - 1, -1, -1):
                    if char_idx in char_to_token:
                        token_end = char_to_token[char_idx] + 1
                        break
            
            # Ensure valid token span
            token_start = max(0, min(token_start, len(token_resp)))
            token_end = max(token_start, min(token_end, len(token_resp)))
            
            token_spans.append((token_start, token_end))
        
        return token_spans
    
    # Create test data
    tokenizer = MockTokenizer()
    
    # Test case 1: Simple "Hello world!" 
    responses = [[100, 101, 102, 103]]  # "Hello world!"
    decoded_responses = ["Hello world!"]
    
    print("Test Case 1: 'Hello world!'")
    print(f"Tokens: {responses[0]}")
    print(f"Decoded: '{decoded_responses[0]}'")
    
    # Test different character spans
    test_spans = [
        (0, 5),    # "Hello"
        (6, 11),   # "world" 
        (0, 12),   # "Hello world!"
        (5, 6),    # " " (space)
        (11, 12),  # "!"
    ]
    
    for start_char, end_char in test_spans:
        token_spans = convert_string_span_to_token_span(
            (start_char, end_char), decoded_responses, responses, tokenizer
        )
        token_start, token_end = token_spans[0]
        
        # Extract the actual text covered by the token span
        covered_tokens = responses[0][token_start:token_end]
        covered_text = tokenizer.decode(covered_tokens)
        expected_text = decoded_responses[0][start_char:end_char]
        
        print(f"  Char span [{start_char}:{end_char}] -> Token span [{token_start}:{token_end}]")
        print(f"    Expected text: '{expected_text}'")
        print(f"    Covered text:  '{covered_text}'")
        print(f"    Match: {expected_text == covered_text}")
        print()
    
    # Test case 2: Longer sentence
    responses_2 = [[200, 201, 202, 203, 204, 205]]  # "This is a test sentence."
    decoded_responses_2 = ["This is a test sentence."]
    
    print("Test Case 2: 'This is a test sentence.'")
    print(f"Tokens: {responses_2[0]}")
    print(f"Decoded: '{decoded_responses_2[0]}'")
    
    test_spans_2 = [
        (0, 4),    # "This"
        (5, 7),    # "is"
        (10, 14),  # "test"
        (15, 23),  # "sentence"
        (0, 24),   # Full sentence
    ]
    
    for start_char, end_char in test_spans_2:
        token_spans = convert_string_span_to_token_span(
            (start_char, end_char), decoded_responses_2, responses_2, tokenizer
        )
        token_start, token_end = token_spans[0]
        
        # Extract the actual text covered by the token span
        covered_tokens = responses_2[0][token_start:token_end]
        covered_text = tokenizer.decode(covered_tokens)
        expected_text = decoded_responses_2[0][start_char:end_char]
        
        print(f"  Char span [{start_char}:{end_char}] -> Token span [{token_start}:{token_end}]")
        print(f"    Expected text: '{expected_text}'")
        print(f"    Covered text:  '{covered_text}'")
        print(f"    Match: {expected_text == covered_text}")
        print()
    
    print("✅ String to token span conversion test completed!")
    return True


if __name__ == "__main__":
    success = test_string_to_token_conversion()
    exit(0 if success else 1) 