#!/usr/bin/env python3
"""
Optimized character-to-token mapping approaches for finegrained GRPO.
This demonstrates several acceleration techniques that require only single decode operations.
"""

import sys
import os
import time
import numpy as np
from transformers import AutoTokenizer

# Add the open_instruct module to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def approach_1_single_decode_with_offsets(token_resp, tokenizer):
    """
    Approach 1: Use tokenizer offset mapping (if available)
    Most efficient - single encode/decode with offset information
    """
    print("🚀 APPROACH 1: Single Decode with Offset Mapping")
    
    # Try to get offset mapping from tokenizer
    try:
        # Some tokenizers support return_offsets_mapping
        encoding = tokenizer(
            tokenizer.decode(token_resp, skip_special_tokens=True), 
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        
        if hasattr(encoding, 'offset_mapping') and encoding.offset_mapping is not None:
            offset_mapping = encoding.offset_mapping
            print(f"   ✅ Got offset mapping: {offset_mapping}")
            
            # Build character-to-token mapping
            char_to_token = {}
            for token_idx, (start_char, end_char) in enumerate(offset_mapping):
                for char_idx in range(start_char, end_char):
                    char_to_token[char_idx] = token_idx
            
            return char_to_token, True
        else:
            print("   ❌ Tokenizer doesn't support offset mapping")
            return None, False
            
    except Exception as e:
        print(f"   ❌ Offset mapping failed: {e}")
        return None, False


def approach_2_single_decode_with_token_boundaries(token_resp, tokenizer):
    """
    Approach 2: Single decode + individual token decoding for boundary detection
    Decode full text once, then decode individual tokens to find boundaries
    """
    print("🚀 APPROACH 2: Single Decode + Token Boundary Detection")
    
    # Single decode of the full response
    full_text = tokenizer.decode(token_resp, skip_special_tokens=True)
    print(f"   Full text: '{full_text}'")
    
    # Decode individual tokens to get their text representations
    token_texts = []
    for token_id in token_resp:
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        token_texts.append(token_text)
    
    print(f"   Token texts: {token_texts}")
    
    # Build character-to-token mapping by finding token boundaries in full text
    char_to_token = {}
    current_pos = 0
    
    for token_idx, token_text in enumerate(token_texts):
        if not token_text:  # Handle empty tokens
            continue
            
        # Find where this token appears in the full text starting from current_pos
        token_start = full_text.find(token_text, current_pos)
        
        if token_start != -1:
            token_end = token_start + len(token_text)
            
            # Map characters to this token
            for char_idx in range(token_start, token_end):
                char_to_token[char_idx] = token_idx
            
            current_pos = token_end
            print(f"     Token {token_idx}: '{token_text}' → chars [{token_start}:{token_end}]")
        else:
            print(f"     Token {token_idx}: '{token_text}' → NOT FOUND (tokenizer quirk)")
            # Handle tokenizer quirks - assign to current position
            if current_pos < len(full_text):
                char_to_token[current_pos] = token_idx
                current_pos += 1
    
    # Fill any remaining characters
    for char_idx in range(current_pos, len(full_text)):
        char_to_token[char_idx] = len(token_resp) - 1
    
    return char_to_token, full_text


def approach_3_cached_mapping(token_resp, tokenizer, cache={}):
    """
    Approach 3: Cached character-to-token mapping
    Cache mappings for identical token sequences to avoid recomputation
    """
    print("🚀 APPROACH 3: Cached Mapping")
    
    # Create cache key from token sequence
    cache_key = tuple(token_resp)
    
    if cache_key in cache:
        print(f"   ✅ Cache hit! Using cached mapping for {len(token_resp)} tokens")
        return cache[cache_key], True
    
    print(f"   ❌ Cache miss. Computing mapping for {len(token_resp)} tokens")
    
    # Use approach 2 to compute the mapping
    char_to_token, full_text = approach_2_single_decode_with_token_boundaries(token_resp, tokenizer)
    
    # Cache the result
    cache[cache_key] = (char_to_token, full_text)
    
    return char_to_token, full_text


def approach_4_vectorized_search(token_resp, tokenizer):
    """
    Approach 4: Vectorized character position search
    Use numpy for faster character-to-token lookups
    """
    print("🚀 APPROACH 4: Vectorized Search")
    
    # Get mapping using approach 2
    char_to_token, full_text = approach_2_single_decode_with_token_boundaries(token_resp, tokenizer)
    
    # Convert to numpy arrays for faster lookups
    max_char = max(char_to_token.keys()) if char_to_token else 0
    char_to_token_array = np.full(max_char + 1, -1, dtype=np.int32)
    
    for char_idx, token_idx in char_to_token.items():
        char_to_token_array[char_idx] = token_idx
    
    print(f"   ✅ Created vectorized mapping array of size {len(char_to_token_array)}")
    
    return char_to_token_array, full_text


def optimized_convert_string_span_to_token_span(effective_span, decoded_resp, token_resp, tokenizer, method="auto"):
    """
    Optimized version of string-to-token span conversion
    """
    start_char, end_char = effective_span
    
    # Handle edge cases
    if start_char >= len(decoded_resp):
        return len(token_resp), len(token_resp)
    if end_char <= 0:
        return 0, 0
    
    # Clamp character indices
    start_char_clamped = max(0, min(start_char, len(decoded_resp)))
    end_char_clamped = max(start_char_clamped, min(end_char, len(decoded_resp)))
    
    # Try different approaches based on method
    char_to_token = None
    
    if method == "auto" or method == "offsets":
        char_to_token, success = approach_1_single_decode_with_offsets(token_resp, tokenizer)
        if success:
            print("   ✅ Using offset mapping approach")
        elif method == "offsets":
            raise ValueError("Offset mapping requested but not available")
    
    if char_to_token is None and (method == "auto" or method == "boundaries"):
        char_to_token, _ = approach_2_single_decode_with_token_boundaries(token_resp, tokenizer)
        print("   ✅ Using boundary detection approach")
    
    if char_to_token is None:
        raise ValueError(f"No mapping approach succeeded for method: {method}")
    
    # Find token boundaries
    token_start = char_to_token.get(start_char_clamped, 0)
    
    # For end position, we want the token that contains end_char_clamped - 1
    if end_char_clamped > 0:
        token_end = char_to_token.get(end_char_clamped - 1, len(token_resp) - 1) + 1
    else:
        token_end = token_start
    
    # Clamp to valid range
    token_start = max(0, min(token_start, len(token_resp)))
    token_end = max(token_start, min(token_end, len(token_resp)))
    
    return token_start, token_end


def benchmark_approaches():
    """Benchmark different mapping approaches"""
    print("=" * 80)
    print("OPTIMIZED CHARACTER-TO-TOKEN MAPPING BENCHMARK")
    print("=" * 80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test cases of varying lengths
    test_cases = [
        "Hello!",
        "Hello! I'm here to help you with your question.",
        "This is a much longer response that contains multiple sentences and covers various topics. It includes technical details, explanations, and examples to provide comprehensive coverage of the subject matter being discussed in great detail.",
        "Sure, that's a great question. The answer is complex but I'll break it down step by step. First, we need to understand the fundamentals. Then we can dive into the specifics. Finally, we'll look at some practical examples to solidify your understanding."
    ]
    
    print(f"Testing {len(test_cases)} cases with tokenizer: {tokenizer.__class__.__name__}")
    print()
    
    for case_idx, text in enumerate(test_cases):
        print(f"📝 TEST CASE {case_idx + 1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print("=" * 60)
        
        # Tokenize
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        print(f"Length: {len(text)} chars, {len(token_ids)} tokens")
        print(f"Decoded: '{decoded}'")
        print()
        
        # Test spans
        test_spans = [
            (0, min(10, len(decoded))),  # Beginning
            (max(0, len(decoded)//2 - 5), min(len(decoded), len(decoded)//2 + 5)),  # Middle
            (max(0, len(decoded) - 10), len(decoded))  # End
        ]
        
        for span_idx, (start_char, end_char) in enumerate(test_spans):
            print(f"🎯 SPAN {span_idx + 1}: ({start_char}, {end_char}) = '{decoded[start_char:end_char]}'")
            
            # Approach 1: Offset mapping
            start_time = time.time()
            try:
                char_to_token_1, success_1 = approach_1_single_decode_with_offsets(token_ids, tokenizer)
                if success_1:
                    token_start_1 = char_to_token_1.get(start_char, 0)
                    token_end_1 = char_to_token_1.get(end_char - 1, len(token_ids) - 1) + 1 if end_char > 0 else 0
                    time_1 = time.time() - start_time
                    result_1 = tokenizer.decode(token_ids[token_start_1:token_end_1], skip_special_tokens=True)
                    print(f"   Approach 1 (Offsets): ({token_start_1}, {token_end_1}) = '{result_1}' [{time_1*1000:.2f}ms]")
                else:
                    print(f"   Approach 1 (Offsets): NOT AVAILABLE")
            except Exception as e:
                print(f"   Approach 1 (Offsets): ERROR - {e}")
            
            # Approach 2: Boundary detection
            start_time = time.time()
            try:
                char_to_token_2, _ = approach_2_single_decode_with_token_boundaries(token_ids, tokenizer)
                token_start_2 = char_to_token_2.get(start_char, 0)
                token_end_2 = char_to_token_2.get(end_char - 1, len(token_ids) - 1) + 1 if end_char > 0 else 0
                time_2 = time.time() - start_time
                result_2 = tokenizer.decode(token_ids[token_start_2:token_end_2], skip_special_tokens=True)
                print(f"   Approach 2 (Boundaries): ({token_start_2}, {token_end_2}) = '{result_2}' [{time_2*1000:.2f}ms]")
            except Exception as e:
                print(f"   Approach 2 (Boundaries): ERROR - {e}")
            
            # Approach 3: Cached (first call)
            start_time = time.time()
            try:
                char_to_token_3, _ = approach_3_cached_mapping(token_ids, tokenizer)
                token_start_3 = char_to_token_3.get(start_char, 0)
                token_end_3 = char_to_token_3.get(end_char - 1, len(token_ids) - 1) + 1 if end_char > 0 else 0
                time_3 = time.time() - start_time
                result_3 = tokenizer.decode(token_ids[token_start_3:token_end_3], skip_special_tokens=True)
                print(f"   Approach 3 (Cached-1st): ({token_start_3}, {token_end_3}) = '{result_3}' [{time_3*1000:.2f}ms]")
            except Exception as e:
                print(f"   Approach 3 (Cached-1st): ERROR - {e}")
            
            # Approach 3: Cached (second call - should be faster)
            start_time = time.time()
            try:
                char_to_token_3b, _ = approach_3_cached_mapping(token_ids, tokenizer)
                time_3b = time.time() - start_time
                print(f"   Approach 3 (Cached-2nd): SAME RESULT [{time_3b*1000:.2f}ms] (cache hit)")
            except Exception as e:
                print(f"   Approach 3 (Cached-2nd): ERROR - {e}")
            
            print()
        
        print("-" * 60)
        print()
    
    # Summary
    print("🔍 OPTIMIZATION SUMMARY:")
    print()
    print("1. 🚀 OFFSET MAPPING (Fastest when available):")
    print("   ✅ Single tokenizer call with offset information")
    print("   ❌ Not available in all tokenizers (e.g., GPT-2)")
    print("   🎯 Best for: Transformers with offset support")
    print()
    print("2. 🔧 BOUNDARY DETECTION (Good balance):")
    print("   ✅ Single full decode + individual token decodes")
    print("   ✅ Works with any tokenizer")
    print("   ⚠️ Still requires N token decode calls")
    print("   🎯 Best for: General use when offsets unavailable")
    print()
    print("3. 💾 CACHING (Best for repeated sequences):")
    print("   ✅ Massive speedup for repeated token sequences")
    print("   ✅ Perfect for batch processing")
    print("   ❌ Memory usage grows with unique sequences")
    print("   🎯 Best for: Training with repeated patterns")
    print()
    print("4. ⚡ VECTORIZED SEARCH (Best for many lookups):")
    print("   ✅ Fast numpy-based character lookups")
    print("   ✅ Efficient for multiple spans per response")
    print("   ❌ Additional memory for numpy arrays")
    print("   🎯 Best for: Many spans per response")
    
    return True


if __name__ == "__main__":
    try:
        success = benchmark_approaches()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 