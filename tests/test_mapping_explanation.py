#!/usr/bin/env python3
"""
Detailed explanation and visualization of character-to-token mapping in finegrained GRPO.
This shows exactly how the mapping works step by step.
"""

import sys
import os
from transformers import AutoTokenizer

# Add the open_instruct module to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mock_proportional_mapping(start_char, end_char, char_len, token_len):
    """
    Mock version: Simple proportional mapping
    Maps characters to tokens based on proportional position
    """
    print("🔸 MOCK APPROACH: Proportional Mapping")
    print(f"   Formula: token_pos = (char_pos / char_len) * token_len")
    
    if char_len == 0:
        return 0, 0
        
    token_start = int((start_char / char_len) * token_len)
    token_end = int((end_char / char_len) * token_len)
    
    # Clamp to valid range
    token_start = max(0, min(token_start, token_len))
    token_end = max(token_start, min(token_end, token_len))
    
    print(f"   start_char={start_char}, end_char={end_char}")
    print(f"   char_len={char_len}, token_len={token_len}")
    print(f"   token_start = int({start_char}/{char_len} * {token_len}) = {token_start}")
    print(f"   token_end = int({end_char}/{char_len} * {token_len}) = {token_end}")
    print(f"   Result: token span ({token_start}, {token_end})")
    
    return token_start, token_end


def real_incremental_mapping(start_char, end_char, decoded_resp, token_resp, tokenizer):
    """
    Real version: Incremental decoding to build precise character-to-token mapping
    """
    print("🔹 REAL APPROACH: Incremental Decoding")
    print("   Step-by-step token decoding to build char→token map")
    
    # Build character-to-token mapping
    char_to_token = {}
    cumulative_text = ""
    
    print(f"   Building char→token mapping:")
    for token_idx, token_id in enumerate(token_resp):
        try:
            # Decode up to this token
            decoded_prefix = tokenizer.decode(token_resp[:token_idx + 1], skip_special_tokens=True)
            
            # Map new characters to this token
            prev_len = len(cumulative_text)
            new_len = len(decoded_prefix)
            
            if new_len >= prev_len:
                for char_idx in range(prev_len, new_len):
                    if char_idx < len(decoded_resp):
                        char_to_token[char_idx] = token_idx
                
                # Show the mapping for this token
                new_chars = decoded_prefix[prev_len:new_len]
                char_range = f"[{prev_len}:{new_len}]" if new_len > prev_len else "[no new chars]"
                print(f"     Token {token_idx} (id={token_id}): '{new_chars}' → chars {char_range}")
            
            cumulative_text = decoded_prefix
            
        except Exception as e:
            print(f"     Token {token_idx} (id={token_id}): DECODE_ERROR → char {len(cumulative_text)}")
            if len(cumulative_text) < len(decoded_resp):
                char_to_token[len(cumulative_text)] = token_idx
            cumulative_text += f"[{token_id}]"
    
    # Fill in any remaining characters
    for char_idx in range(len(cumulative_text), len(decoded_resp)):
        char_to_token[char_idx] = len(token_resp) - 1
    
    print(f"   Final char→token mapping: {dict(sorted(char_to_token.items()))}")
    
    # Find token indices for the character span
    start_char_clamped = max(0, min(start_char, len(decoded_resp)))
    end_char_clamped = max(start_char_clamped, min(end_char, len(decoded_resp)))
    
    # Find token start
    if start_char_clamped in char_to_token:
        token_start = char_to_token[start_char_clamped]
    else:
        token_start = 0
        for char_idx in range(start_char_clamped, len(decoded_resp)):
            if char_idx in char_to_token:
                token_start = char_to_token[char_idx]
                break
    
    # Find token end (exclusive)
    if end_char_clamped > 0 and (end_char_clamped - 1) in char_to_token:
        token_end = char_to_token[end_char_clamped - 1] + 1
    else:
        token_end = len(token_resp)
        for char_idx in range(end_char_clamped - 1, -1, -1):
            if char_idx in char_to_token:
                token_end = char_to_token[char_idx] + 1
                break
    
    # Clamp to valid range
    token_start = max(0, min(token_start, len(token_resp)))
    token_end = max(token_start, min(token_end, len(token_resp)))
    
    print(f"   Char span ({start_char}, {end_char}) → clamped ({start_char_clamped}, {end_char_clamped})")
    print(f"   start_char {start_char_clamped} → token {token_start}")
    print(f"   end_char {end_char_clamped-1} → token {token_end-1} → end_token {token_end}")
    print(f"   Result: token span ({token_start}, {token_end})")
    
    return token_start, token_end, char_to_token


def demonstrate_mapping():
    """Demonstrate both mapping approaches with real examples"""
    print("=" * 80)
    print("CHARACTER-TO-TOKEN MAPPING EXPLANATION")
    print("=" * 80)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Loaded: {tokenizer.__class__.__name__}")
    print()
    
    # Test examples
    examples = [
        {
            "text": "Hello! I'm here to help you.",
            "spans": [(0, 6), (7, 11), (12, 16), (20, 24)]  # "Hello!", "I'm", "here", "help"
        },
        {
            "text": "Sure, that's great!",
            "spans": [(0, 4), (6, 12), (13, 19)]  # "Sure", "that's", "great!"
        }
    ]
    
    for ex_idx, example in enumerate(examples):
        print(f"📝 EXAMPLE {ex_idx + 1}: '{example['text']}'")
        print("=" * 60)
        
        # Tokenize
        token_ids = tokenizer.encode(example['text'], add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        print(f"Original text: '{example['text']}'")
        print(f"Decoded text:  '{decoded}'")
        print(f"Token IDs:     {token_ids}")
        print(f"Token count:   {len(token_ids)}")
        print()
        
        # Show individual tokens
        print("Individual tokens:")
        for i, token_id in enumerate(token_ids):
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            print(f"  Token {i}: {token_id} → '{token_text}'")
        print()
        
        # Test each span
        for span_idx, (start_char, end_char) in enumerate(example['spans']):
            print(f"🎯 SPAN {span_idx + 1}: characters ({start_char}, {end_char})")
            char_text = decoded[start_char:end_char]
            print(f"   Character span text: '{char_text}'")
            print()
            
            # Mock approach
            mock_start, mock_end = mock_proportional_mapping(
                start_char, end_char, len(decoded), len(token_ids)
            )
            if mock_start < len(token_ids) and mock_end <= len(token_ids):
                mock_tokens = token_ids[mock_start:mock_end]
                mock_text = tokenizer.decode(mock_tokens, skip_special_tokens=True) if mock_tokens else ""
                print(f"   Mock result text: '{mock_text}'")
            else:
                print(f"   Mock result text: [INVALID_SPAN]")
            print()
            
            # Real approach
            real_start, real_end, char_map = real_incremental_mapping(
                start_char, end_char, decoded, token_ids, tokenizer
            )
            if real_start < len(token_ids) and real_end <= len(token_ids):
                real_tokens = token_ids[real_start:real_end]
                real_text = tokenizer.decode(real_tokens, skip_special_tokens=True) if real_tokens else ""
                print(f"   Real result text: '{real_text}'")
            else:
                print(f"   Real result text: [INVALID_SPAN]")
            print()
            
            # Compare results
            accuracy_mock = "✅ EXACT" if char_text == mock_text else "⚠️ APPROXIMATE" if char_text in mock_text or mock_text in char_text else "❌ POOR"
            accuracy_real = "✅ EXACT" if char_text == real_text else "⚠️ APPROXIMATE" if char_text in real_text or real_text in char_text else "❌ POOR"
            
            print(f"   📊 COMPARISON:")
            print(f"      Target:    '{char_text}'")
            print(f"      Mock:      '{mock_text}' {accuracy_mock}")
            print(f"      Real:      '{real_text}' {accuracy_real}")
            print("   " + "-" * 50)
            print()
    
    # Explain the key differences
    print("🔍 KEY DIFFERENCES:")
    print()
    print("1. 🔸 MOCK APPROACH (Proportional):")
    print("   ✅ Pros: Fast, simple, no tokenizer calls")
    print("   ❌ Cons: Inaccurate, assumes uniform token distribution")
    print("   📝 Use case: Quick approximation, testing")
    print()
    print("2. 🔹 REAL APPROACH (Incremental Decoding):")
    print("   ✅ Pros: Precise, handles tokenizer quirks, exact mapping")
    print("   ❌ Cons: Slower, requires multiple tokenizer.decode() calls")
    print("   📝 Use case: Production training, precise gradient masking")
    print()
    print("🎯 WHY PRECISION MATTERS:")
    print("   In finegrained GRPO, gradients only flow through masked tokens.")
    print("   Wrong token boundaries = wrong gradients = poor training!")
    print("   The real approach ensures rewards apply to exactly the right tokens.")
    
    return True


if __name__ == "__main__":
    try:
        success = demonstrate_mapping()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 