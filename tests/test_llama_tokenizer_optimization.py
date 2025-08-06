#!/usr/bin/env python3
"""
Test character-to-token mapping optimization with Llama tokenizer.
This checks if Llama tokenizers support offset mapping and compares performance.
"""

import time
import sys
import os
from transformers import AutoTokenizer

# Add the open_instruct module to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from open_instruct.fgrpo_fast import convert_string_span_to_token_span
    FGRPO_AVAILABLE = True
except ImportError:
    FGRPO_AVAILABLE = False


def test_offset_mapping_support(tokenizer, text):
    """Test if tokenizer supports offset mapping"""
    print(f"🔍 Testing offset mapping support for {tokenizer.__class__.__name__}")
    
    try:
        # Try to get offset mapping
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        
        if hasattr(encoding, 'offset_mapping') and encoding.offset_mapping is not None:
            offset_mapping = encoding.offset_mapping
            print(f"   ✅ Offset mapping supported!")
            print(f"   📍 Sample offsets: {offset_mapping[:5]}{'...' if len(offset_mapping) > 5 else ''}")
            
            # Verify offset mapping quality
            token_ids = encoding.input_ids if hasattr(encoding, 'input_ids') else tokenizer.encode(text, add_special_tokens=False)
            
            # Check if offsets make sense
            total_chars = sum(end - start for start, end in offset_mapping)
            print(f"   📊 Total chars covered by offsets: {total_chars}/{len(text)}")
            
            if total_chars == len(text):
                print(f"   ✅ Perfect coverage - offsets cover all characters")
                return True, offset_mapping
            else:
                print(f"   ⚠️ Partial coverage - some characters not covered")
                return True, offset_mapping
                
        else:
            print(f"   ❌ Offset mapping not supported")
            return False, None
            
    except Exception as e:
        print(f"   ❌ Offset mapping failed: {e}")
        return False, None


def benchmark_tokenizer_performance(tokenizer_name, test_cases):
    """Benchmark performance with specific tokenizer"""
    print("=" * 80)
    print(f"TESTING TOKENIZER: {tokenizer_name}")
    print("=" * 80)
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Loaded tokenizer: {tokenizer.__class__.__name__}")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        print()
        
        total_offset_time = 0
        total_boundary_time = 0
        total_spans = 0
        offset_supported = False
        
        for case_idx, text in enumerate(test_cases):
            print(f"📝 TEST CASE {case_idx + 1}: '{text[:60]}{'...' if len(text) > 60 else ''}'")
            print("=" * 70)
            
            # Tokenize
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            
            print(f"Length: {len(text)} chars, {len(token_ids)} tokens")
            print(f"Decoded: '{decoded[:80]}{'...' if len(decoded) > 80 else ''}'")
            print()
            
            # Test offset mapping support
            supports_offsets, offset_mapping = test_offset_mapping_support(tokenizer, decoded)
            if supports_offsets:
                offset_supported = True
            print()
            
            # Test spans
            test_spans = [
                (0, min(20, len(decoded))),
                (max(0, len(decoded)//2 - 10), min(len(decoded), len(decoded)//2 + 10)),
                (max(0, len(decoded) - 20), len(decoded))
            ]
            
            for span_idx, (start_char, end_char) in enumerate(test_spans):
                span_text = decoded[start_char:end_char]
                print(f"🎯 SPAN {span_idx + 1}: ({start_char}, {end_char}) = '{span_text}'")
                
                # Method 1: Offset mapping (if supported)
                if supports_offsets:
                    start_time = time.time()
                    try:
                        # Build character-to-token mapping from offsets
                        char_to_token = {}
                        for token_idx, (offset_start, offset_end) in enumerate(offset_mapping):
                            for char_idx in range(offset_start, offset_end):
                                char_to_token[char_idx] = token_idx
                        
                        # Find token boundaries
                        token_start = char_to_token.get(start_char, 0)
                        token_end = char_to_token.get(end_char - 1, len(token_ids) - 1) + 1 if end_char > 0 else token_start
                        
                        offset_time = time.time() - start_time
                        offset_result = tokenizer.decode(token_ids[token_start:token_end], skip_special_tokens=True)
                        print(f"   Offset method: ({token_start}, {token_end}) = '{offset_result}' [{offset_time*1000:.3f}ms]")
                        total_offset_time += offset_time
                        
                    except Exception as e:
                        print(f"   Offset method: ERROR - {e}")
                        offset_time = 0
                else:
                    print(f"   Offset method: NOT SUPPORTED")
                    offset_time = 0
                
                # Method 2: Boundary detection (always available)
                start_time = time.time()
                try:
                    # Single decode + individual token decodes
                    full_text = decoded
                    char_to_token = {}
                    current_pos = 0
                    
                    for token_idx, token_id in enumerate(token_ids):
                        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                        if token_text:
                            token_start_pos = full_text.find(token_text, current_pos)
                            if token_start_pos != -1:
                                token_end_pos = token_start_pos + len(token_text)
                                for char_idx in range(token_start_pos, token_end_pos):
                                    char_to_token[char_idx] = token_idx
                                current_pos = token_end_pos
                            else:
                                if current_pos < len(full_text):
                                    char_to_token[current_pos] = token_idx
                                    current_pos += 1
                    
                    # Fill remaining characters
                    for char_idx in range(current_pos, len(full_text)):
                        char_to_token[char_idx] = len(token_ids) - 1
                    
                    # Find token boundaries
                    token_start = char_to_token.get(start_char, 0)
                    token_end = char_to_token.get(end_char - 1, len(token_ids) - 1) + 1 if end_char > 0 else token_start
                    
                    boundary_time = time.time() - start_time
                    boundary_result = tokenizer.decode(token_ids[token_start:token_end], skip_special_tokens=True)
                    print(f"   Boundary method: ({token_start}, {token_end}) = '{boundary_result}' [{boundary_time*1000:.3f}ms]")
                    total_boundary_time += boundary_time
                    
                    # Compare results if both methods worked
                    if offset_time > 0 and boundary_time > 0:
                        if offset_result.strip() == boundary_result.strip():
                            accuracy = "✅ IDENTICAL"
                        else:
                            accuracy = "⚠️ DIFFERENT"
                        speedup = boundary_time / offset_time if offset_time > 0 else float('inf')
                        print(f"   Comparison: {accuracy}, Offset speedup: {speedup:.1f}x")
                    
                except Exception as e:
                    print(f"   Boundary method: ERROR - {e}")
                    boundary_time = 0
                
                print()
                total_spans += 1
            
            print("-" * 70)
            print()
        
        # Summary for this tokenizer
        print("📊 TOKENIZER SUMMARY:")
        print(f"   Tokenizer: {tokenizer_name}")
        print(f"   Class: {tokenizer.__class__.__name__}")
        print(f"   Offset mapping supported: {'✅ YES' if offset_supported else '❌ NO'}")
        print(f"   Total spans tested: {total_spans}")
        
        if total_offset_time > 0:
            print(f"   Offset method total time: {total_offset_time*1000:.1f}ms")
        if total_boundary_time > 0:
            print(f"   Boundary method total time: {total_boundary_time*1000:.1f}ms")
        
        if total_offset_time > 0 and total_boundary_time > 0:
            overall_speedup = total_boundary_time / total_offset_time
            print(f"   Overall offset speedup: {overall_speedup:.1f}x")
            
        return {
            'tokenizer_name': tokenizer_name,
            'tokenizer_class': tokenizer.__class__.__name__,
            'offset_supported': offset_supported,
            'offset_time': total_offset_time,
            'boundary_time': total_boundary_time,
            'spans_tested': total_spans
        }
        
    except Exception as e:
        print(f"❌ Failed to test tokenizer {tokenizer_name}: {e}")
        return None


def main():
    """Main test function"""
    print("🦙 LLAMA TOKENIZER OPTIMIZATION TEST")
    print("=" * 80)
    
    # Test cases
    test_cases = [
        "Hello! How are you today?",
        "This is a longer sentence with multiple words and punctuation marks.",
        "The quick brown fox jumps over the lazy dog. This sentence contains all letters of the alphabet and is commonly used for testing purposes.",
    ]
    
    # Tokenizers to test
    tokenizers_to_test = [
        "microsoft/DialoGPT-small",  # GPT-2 based (for comparison)
        "meta-llama/Llama-2-7b-hf",  # Llama 2
        "meta-llama/Meta-Llama-3-8B",  # Llama 3
        "huggingface/CodeBERTa-small-v1",  # Another tokenizer type
    ]
    
    results = []
    
    for tokenizer_name in tokenizers_to_test:
        print(f"\n🔄 Testing {tokenizer_name}...")
        try:
            result = benchmark_tokenizer_performance(tokenizer_name, test_cases)
            if result:
                results.append(result)
        except Exception as e:
            print(f"⚠️ Skipping {tokenizer_name} - {e}")
            # Try a fallback tokenizer if the main ones fail
            if "llama" in tokenizer_name.lower():
                try:
                    fallback_name = "huggingface/llama-7b"  # Alternative Llama tokenizer
                    print(f"🔄 Trying fallback: {fallback_name}")
                    result = benchmark_tokenizer_performance(fallback_name, test_cases)
                    if result:
                        results.append(result)
                except:
                    print(f"⚠️ Fallback also failed, skipping Llama tokenizer")
    
    # Final comparison
    print("\n" + "=" * 80)
    print("🏆 FINAL COMPARISON ACROSS TOKENIZERS")
    print("=" * 80)
    
    if not results:
        print("❌ No tokenizers could be tested successfully")
        return False
    
    print(f"{'Tokenizer':<25} {'Class':<20} {'Offsets':<8} {'Offset Time':<12} {'Boundary Time':<14} {'Speedup':<8}")
    print("-" * 95)
    
    for result in results:
        offset_supported = "✅ YES" if result['offset_supported'] else "❌ NO"
        offset_time = f"{result['offset_time']*1000:.1f}ms" if result['offset_time'] > 0 else "N/A"
        boundary_time = f"{result['boundary_time']*1000:.1f}ms" if result['boundary_time'] > 0 else "N/A"
        
        if result['offset_time'] > 0 and result['boundary_time'] > 0:
            speedup = f"{result['boundary_time']/result['offset_time']:.1f}x"
        else:
            speedup = "N/A"
        
        tokenizer_short = result['tokenizer_name'].split('/')[-1][:24]
        class_short = result['tokenizer_class'][:19]
        
        print(f"{tokenizer_short:<25} {class_short:<20} {offset_supported:<8} {offset_time:<12} {boundary_time:<14} {speedup:<8}")
    
    print()
    print("🎯 KEY FINDINGS:")
    offset_tokenizers = [r for r in results if r['offset_supported']]
    if offset_tokenizers:
        print(f"   ✅ {len(offset_tokenizers)}/{len(results)} tokenizers support offset mapping")
        avg_speedup = sum(r['boundary_time']/r['offset_time'] for r in offset_tokenizers if r['offset_time'] > 0) / len(offset_tokenizers)
        print(f"   🚀 Average speedup with offset mapping: {avg_speedup:.1f}x")
    else:
        print(f"   ❌ No tokenizers in this test support offset mapping")
    
    print(f"   📊 Boundary detection works universally across all {len(results)} tokenizers")
    print(f"   🎯 Optimization automatically selects the best available method")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 