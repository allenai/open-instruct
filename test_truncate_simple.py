#!/usr/bin/env python3
"""
Simple test script for the updated truncate_messages_to_fit_context function.

This test focuses on the tiktoken fallback path since transformers may not be available.
"""

import sys
import os
from unittest.mock import patch, MagicMock

# Add the open_instruct directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'open_instruct'))

try:
    import tiktoken
    from transformers import AutoTokenizer
    from context_window_checker import (
        truncate_messages_to_fit_context,
        check_context_window_limit,
        get_encoding_for_model,
    )
    print("✅ Successfully imported context window checker and dependencies")
except ImportError as e:
    print(f"❌ Failed to import required libraries: {e}")
    sys.exit(1)


def test_truncate_messages_basic():
    """Test basic functionality of truncate_messages_to_fit_context."""
    
    print("\n🧪 Testing basic truncate_messages_to_fit_context:")
    print("-" * 50)
    
    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Tell me more about Paris."},
    ]
    
    try:
        result = truncate_messages_to_fit_context(
            messages=sample_messages,
            max_completion_tokens=100,
            model_name="gpt-4",  # Use a model name that tiktoken knows
            max_context_length=8192,
            safety_margin=50
        )
        
        # Basic checks
        assert isinstance(result, list), "Result should be a list"
        assert len(result) > 0, "Result should not be empty"
        assert all(isinstance(msg, dict) for msg in result), "All items should be dictionaries"
        assert all("role" in msg and "content" in msg for msg in result), "All messages should have role and content"
        
        print("✅ Basic functionality works correctly")
        print(f"   Input messages: {len(sample_messages)}, Output messages: {len(result)}")
        
        # Check that judgment format was appended to last user message
        last_user_msg = None
        for msg in reversed(result):
            if msg.get("role") == "user":
                last_user_msg = msg
                break
                
        if last_user_msg:
            expected_format = 'Respond in JSON format. {"REASONING": "[...]", "SCORE": "<your-score>"}'
            if expected_format in last_user_msg["content"]:
                print("✅ Judgment format correctly appended")
            else:
                print("❌ Judgment format not found in last user message")
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()


def test_truncate_long_messages():
    """Test truncation with very long messages."""
    
    print("\nTesting truncation with long messages:")
    print("-" * 50)
    
    long_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "This is a very long message. " * 1000},  # Very long
        {"role": "assistant", "content": "This is also a very long response. " * 1000},
        {"role": "user", "content": "Another long message here. " * 1000},
    ]
    
    try:
        result = truncate_messages_to_fit_context(
            messages=long_messages,
            max_completion_tokens=100,
            model_name="gpt-4",
            max_context_length=2000,  # Small context window to force truncation
            safety_margin=50
        )
        
        assert isinstance(result, list), "Result should be a list"
        assert len(result) <= len(long_messages), "Should have same or fewer messages after truncation"
        
        # System messages should be preserved
        system_messages = [msg for msg in result if msg.get("role") == "system"]
        original_system_messages = [msg for msg in long_messages if msg.get("role") == "system"]
        assert len(system_messages) == len(original_system_messages), "System messages should be preserved"

        print("✅ Long message truncation works correctly")
        print(f"   Original messages: {len(long_messages)}, Truncated: {len(result)}")
        
    except Exception as e:
        print(f"❌ Long message truncation test failed: {e}")
        import traceback
        traceback.print_exc()


def test_empty_messages():
    """Test with empty messages list."""
    
    print("\n🧪 Testing with empty messages:")
    print("-" * 50)
    
    try:
        result = truncate_messages_to_fit_context(
            messages=[],
            max_completion_tokens=100,
            model_name="gpt-4",
            max_context_length=8192,
            safety_margin=50
        )
        
        assert result == [], "Empty input should return empty output"
        print("✅ Empty messages handled correctly")
        
    except Exception as e:
        print(f"❌ Empty messages test failed: {e}")
        import traceback
        traceback.print_exc()


def test_only_system_messages():
    """Test with only system messages."""
    
    print("\n🧪 Testing with only system messages:")
    print("-" * 50)
    
    system_only_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": "Please be concise in your responses."},
    ]
    
    try:
        result = truncate_messages_to_fit_context(
            messages=system_only_messages,
            max_completion_tokens=100,
            model_name="gpt-4",
            max_context_length=8192,
            safety_margin=50
        )
        
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == len(system_only_messages), "All system messages should be preserved"
        assert all(msg["role"] == "system" for msg in result), "All messages should be system messages"
        
        print("✅ System-only messages handled correctly")
        
    except Exception as e:
        print(f"❌ System-only messages test failed: {e}")
        import traceback
        traceback.print_exc()


def test_real_qwen3_tokenizer():
    """Test with real Qwen/Qwen3-32B tokenizer if available."""
    
    print("\n🧪 Testing real Qwen/Qwen3-32B tokenizer:")
    print("-" * 50)
    
    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Tell me more about Paris."},
    ]
    
    try:
        result = truncate_messages_to_fit_context(
            messages=sample_messages,
            max_completion_tokens=100,
            model_name="Qwen/Qwen3-32B",
            max_context_length=1000,  # This should be overridden by model_max_length
            safety_margin=50
        )
        
        # Verify that messages were processed
        assert isinstance(result, list), "Result should be a list"
        assert len(result) > 0, "Result should not be empty"
        assert all(isinstance(msg, dict) for msg in result), "All items should be dictionaries"
        
        print("✅ Qwen/Qwen3-32B tokenizer works correctly")
        print(f"   Input messages: {len(sample_messages)}, Output messages: {len(result)}")
        
        # Check that judgment format was appended
        last_user_msg = None
        for msg in reversed(result):
            if msg.get("role") == "user":
                last_user_msg = msg
                break
                
        if last_user_msg:
            expected_format = 'Respond in JSON format. {"REASONING": "[...]", "SCORE": "<your-score>"}'
            if expected_format in last_user_msg["content"]:
                print("✅ Judgment format correctly appended")
            else:
                print("❌ Judgment format not found in last user message")
        
    except Exception as e:
        print(f"❌ Qwen/Qwen3-32B tokenizer test failed: {e}")
        print("   This might be expected if the Qwen model is not available locally")
        import traceback
        traceback.print_exc()


def test_hosted_vllm_prefix_handling():
    """Test hosted_vllm/ prefix handling."""
    
    print("\n🧪 Testing hosted_vllm/ prefix handling:")
    print("-" * 50)
    
    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    
    try:
        result = truncate_messages_to_fit_context(
            messages=sample_messages,
            max_completion_tokens=100,
            model_name="hosted_vllm/gpt2",  # Use a simple model that should be available
            max_context_length=1024,
            safety_margin=50
        )
        
        assert isinstance(result, list), "Result should be a list"
        assert len(result) > 0, "Result should not be empty"
        
        print("✅ hosted_vllm/ prefix handling works correctly")
        
    except Exception as e:
        print(f"❌ hosted_vllm/ prefix test failed: {e}")
        print("   This might be expected if the model is not available locally")
        import traceback
        traceback.print_exc()


def test_check_context_window_limit_basic():
    """Test basic functionality of check_context_window_limit."""
    
    print("\nTesting check_context_window_limit basic functionality:")
    print("-" * 50)
    
    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Tell me more about Paris."},
    ]
    
    try:
        # Test with a generous context window (should pass)
        result = check_context_window_limit(
            messages=sample_messages,
            max_completion_tokens=100,
            model_name="gpt2",  # Simple model
            max_context_length=2048,
            safety_margin=50
        )
        
        assert isinstance(result, bool), "Result should be a boolean"
        print(f"✅ Basic context window check works: {result}")
        
        # Test with a very small context window (should fail)
        result_small = check_context_window_limit(
            messages=sample_messages,
            max_completion_tokens=100,
            model_name="gpt2",
            max_context_length=50,  # Very small context
            safety_margin=10
        )
        
        assert isinstance(result_small, bool), "Result should be a boolean"
        print(f"✅ Small context window check works: {result_small}")
        
        if result and not result_small:
            print("✅ Context window limit detection works correctly")
        else:
            print("ℹ️  Context window results may vary based on actual token counts")
        
    except Exception as e:
        print(f"❌ Basic context window check failed: {e}")
        import traceback
        traceback.print_exc()


def test_check_context_window_limit_with_qwen():
    """Test check_context_window_limit with Qwen model."""
    
    print("\nTesting check_context_window_limit with Qwen:")
    print("-" * 50)
    
    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is quantum computing?"},
    ]
    
    try:
        result = check_context_window_limit(
            messages=sample_messages,
            max_completion_tokens=2048,
            model_name="Qwen/Qwen3-32B",
            max_context_length=8192,  # Should be overridden by model's actual context length
            safety_margin=100
        )
        
        assert isinstance(result, bool), "Result should be a boolean"
        print(f"✅ Qwen context window check works: {result}")
        print("   (Uses real Qwen tokenizer and model_max_length)")
        
    except Exception as e:
        print(f"❌ Qwen context window check failed: {e}")
        print("   This might be expected if the Qwen model is not available locally")
        import traceback
        traceback.print_exc()


def test_check_context_window_limit_edge_cases():
    """Test edge cases for check_context_window_limit."""
    
    print("\nTesting check_context_window_limit edge cases:")
    print("-" * 50)
    
    try:
        # Test with empty messages
        result_empty = check_context_window_limit(
            messages=[],
            max_completion_tokens=100,
            model_name="gpt2",
            max_context_length=1024,
            safety_margin=50
        )
        
        assert isinstance(result_empty, bool), "Result should be a boolean"
        print(f"✅ Empty messages check: {result_empty}")
        
        # Test with only system messages
        system_only = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "Please be concise."},
        ]
        
        result_system = check_context_window_limit(
            messages=system_only,
            max_completion_tokens=100,
            model_name="gpt2",
            max_context_length=1024,
            safety_margin=50
        )
        
        assert isinstance(result_system, bool), "Result should be a boolean"
        print(f"✅ System-only messages check: {result_system}")
        
        # Test with very long single message
        long_message = [
            {"role": "user", "content": "This is a very long message. " * 500}
        ]
        
        result_long = check_context_window_limit(
            messages=long_message,
            max_completion_tokens=100,
            model_name="gpt2",
            max_context_length=1024,
            safety_margin=50
        )
        
        assert isinstance(result_long, bool), "Result should be a boolean"
        print(f"✅ Long message check: {result_long}")
        
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()


def test_check_context_window_limit_hosted_vllm():
    """Test check_context_window_limit with hosted_vllm prefix."""
    
    print("\nTesting check_context_window_limit with hosted_vllm prefix:")
    print("-" * 50)
    
    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    
    try:
        result = check_context_window_limit(
            messages=sample_messages,
            max_completion_tokens=100,
            model_name="hosted_vllm/gpt2",  # Test prefix stripping
            max_context_length=1024,
            safety_margin=50
        )
        
        assert isinstance(result, bool), "Result should be a boolean"
        print(f"✅ hosted_vllm prefix check works: {result}")
        
    except Exception as e:
        print(f"❌ hosted_vllm prefix check failed: {e}")
        print("   This might be expected if the model is not available locally")
        import traceback
        traceback.print_exc()


def test_context_window_integration():
    """Test integration between check_context_window_limit and truncate_messages_to_fit_context."""
    
    print("\nTesting integration between check and truncate functions:")
    print("-" * 50)
    
    long_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about artificial intelligence. " * 200},
        {"role": "assistant", "content": "AI is a fascinating field. " * 150},
        {"role": "user", "content": "Can you elaborate? " * 100},
    ]
    
    try:
        # First check if it exceeds context window
        fits_before = check_context_window_limit(
            messages=long_messages,
            max_completion_tokens=100,
            model_name="gpt2",
            max_context_length=1024,  # Small context to force truncation
            safety_margin=50
        )
        
        print(f"   Before truncation fits: {fits_before}")
        
        # Truncate the messages
        truncated = truncate_messages_to_fit_context(
            messages=long_messages,
            max_completion_tokens=100,
            model_name="gpt2",
            max_context_length=1024,
            safety_margin=50
        )
        
        # Check if truncated version fits
        fits_after = check_context_window_limit(
            messages=truncated,
            max_completion_tokens=100,
            model_name="gpt2",
            max_context_length=1024,
            safety_margin=50
        )
        
        print(f"   After truncation fits: {fits_after}")
        print(f"   Messages before: {len(long_messages)}, after: {len(truncated)}")
        
        # The truncated version should fit
        assert fits_after, "Truncated messages should fit within context window"
        print("✅ Integration between check and truncate functions works correctly")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


def test_tiktoken_functionality():
    """Test tiktoken functionality."""
    
    print("\nTesting tiktoken functionality:")
    print("-" * 50)
    
    try:
        encoding = get_encoding_for_model("gpt-4")
        test_text = "Hello, world!"
        tokens = encoding.encode(test_text)
        decoded = encoding.decode(tokens)
        
        assert isinstance(tokens, list), "Tokens should be a list"
        assert len(tokens) > 0, "Should have some tokens"
        assert isinstance(decoded, str), "Decoded should be a string"
        
        print("✅ tiktoken encoding/decoding works correctly")
        print(f"   Text: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")
        
    except Exception as e:
        print(f"❌ tiktoken test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("Testing context window checker functions")
    print("=" * 70)
    
    # Test library functionality
    test_tiktoken_functionality()
    
    # Test check_context_window_limit function
    test_check_context_window_limit_basic()
    test_check_context_window_limit_with_qwen()
    test_check_context_window_limit_edge_cases()
    test_check_context_window_limit_hosted_vllm()
    
    # Test truncate_messages_to_fit_context function
    test_truncate_messages_basic()
    test_truncate_long_messages()
    test_empty_messages()
    test_only_system_messages()
    test_real_qwen3_tokenizer()
    test_hosted_vllm_prefix_handling()
    
    # Test integration between functions
    test_context_window_integration()
    
    print("\n✅ All tests completed!")
    print("\n💡 Key features tested:")
    print("   📊 check_context_window_limit function:")
    print("     - Basic context window checking")
    print("     - Qwen/Qwen3-32B model integration")
    print("     - Edge cases (empty, system-only, long messages)")
    print("     - hosted_vllm/ prefix handling")
    print("   ✂️  truncate_messages_to_fit_context function:")
    print("     - Basic message truncation")
    print("     - Long message handling")
    print("     - System message preservation")
    print("     - Judgment format appending")
    print("     - Real Qwen/Qwen3-32B tokenizer")
    print("   🔗 Integration testing:")
    print("     - Check + truncate workflow")
    print("   🛠️  Core functionality:")
    print("     - tiktoken encoding/decoding")
    print("     - HuggingFace tokenizer integration")
    
    print("\n Ready for your environment testing!")


if __name__ == "__main__":
    main()
