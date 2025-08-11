#!/usr/bin/env python
"""Test the fixed parse_tool_calls method."""

import json
import sys
import os

# Add the parent directory to the path to import open_instruct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from open_instruct.ground_truth_utils import CodeSearchVerifier, VerifierConfig

def test_parse_tool_calls():
    """Test the parse_tool_calls method with various inputs."""
    
    # Create a verifier instance
    config = VerifierConfig(is_async=False, num_workers=1, timeout=10)
    verifier = CodeSearchVerifier(config)
    
    test_cases = [
        # Normal valid JSON
        (
            'Text before <tool_call>{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/file.py"}}</tool_call> text after',
            1,
            "str_replace_editor"
        ),
        
        # Multiple tool calls
        (
            '<tool_call>{"name": "tool1", "arguments": {}}</tool_call> middle <tool_call>{"name": "tool2", "arguments": {"key": "value"}}</tool_call>',
            2,
            "tool1"
        ),
        
        # Single quotes (common LLM mistake)
        (
            "<tool_call>{'name': 'bad_tool', 'arguments': {'cmd': 'test'}}</tool_call>",
            1,
            "bad_tool"
        ),
        
        # Malformed JSON with unescaped quotes
        (
            '<tool_call>{"name": "tool", "arguments": {"path": "file"with"quotes"}}</tool_call>',
            1,  # Should still extract the name at least
            "tool"
        ),
        
        # Missing closing bracket
        (
            '<tool_call>{"name": "incomplete", "arguments": {"test": "value"}</tool_call>',
            1,
            "incomplete"
        ),
        
        # Empty tool call
        (
            '<tool_call></tool_call>',
            0,
            None
        ),
        
        # No tool calls
        (
            'Just some text without any tool calls',
            0,
            None
        ),
    ]
    
    print("Testing parse_tool_calls with various inputs...")
    print("=" * 60)
    
    all_passed = True
    for i, (prediction, expected_count, expected_first_name) in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Input: {prediction[:100]}...")
        
        try:
            result = verifier.parse_tool_calls(prediction)
            print(f"Result: Found {len(result)} tool call(s)")
            
            if len(result) != expected_count:
                print(f"  ✗ Expected {expected_count} tool calls, got {len(result)}")
                all_passed = False
            else:
                print(f"  ✓ Correct number of tool calls")
            
            if expected_first_name and len(result) > 0:
                first_name = result[0].get("name")
                if first_name == expected_first_name:
                    print(f"  ✓ First tool name matches: {first_name}")
                else:
                    print(f"  ✗ Expected first tool name '{expected_first_name}', got '{first_name}'")
                    all_passed = False
            
            if len(result) > 0:
                print(f"  Parsed tools: {[t.get('name') for t in result]}")
                
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    
    return all_passed

if __name__ == "__main__":
    success = test_parse_tool_calls()
    sys.exit(0 if success else 1)