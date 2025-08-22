#!/usr/bin/env python
"""Simple test of the parse_tool_calls fix without full initialization."""

import json
import re
import warnings

def parse_tool_calls(prediction: str):
    """
    Parse the tool calls from the prediction - copy of the fixed version.
    """
    tool_call_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    tool_calls = re.findall(tool_call_pattern, prediction, re.DOTALL)
    
    parsed_calls = []
    for tool_call in tool_calls:
        try:
            # Try to parse the JSON directly
            parsed = json.loads(tool_call)
            parsed_calls.append(parsed)
        except json.JSONDecodeError as e:
            # Log the error for debugging
            warnings.warn(f"Failed to parse tool call JSON: {e}\nRaw content: {tool_call[:200]}...")
            
            # Try to fix common JSON issues
            try:
                # Attempt to fix single quotes (common LLM output issue)
                fixed_json = tool_call.replace("'", '"')
                parsed = json.loads(fixed_json)
                parsed_calls.append(parsed)
            except:
                # If that doesn't work, try regex-based extraction for common format
                name_match = re.search(r'"name"\s*:\s*"([^"]+)"', tool_call)
                args_match = re.search(r'"arguments"\s*:\s*(\{[^}]*\})', tool_call)
                
                if name_match:
                    # Create a minimal valid tool call
                    parsed = {
                        "name": name_match.group(1),
                        "arguments": {}
                    }
                    if args_match:
                        try:
                            parsed["arguments"] = json.loads(args_match.group(1))
                        except:
                            pass  # Keep empty arguments if parsing fails
                    parsed_calls.append(parsed)
                # If we can't parse it at all, skip this tool call
    
    return parsed_calls

def test():
    """Test the parse_tool_calls function."""
    
    test_cases = [
        # Case that would cause "Expecting ',' delimiter" error
        '<tool_call>{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/path/with"quotes"problem.py", "view_range": [1, 100]}}</tool_call>',
        
        # Valid JSON
        '<tool_call>{"name": "code_view", "arguments": {"command": "view", "path": "/testbed/file.py"}}</tool_call>',
        
        # Single quotes
        "<tool_call>{'name': 'tool', 'arguments': {'key': 'value'}}</tool_call>",
        
        # Broken JSON but extractable name
        '<tool_call>{"name": "extractable", broken json here}</tool_call>',
    ]
    
    print("Testing parse_tool_calls fix...")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Input: {test[:80]}...")
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = parse_tool_calls(test)
            
            if w:
                print(f"  Warning issued: {w[0].message}")
            
            print(f"  Result: {len(result)} tool(s) parsed")
            if result:
                for j, tool in enumerate(result, 1):
                    print(f"    Tool {j}: name='{tool.get('name')}', args={tool.get('arguments')}")
    
    print("\n" + "=" * 60)
    print("✓ Test completed without crashing!")

if __name__ == "__main__":
    test()