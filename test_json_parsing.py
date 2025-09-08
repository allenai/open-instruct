#!/usr/bin/env python
"""Test script to reproduce and investigate JSON parsing error."""

import json
import re

def test_json_parsing():
    # Test cases of potentially malformed JSON that might cause the error
    test_cases = [
        # Normal case
        '{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/file.py"}}',
        
        # Missing quotes around property names
        '{name: "str_replace_editor", arguments: {command: "view", path: "/testbed/file.py"}}',
        
        # Single quotes instead of double quotes
        "{'name': 'str_replace_editor', 'arguments': {'command': 'view', 'path': '/testbed/file.py'}}",
        
        # Unescaped quotes in string values
        '{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/file"with"quotes.py"}}',
        
        # Missing closing bracket
        '{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/file.py"}',
        
        # Extra comma
        '{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/file.py",}}',
        
        # Newlines in JSON that might cause issues
        '{"name": "str_replace_editor",\n"arguments": {"command": "view",\n"path": "/testbed/file.py"}}',
        
        # Special characters that might need escaping
        '{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/file\\with\\backslash.py"}}',
    ]
    
    for i, test_json in enumerate(test_cases):
        print(f"\nTest case {i + 1}:")
        print(f"JSON: {test_json[:100]}...")  # Show first 100 chars
        try:
            result = json.loads(test_json)
            print(f"✓ Parsed successfully: {result.get('name', 'N/A')}")
        except json.JSONDecodeError as e:
            print(f"✗ JSON Error: {e}")
            print(f"  Position: line {e.lineno}, column {e.colno} (char {e.pos})")
            if e.pos and e.pos < len(test_json):
                # Show context around the error
                start = max(0, e.pos - 20)
                end = min(len(test_json), e.pos + 20)
                context = test_json[start:end]
                error_offset = e.pos - start
                print(f"  Context: ...{context}...")
                print(f"  Error at: {' ' * (10 + error_offset)}^")

def test_tool_call_extraction():
    """Test extraction of tool calls from text with <tool_call> tags."""
    
    # Simulate what might be in the prediction string
    test_predictions = [
        # Normal case
        '''Some text before
<tool_call>
{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/file.py"}}
</tool_call>
Some text after''',
        
        # Multiple tool calls
        '''Text
<tool_call>
{"name": "tool1", "arguments": {}}
</tool_call>
More text
<tool_call>
{"name": "tool2", "arguments": {}}
</tool_call>''',
        
        # Malformed JSON in tool call
        '''Text
<tool_call>
{name: "bad_json", arguments: {}}
</tool_call>''',
    ]
    
    tool_call_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    
    for i, prediction in enumerate(test_predictions):
        print(f"\n\nPrediction {i + 1}:")
        print("=" * 50)
        tool_calls = re.findall(tool_call_pattern, prediction, re.DOTALL)
        print(f"Found {len(tool_calls)} tool call(s)")
        
        for j, tool_call in enumerate(tool_calls):
            print(f"\n  Tool call {j + 1}:")
            print(f"  Raw: {tool_call[:100]}...")
            try:
                parsed = json.loads(tool_call)
                print(f"  ✓ Parsed successfully: {parsed.get('name', 'N/A')}")
            except json.JSONDecodeError as e:
                print(f"  ✗ JSON Error: {e}")
                print(f"    Position: line {e.lineno}, column {e.colno} (char {e.pos})")

if __name__ == "__main__":
    print("Testing JSON parsing scenarios...")
    print("=" * 60)
    test_json_parsing()
    
    print("\n\n" + "=" * 60)
    print("Testing tool call extraction...")
    print("=" * 60)
    test_tool_call_extraction()