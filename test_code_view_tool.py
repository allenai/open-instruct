#!/usr/bin/env python3
"""
Test script for CodeViewTool, similar to the SearchTool test.
Uses the first example from ft_hermes_search_swesmith_think_atk_ru_rc_SYSTEM_WITH_TOOL_FIND.jsonl
"""

import json
from open_instruct.tool_utils.tool_vllm import CodeViewTool

def main():
    # Initialize the CodeViewTool (you'll need to adjust the API endpoint)
    tool = CodeViewTool(
        start_str="<tool_call>",
        end_str="</tool_call>",
        api_endpoint="http://localhost:1234/view_file",  # Adjust this to your actual endpoint
        repo_name="starlette/starlette"
    )

    # Test case based on the first example from the JSONL file
    # This simulates the str_replace_editor tool call to view starlette/config.py
    test_prompt = '''<tool_call>
{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/starlette/config.py", "repo_name": "encode/starlette"}}
</tool_call>'''

    print("Testing CodeViewTool with starlette config.py example...")
    print(f"Test prompt: {test_prompt}")
    print("-" * 80)
    
    # Call the tool
    result = tool(test_prompt)
    
    # Display results
    print(f"Called: {result.called}")
    print(f"Runtime: {result.runtime:.2f}s")
    print(f"Error: {result.error}")
    print(f"Timeout: {result.timeout}")
    print("-" * 80)
    print("Output:")
    print(result.output)
    print("-" * 80)
    
    # Test case 2: View specific lines (Config.__call__ method)
    test_prompt_2 = '''<tool_call>
{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/starlette/config.py", "view_range": [84, 91], "repo_name": "encode/starlette"}}
</tool_call>'''
    
    print("\nTesting CodeViewTool with specific line range...")
    print(f"Test prompt: {test_prompt_2}")
    print("-" * 80)
    
    result_2 = tool(test_prompt_2)
    
    print(f"Called: {result_2.called}")
    print(f"Runtime: {result_2.runtime:.2f}s") 
    print(f"Error: {result_2.error}")
    print(f"Timeout: {result_2.timeout}")
    print("-" * 80)
    print("Output:")
    print(result_2.output)
    print("-" * 80)

    # Test case 3: Multiple tool calls in one prompt
    test_prompt_3 = '''<tool_call>
{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/starlette/config.py", "view_range": [84, 91], "repo_name": "encode/starlette"}}
</tool_call>

<tool_call>
{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/starlette/config.py", "view_range": [121, 138], "repo_name": "encode/starlette"}}
</tool_call>'''

    print("\nTesting CodeViewTool with multiple tool calls...")
    print(f"Test prompt: {test_prompt_3}")
    print("-" * 80)
    
    result_3 = tool(test_prompt_3)
    
    print(f"Called: {result_3.called}")
    print(f"Runtime: {result_3.runtime:.2f}s")
    print(f"Error: {result_3.error}")
    print(f"Timeout: {result_3.timeout}")
    print("-" * 80)
    print("Output:")
    print(result_3.output)

def test_format_citation_data_into_sqa_format():
    """
    Test function similar to the one in the search tool test.
    This would be used if CodeViewTool had a similar formatting function.
    """
    # For now, this is a placeholder since CodeViewTool doesn't have
    # a citation formatting function like SearchTool does
    print("CodeViewTool citation formatting test - placeholder")

if __name__ == "__main__":
    main()