#!/usr/bin/env python3
"""
Verification script for the code search datasets.
Shows how to load and use the datasets with the CodeSearchVerifier and CodeViewTool.
"""

import json
from pathlib import Path
from datasets import load_from_disk
from open_instruct.ground_truth_utils import CodeSearchVerifier, CodeVerifierConfig
from open_instruct.tool_utils.tool_vllm import CodeViewTool


def verify_multi_step_dataset():
    """Load and verify the multi-step tool dataset."""
    print("\n" + "="*60)
    print("Multi-Step Tool Dataset Verification")
    print("="*60)
    
    # Load the dataset
    dataset = load_from_disk("test_datasets/multi_step_tool_dataset")
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        
        print("\n--- Sample Entry ---")
        print(f"Instance ID: {sample['instance_id']}")
        print(f"Number of messages: {len(sample['messages'])}")
        print(f"Number of turns: {sample['num_turns']}")
        print(f"Tool calls made: {sample['tool_calls_made']}")
        
        # Show buggy info
        if sample['buggy_info']:
            print(f"\nBuggy Info:")
            print(f"  File: {sample['buggy_info'].get('file_path')}")
            print(f"  Line: {sample['buggy_info'].get('buggy_line')}")
            print(f"  View range: {sample['buggy_info'].get('view_range')}")
        
        # Show first few messages
        print("\nFirst 3 messages:")
        for i, msg in enumerate(sample['messages'][:3]):
            role = msg['role']
            content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
            print(f"\n[{i+1}] {role.upper()}:")
            print(content)
        
        # Test with CodeSearchVerifier
        print("\n--- Testing with CodeSearchVerifier ---")
        config = CodeVerifierConfig(
            code_api_url="http://localhost:1234",
            code_max_execution_time=5.0,
            code_pass_rate_reward_threshold=0.5,
            code_apply_perf_penalty=True
        )
        
        verifier = CodeSearchVerifier(config)
        
        # Extract assistant responses with tool calls
        assistant_responses = [msg['content'] for msg in sample['messages'] if msg['role'] == 'assistant']
        
        if assistant_responses:
            # Test with the first assistant response
            prediction = assistant_responses[0]
            label = sample['buggy_info']
            
            # Extract viewed files
            viewed_files = verifier.extract_viewed_files(prediction)
            print(f"Files viewed in first response: {len(viewed_files)}")
            for vf in viewed_files[:3]:  # Show first 3
                print(f"  - {vf['path']}")
                if vf.get('view_range'):
                    print(f"    Range: {vf['view_range']}")


def verify_single_turn_dataset():
    """Load and verify the single-turn dataset."""
    print("\n" + "="*60)
    print("Single-Turn Dataset Verification")
    print("="*60)
    
    # Load the dataset
    dataset = load_from_disk("test_datasets/single_turn_dataset")
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        
        print("\n--- Sample Entry ---")
        print(f"Instance ID: {sample['instance_id']}")
        print(f"Buggy file: {sample['buggy_file']}")
        print(f"Buggy line: {sample['buggy_line']}")
        print(f"Bug description: {sample['bug_description'][:100]}...")
        
        # Show the conversation
        print("\nConversation:")
        for msg in sample['messages']:
            role = msg['role']
            content = msg['content'][:300] + "..." if len(msg['content']) > 300 else msg['content']
            print(f"\n{role.upper()}:")
            print(content)
        
        # Test tool call extraction
        print("\n--- Extracted Tool Call ---")
        assistant_msg = sample['messages'][-1]['content']
        
        # Extract tool call
        import re
        tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        tool_calls = re.findall(tool_call_pattern, assistant_msg, re.DOTALL)
        
        if tool_calls:
            tool_call = json.loads(tool_calls[0])
            print(json.dumps(tool_call, indent=2))
            
            # Verify it matches the expected buggy file
            expected_file = sample['buggy_file']
            actual_file = tool_call['arguments'].get('path', '')
            
            print(f"\nExpected file: {expected_file}")
            print(f"Actual file: {actual_file}")
            print(f"Match: {expected_file == actual_file}")


def test_with_code_view_tool():
    """Test using the CodeViewTool with the dataset."""
    print("\n" + "="*60)
    print("Testing with CodeViewTool")
    print("="*60)
    
    # Initialize the tool (would need API endpoint running)
    tool = CodeViewTool(
        api_endpoint="http://localhost:1234",
        repo_name="starlette",
        start_str="<tool_call>",
        end_str="</tool_call>"
    )
    
    # Load single-turn dataset
    dataset = load_from_disk("test_datasets/single_turn_dataset")
    
    if len(dataset) > 0:
        sample = dataset[0]
        assistant_response = sample['messages'][-1]['content']
        
        print("Testing tool extraction from assistant response...")
        
        # Extract tool calls
        tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        tool_calls = re.findall(tool_call_pattern, assistant_response, re.DOTALL)
        
        if tool_calls:
            print(f"Found {len(tool_calls)} tool call(s)")
            tool_call = json.loads(tool_calls[0])
            print("\nExtracted tool call:")
            print(json.dumps(tool_call, indent=2))
            
            print("\nThis tool call would:")
            print(f"  - View file: {tool_call['arguments']['path']}")
            if 'view_range' in tool_call['arguments']:
                print(f"  - Lines: {tool_call['arguments']['view_range']}")
            
            print("\nNote: To actually execute, the API server needs to be running at http://localhost:1234")


def main():
    """Run all verification tests."""
    print("Code Search Dataset Verification")
    print("="*60)
    
    # Check if datasets exist
    if not Path("test_datasets/multi_step_tool_dataset").exists():
        print("ERROR: Datasets not found. Please run create_code_search_datasets.py first.")
        return
    
    # Run verifications
    verify_multi_step_dataset()
    verify_single_turn_dataset()
    test_with_code_view_tool()
    
    print("\n" + "="*60)
    print("Verification Complete!")
    print("="*60)
    
    print("\nThe datasets are ready for:")
    print("1. Training models with multi-step tool use (CodeSearchTool)")
    print("2. Training models for single-turn file viewing")
    print("3. Evaluation using CodeSearchVerifier")
    print("\nIntegration points:")
    print("- CodeViewTool in tool_vllm.py for execution")
    print("- CodeSearchVerifier in ground_truth_utils.py for evaluation")
    print("- view_file API endpoint in api.py for file access")


if __name__ == "__main__":
    import re  # Make sure re is imported at module level
    main()