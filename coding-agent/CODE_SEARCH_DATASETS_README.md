# Code Search Datasets Documentation

## Overview

This document describes the two HuggingFace datasets created for code search tasks:

1. **Multi-Step Tool Dataset** (`HF_OUTPUT_MULTI_STEP_TOOL`) - For models that use the CodeSearchTool with multiple interactions
2. **Single-Turn Dataset** - For models that output a single view call to find buggy code

## Dataset Creation

### Script: `create_code_search_datasets.py`

The main script that transforms raw coding-agent data into structured datasets.

**Usage:**
```bash
# Process all data files
python create_code_search_datasets.py \
    --data-dir coding-agent/data \
    --output-dir code_search_datasets

# Test with sample.json only
python create_code_search_datasets.py \
    --single-file-test \
    --output-dir test_datasets

# Push to HuggingFace Hub
python create_code_search_datasets.py \
    --push-to-hub \
    --hub-org your-org-name
```

## Dataset Formats

### 1. Multi-Step Tool Dataset

**Purpose:** Training models to use CodeSearchTool for exploring repositories with multiple tool calls.

**Schema:**
```python
{
    "instance_id": str,           # Unique identifier (e.g., "starlette_10457")
    "messages": List[Dict],       # Full conversation with system, user, assistant messages
    "buggy_info": Dict,          # Information about the bug location
    "num_turns": int,            # Total number of conversation turns
    "tool_calls_made": int       # Number of tool calls in the conversation
}
```

**Message Format:**
- Uses the full conversation history including tool responses
- Includes system prompts with tool descriptions
- Multiple assistant responses with tool calls

**Example Entry:**
```python
{
    "instance_id": "starlette_10457",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": "Find the bug in Config.__call__..."},
        {"role": "assistant", "content": "<tool_call>{...}</tool_call>"},
        {"role": "user", "content": "<tool_response>...</tool_response>"},
        ...
    ],
    "buggy_info": {
        "file_path": "/testbed/starlette/config.py",
        "buggy_line": 130,
        "view_range": [121, 138]
    },
    "num_turns": 32,
    "tool_calls_made": 15
}
```

### 2. Single-Turn Dataset

**Purpose:** Training models to directly identify and view the buggy file in a single response.

**Schema:**
```python
{
    "instance_id": str,          # Unique identifier
    "messages": List[Dict],      # Single-turn conversation (system, user, assistant)
    "buggy_file": str,          # Path to the file containing the bug
    "buggy_line": int,          # Line number of the bug (-1 if unknown)
    "bug_description": str       # Truncated description of the bug
}
```

**Message Format:**
- Simplified single-turn format
- System prompt explains the task
- User provides bug description
- Assistant responds with a single view tool call

**Example Entry:**
```python
{
    "instance_id": "starlette_10457",
    "messages": [
        {
            "role": "system", 
            "content": "You are a code search assistant..."
        },
        {
            "role": "user",
            "content": "Repository location: /testbed\n\nBug description:\n..."
        },
        {
            "role": "assistant",
            "content": "I'll examine the file...\n\n<tool_call>{...}</tool_call>"
        }
    ],
    "buggy_file": "/testbed/starlette/config.py",
    "buggy_line": 130,
    "bug_description": "Possible bug in Config.__call__..."
}
```

## Integration with Existing Components

### 1. CodeViewTool (`tool_vllm.py`)

The `CodeViewTool` class processes tool calls from model outputs:

```python
from open_instruct.tool_utils.tool_vllm import CodeViewTool

tool = CodeViewTool(
    api_endpoint="http://localhost:1234",
    repo_name="repository-name",
    start_str="<tool_call>",
    end_str="</tool_call>"
)

# Process model output containing tool calls
result = tool(model_output)
```

### 2. CodeSearchVerifier (`ground_truth_utils.py`)

The `CodeSearchVerifier` evaluates model predictions:

```python
from open_instruct.ground_truth_utils import CodeSearchVerifier, CodeVerifierConfig

config = CodeVerifierConfig(
    code_api_url="http://localhost:1234",
    code_max_execution_time=5.0,
    code_pass_rate_reward_threshold=0.5,
    code_apply_perf_penalty=True
)

verifier = CodeSearchVerifier(config)

# Evaluate prediction
result = verifier(
    tokenized_prediction=[],
    prediction=model_output,
    label=expected_files,
    query=original_query
)
```

### 3. View File API (`api.py`)

The API endpoint for viewing repository files:

```python
# POST /view_file
{
    "repo_name": "cool-RR/PySnooper",
    "path": "pysnooper/pycompat.py",
    "view_range": [86, 88],  # Optional
    "base_commit": "abc123"  # Optional
}

# Response
{
    "content": "file content here...",
    "repo_path": "/path/to/cloned/repo"
}
```

## Loading and Using the Datasets

```python
from datasets import load_from_disk

# Load datasets
multi_step = load_from_disk('code_search_datasets/multi_step_tool_dataset')
single_turn = load_from_disk('code_search_datasets/single_turn_dataset')

# Access samples
for sample in multi_step:
    print(f"Instance: {sample['instance_id']}")
    print(f"Tool calls: {sample['tool_calls_made']}")
    
for sample in single_turn:
    print(f"Bug file: {sample['buggy_file']}")
    print(f"Bug line: {sample['buggy_line']}")
```

## Training Considerations

### For Multi-Step Models:
- Use the full conversation history
- Train to generate appropriate tool calls based on context
- Handle tool responses and continue reasoning

### For Single-Turn Models:
- Focus on extracting key information from bug descriptions
- Generate precise file paths and view ranges
- Optimize for accuracy in first attempt

## Evaluation Metrics

The CodeSearchVerifier provides several metrics:
- **File Match Score**: Whether the correct file was viewed
- **Line Coverage**: Whether the buggy line was within the view range
- **Efficiency Penalty**: Penalty for viewing unnecessary files
- **Response Time**: Time taken to identify the bug

## ChatML Template Support

Both datasets support the ChatML template format implemented in `dataset_transformation.py`:

```python
from open_instruct.dataset_transformation import TokenizerConfig

tc = TokenizerConfig(
    tokenizer_name_or_path="your-model",
    chat_template_name="chatml"  # Use ChatML format
)
```

This formats messages as:
```
<|im_start|>system
You are a code search assistant.<|im_end|>
<|im_start|>user
Find the bug in this code.<|im_end|>
<|im_start|>assistant
I'll search for the bug.<|im_end|>
```

## Future Enhancements

1. **Multi-repository support**: Extend to handle bugs across multiple repositories
2. **Enhanced scoring**: More sophisticated evaluation metrics for partial matches
3. **Difficulty levels**: Categorize bugs by complexity
4. **Cross-file bugs**: Support bugs that span multiple files
5. **Test generation**: Include test cases that reproduce the bugs