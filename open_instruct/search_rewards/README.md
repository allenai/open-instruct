# Search Rewards Module

This module provides a comprehensive reward scoring system for evaluating responses to paper search queries. It uses a rubric-based approach with multiple scoring components to assess the quality of answers, with special emphasis on citation quality and accuracy.

## Overview

The reward function evaluates responses based on several criteria including:
- **Length appropriateness** - Optimal response length (300-600 words)
- **Expertise level** - Alignment with expected user expertise
- **Citation quality** - Advanced citation recall and precision scoring
- **Citation format** - Proper citation formatting and validation
- **Custom rubric properties** - Domain-specific evaluation criteria
- **Evidence matching** - Presence of required supporting information

## Key Features

### Advanced Citation Scoring
The module includes sophisticated citation evaluation capabilities:
- **Citation Recall**: Evaluates whether factual statements are properly supported by citations
- **Citation Precision**: Assesses the relevance of cited snippets to the claims made
- **Citation Format Validation**: Checks for proper citation formatting and prevents hallucinated citations
- **F1 Score Calculation**: Combines recall and precision for comprehensive citation quality assessment

### Flexible Response Format Support
- **Standard Format**: Responses wrapped in `<answer>` and `</answer>` tags
- **Citation-Enhanced Format**: Support for `<cite id="xxx">` tags within responses
- **Context-Aware Processing**: Automatic extraction of citations from context sections

## Input Format

### Response Format
The response can be in multiple formats:

#### Standard Format
```xml
<answer>
Your detailed answer here with proper citations and supporting evidence.
</answer>
```

#### Citation-Enhanced Format
```xml
<context>
<snippets id="a1b2c3d4">Citation content here</snippets>
<snippets id="e5f6g7h8">Another citation content</snippets>
</context>

<answer>
The Great Wall of China stretches approximately 13,000 miles across northern China. 
<cite id="a1b2c3d4">Construction of the Great Wall began during the 7th century BC under various warring states, but the most famous sections were built during the Ming Dynasty (1368-1644).</cite> 
The wall was primarily constructed as a defensive fortification. 
<cite id="e5f6g7h8">The wall incorporates various materials including stone, brick, tamped earth, wood, and other materials, with different sections built using locally available resources.</cite>
</answer>
```

### Test Case Format
Each test case is a dictionary with the following structure:

```python
{
    "metric_config": {
        "name": "rubric_corpusqa_generic",
        "config": {
            "question": "The question being answered",
            "low_length": 300,                    # Minimum optimal length
            "high_length": 600,                   # Maximum optimal length
            "length_weight": 0.05,                # Weight for length scoring
            "expertise_weight": 0.05,             # Weight for expertise scoring
            "citations_weight": 0.3,              # Weight for citation scoring (enhanced)
            "excerpts_weight": 0.2,               # Weight for excerpt scoring
            "model_name": "gpt-4.5-preview",      # LLM for scoring
            "other_properties": [                 # Custom rubric criteria
                {
                    "name": "criterion_name",
                    "criterion": "Description of what to evaluate",
                    "weight": 0.13333333333333333,
                    "evidence": [                 # Required supporting evidence
                        "Evidence snippet 1",
                        "Evidence snippet 2"
                    ]
                }
            ]
        }
    },
    "case_id": "unique_identifier",
    "annotator": "Annotator name",
    "agreement": True
}
```

## Output Format

The reward function returns a dictionary with the following structure:

```python
{
    "reward": 0.85,                    # Overall reward score (0-1)
    "answer_extracted": "extracted answer text",
    "extraction_success": True,        # Whether answer extraction succeeded
    "citations": {                     # Extracted citations from context
        "a1b2c3d4": "citation content",
        "e5f6g7h8": "another citation"
    },
    "scoring_results": {               # Detailed scoring breakdown
        "score": 0.85,                 # Overall weighted score
        "ann_score": 0.90,             # Annotation-only score (without static components)
        "length": 0.95,                # Length appropriateness score
        "expertise": 0.88,             # Expertise level score
        "citations": 0.92,             # Citation quality score (F1-based)
        "excerpts": 0.75,              # Excerpt quality score
        "criterion_name": 0.92,        # Custom criterion scores
        "criterion_name_evidence": 0.85 # Evidence matching scores
    },
    "error": None                      # Error message if any step failed
}
```

## How It Works

### 1. Answer and Citation Extraction
The function extracts both the answer content and citations:
- **Answer**: Content between `<answer>` and `</answer>` tags
- **Citations**: Content from `<snippets id="xxx">` tags in the context section
- **Cited Claims**: Claims wrapped in `<cite id="xxx">` tags within the answer

### 2. Scoring Components

#### Length Scoring
- Scores response length relative to optimal range (300-600 words)
- Perfect score for responses within the range
- Penalty increases as length deviates from optimal range

#### Expertise Scoring
- Uses LLM to evaluate if response complexity matches expected user expertise
- Based on the criterion: "The level of expertise required to understand the answer should be roughly aligned with the estimated expertise of a typical person who would ask the question."

#### Advanced Citation Scoring
- **Citation Recall**: Evaluates whether factual statements are supported by citations
  - For cited claims: Checks if the citation content supports the claim
  - For uncited claims: Determines if the claim needs citation
- **Citation Precision**: Assesses relevance of cited snippets to claims
- **Citation Format**: Validates citation IDs and prevents hallucinated citations
- **F1 Score**: Combines recall and precision for comprehensive assessment

#### Custom Rubric Properties
- Each property has a criterion description and optional evidence list
- LLM evaluates response against each criterion on a 0-10 scale
- Evidence matching: LLM checks what fraction of required evidence snippets are present

### 3. Weighted Scoring
All component scores are combined using configurable weights that must sum to 1.0:
- Static components: length, expertise, citations, excerpts
- Dynamic components: custom rubric properties and evidence matching

## Usage Examples

### Basic Response Evaluation
```python
from long_form_rewards import compute_paper_reward

response = """
<answer>
Type inference systems automatically deduce types for expressions in code.
This approach provides static typing benefits without requiring explicit annotations.
</answer>
"""

test_case = {
    "metric_config": {
        "config": {
            "question": "What is type inference?",
            "low_length": 300,
            "high_length": 600,
            "length_weight": 0.05,
            "expertise_weight": 0.05,
            "citations_weight": 0.3,
            "excerpts_weight": 0.2,
            "other_properties": []
        }
    }
}

result = compute_paper_reward(response, test_case)
print(f"Reward: {result['reward']:.3f}")
```

### Citation-Enhanced Response Evaluation
```python
from long_form_rewards import compute_paper_reward

response = """
<snippets id="a1b2c3d4">Type inference is a technique used in programming languages to automatically deduce the types of expressions without requiring explicit type annotations from the programmer.</snippets>

<answer>
Type inference systems automatically deduce types for expressions in code. 
<cite id="a1b2c3d4">Type inference is a technique used in programming languages to automatically deduce the types of expressions without requiring explicit type annotations from the programmer.</cite>
This approach provides static typing benefits without requiring explicit annotations.
</answer>
"""

test_case = {
    "metric_config": {
        "config": {
            "question": "What is type inference?",
            "citations_weight": 0.3,
            "excerpts_weight": 0.2,
            # ... other config
        }
    }
}

result = compute_paper_reward(response, test_case)
print(f"Citation Score: {result['scoring_results']['citations']:.3f}")
```

### Batch Evaluation
```python
from long_form_rewards import batch_compute_paper_rewards

responses = [response1, response2, response3]
test_cases = [test_case1, test_case2, test_case3]

results = batch_compute_paper_rewards(responses, test_cases)
for i, result in enumerate(results):
    print(f"Response {i+1}: {result['reward']:.3f}")
```

## Test Cases

### Type Inference Evaluation
The module includes comprehensive test cases for evaluating responses about type inference systems:

```python
test_case = {
    "metric_config": {
        "config": {
            "question": "What publicly available datasets are typically used for evaluating type inference systems in python?",
            "other_properties": [
                {
                    "name": "most_important_item_0",
                    "criterion": "Near the beginning, the answer should briefly define what is the goal of using a type inference system for programming languages in general.",
                    "weight": 0.13333333333333333,
                    "evidence": [
                        "Goal of type inference: Automatically deduce the most general type for each expression. Two key points: 1. Automatically inferring types: This means the programmer has to write no types, but still gets all the benefit from static typing 2. Inferring the most general type: This means we want to infer polymorphic types whenever possible"
                    ]
                },
                {
                    "name": "most_important_item_1",
                    "criterion": "The answer should emphasize on the importance of an automatic type inference system for Python.",
                    "weight": 0.13333333333333333,
                    "evidence": [
                        "its dynamic type system can lead to potential type errors, leading researchers to explore automatic type inference approaches for Python programs."
                    ]
                }
                # ... additional criteria
            ]
        }
    }
}
```

### Great Wall of China Example
A complete example demonstrating citation-enhanced responses:

```python
response = """
<snippets id="a1b2c3d4">The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states. While walls were built as early as the 7th century BC by various warring states, the most famous sections of the wall were constructed during the Ming Dynasty between 1368 and 1644.</snippets>

<answer>
The Great Wall of China, one of the most iconic structures in human history, stretches approximately 13,000 miles across northern China. 
<cite id="a1b2c3d4">Construction of the Great Wall began during the 7th century BC under various warring states, but the most famous sections were built during the Ming Dynasty (1368-1644).</cite>
</answer>
"""
```

## Running Tests

### Prerequisites
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
export AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
```

### Running All Tests
```bash
# From the search_rewards directory
python -m pytest tests/ -v
```

### Running Specific Test Files
```bash
# Test the main reward function
python -m pytest tests/test_paper_rewards.py -v

# Test citation-enhanced responses
python -m pytest test_formatted_example.py -v

# Test utility functions
python -m pytest tests/test_run_utils.py -v

# Test scoring components
python -m pytest tests/test_scoring.py -v
```

### Running Individual Test Functions
```bash
# Run specific test function
python -m pytest tests/test_paper_rewards.py::test_reward_computation -v

# Run with detailed output
python -m pytest tests/test_paper_rewards.py -v -s
```

### Manual Testing
You can also run the test files directly:
```bash
# Run the main test suite
python tests/test_paper_rewards.py

# Run citation-enhanced example
python test_formatted_example.py

# Run utility tests
python tests/test_run_utils.py
```

## Test Structure

The test suite includes:

- **`test_paper_rewards.py`**: Tests the main reward computation function
  - `test_reward_computation()`: Tests normal reward calculation
  - `test_error_handling()`: Tests error scenarios

- **`test_formatted_example.py`**: Tests citation-enhanced responses
  - `test_formatted_example()`: Tests the Great Wall of China example

- **`test_run_utils.py`**: Tests utility functions
  - `test_extract_json_from_response()`: Tests JSON extraction
  - `test_run_chatopenai()`: Tests LLM API calls
  - `test_run_azure_openai()`: Tests Azure OpenAI integration

- **`test_scoring.py`**: Tests individual scoring components
- **`test_answer.py`**: Tests answer extraction logic
- **`test_case.py`**: Tests test case validation

## Configuration

### Model Configuration
The scoring system supports multiple LLM providers:
```python
"model_name": "gpt-4.5-preview"  # OpenAI models
"model_name": "gpt-4-turbo"       # Alternative OpenAI models
```

### Weight Configuration
Adjust scoring weights based on your evaluation priorities:
```python
"length_weight": 0.05,        # How much to weight response length
"expertise_weight": 0.05,     # How much to weight expertise matching
"citations_weight": 0.3,      # How much to weight citation quality (enhanced)
"excerpts_weight": 0.2,       # How much to weight excerpt quality
```

### Citation Scoring Configuration
The citation scoring system uses advanced prompts for evaluation:
- **Recall Evaluation**: Determines if claims are properly supported
- **Precision Evaluation**: Assesses relevance of citations to claims
- **Format Validation**: Prevents hallucinated citations

## Error Handling

The reward function handles various error scenarios:
- Missing `<answer>` tags in response
- Invalid test case format
- LLM API failures
- JSON parsing errors
- Citation extraction failures
- Invalid citation IDs

All errors are captured in the `error` field of the result dictionary, and the function returns a default reward of 0.0 when errors occur.

## Dependencies

Key dependencies:
- `litellm`: For LLM API calls
- `pydantic`: For configuration validation
- `jsonlines`: For file I/O operations
- `pytest`: For testing framework
- `openai`: For Azure OpenAI integration

## Contributing

When adding new scoring components:
1. Add the component to the `RubricCorpusQaGenericMetric` class
2. Update the weight configuration in test cases
3. Add corresponding tests
4. Update this documentation

For citation-related improvements:
1. Modify the citation scoring functions in `citation_rewards_utils.py`
2. Update the prompts for recall and precision evaluation
3. Add test cases for new citation scenarios
4. Update the documentation with examples
