# Search Rewards Module

This module provides a comprehensive reward scoring system for evaluating responses to paper search queries. It uses a rubric-based approach with multiple scoring components to assess the quality of answers.

## Overview

The reward function evaluates responses based on several criteria including:
- **Length appropriateness** - Optimal response length (300-600 words)
- **Expertise level** - Alignment with expected user expertise
- **Citations and excerpts** - Proper citation usage and supporting quotes
- **Custom rubric properties** - Domain-specific evaluation criteria
- **Evidence matching** - Presence of required supporting information

## Input Format

### Response Format
The response must contain the answer wrapped in `<answer>` and `</answer>` tags:

```xml
<answer>
Your detailed answer here with proper citations and supporting evidence.
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
            "citations_weight": 0.2,              # Weight for citation scoring
            "excerpts_weight": 0.1,               # Weight for excerpt scoring
            "model_name": "gpt-4-turbo",          # LLM for scoring
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
    "scoring_results": {               # Detailed scoring breakdown
        "score": 0.85,                 # Overall weighted score
        "ann_score": 0.90,             # Annotation-only score (without static components)
        "length": 0.95,                # Length appropriateness score
        "expertise": 0.88,             # Expertise level score
        "citations": 0.80,             # Citation usage score
        "excerpts": 0.75,              # Excerpt quality score
        "criterion_name": 0.92,        # Custom criterion scores
        "criterion_name_evidence": 0.85 # Evidence matching scores
    },
    "error": None                      # Error message if any step failed
}
```

## How It Works

### 1. Answer Extraction
The function first extracts the answer content from between `<answer>` and `</answer>` tags using regex pattern matching.

### 2. Scoring Components

#### Length Scoring
- Scores response length relative to optimal range (300-600 words)
- Perfect score for responses within the range
- Penalty increases as length deviates from optimal range

#### Expertise Scoring
- Uses LLM to evaluate if response complexity matches expected user expertise
- Based on the criterion: "The level of expertise required to understand the answer should be roughly aligned with the estimated expertise of a typical person who would ask the question."

#### Citation and Excerpt Scoring
- **Citations**: Evaluates what fraction of claims have associated citations
- **Excerpts**: Evaluates what fraction of citations have supporting excerpts
- Uses LLM to parse response and identify claims, citations, and excerpts

#### Custom Rubric Properties
- Each property has a criterion description and optional evidence list
- LLM evaluates response against each criterion on a 0-10 scale
- Evidence matching: LLM checks what fraction of required evidence snippets are present

### 3. Weighted Scoring
All component scores are combined using configurable weights that must sum to 1.0:
- Static components: length, expertise, citations, excerpts
- Dynamic components: custom rubric properties and evidence matching

## Usage Examples

### Single Response Evaluation
```python
from paper_rewards import compute_paper_reward

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
            "citations_weight": 0.2,
            "excerpts_weight": 0.1,
            "other_properties": []
        }
    }
}

result = compute_paper_reward(response, test_case)
print(f"Reward: {result['reward']:.3f}")
```

### Batch Evaluation
```python
from paper_rewards import batch_compute_paper_rewards

responses = [response1, response2, response3]
test_cases = [test_case1, test_case2, test_case3]

results = batch_compute_paper_rewards(responses, test_cases)
for i, result in enumerate(results):
    print(f"Response {i+1}: {result['reward']:.3f}")
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

# Run utility tests
python tests/test_run_utils.py
```

## Test Structure

The test suite includes:

- **`test_paper_rewards.py`**: Tests the main reward computation function
  - `test_reward_computation()`: Tests normal reward calculation
  - `test_error_handling()`: Tests error scenarios

- **`test_run_utils.py`**: Tests utility functions
  - `test_extract_json_from_response()`: Tests JSON extraction
  - `test_run_chatopenai()`: Tests LLM API calls

- **`test_scoring.py`**: Tests individual scoring components
- **`test_answer.py`**: Tests answer extraction logic
- **`test_case.py`**: Tests test case validation

## Configuration

### Model Configuration
The scoring system uses LLMs for evaluation. Configure the model in the test case:
```python
"model_name": "gpt-4-turbo"  # or "gpt-3.5-turbo"
```

### Weight Configuration
Adjust scoring weights based on your evaluation priorities:
```python
"length_weight": 0.05,        # How much to weight response length
"expertise_weight": 0.05,     # How much to weight expertise matching
"citations_weight": 0.2,      # How much to weight citation usage
"excerpts_weight": 0.1,       # How much to weight excerpt quality
```

## Error Handling

The reward function handles various error scenarios:
- Missing `<answer>` tags in response
- Invalid test case format
- LLM API failures
- JSON parsing errors

All errors are captured in the `error` field of the result dictionary, and the function returns a default reward of 0.0 when errors occur.

## Dependencies

Key dependencies:
- `litellm`: For LLM API calls
- `pydantic`: For configuration validation
- `jsonlines`: For file I/O operations
- `pytest`: For testing framework

## Contributing

When adding new scoring components:
1. Add the component to the `RubricCorpusQaGenericMetric` class
2. Update the weight configuration in test cases
3. Add corresponding tests
4. Update this documentation
