# Tests

This directory contains test cases for the open-instruct project.

## Running Tests

### Prerequisites

Make sure you have the conda environment activated:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate open-instruct
```

### Test Files

- `test_fgrpo_advantages.py`: Tests the finegrained GRPO advantage computation implementation
- `test_fgrpo_gradients.py`: Tests the gradient flow and computation with finegrained advantages
- `test_fgrpo_realistic_spans.py`: Tests realistic finegrained reward scenarios with proper span masking
- `test_fgrpo_span_tuples.py`: Tests efficient span tuple format (start, end, total_length) for memory optimization
- `test_string_span_conversion.py`: Tests conversion from character-based spans to token-based spans
- `test_explicit_response_mapping.py`: Tests the new explicit response mapping format with 4-tuple scores
- `test_realistic_tokenizer_spans.py`: Tests with actual tokenizer and varying reward counts per response
- `test_mapping_explanation.py`: Detailed explanation and visualization of character-to-token mapping approaches
- `test_optimized_mapping.py`: Benchmark of different optimization strategies for character-to-token mapping
- `test_final_optimized_mapping.py`: Performance comparison showing the optimization benefits in production
- `test_llama_tokenizer_optimization.py`: Tests offset mapping support across different tokenizers (GPT-2, Llama, RoBERTa)

### Running Individual Tests

To run a specific test file:

```bash
# From the project root directory
python tests/test_fgrpo_advantages.py
```

### Running on GPU Environment

For tests that require GPU (e.g., for tensor operations), run on a GPU node:

```bash
ssh h100-219-140 "cd /checkpoint/comem/rulin/open-instruct && source ~/miniconda3/etc/profile.d/conda.sh && conda activate open-instruct && python tests/test_fgrpo_advantages.py"
```

## Test Structure

Each test file should:
- Be self-contained with mock data and functions
- Test the logic without requiring actual model training
- Include comprehensive output to verify correctness
- Follow the naming convention `test_*.py`

## Adding New Tests

When adding new tests:
1. Create the test file in this directory
2. Follow the existing patterns for imports and structure
3. Update this README with a description of the new test
4. Ensure tests can run both locally and on GPU environments 