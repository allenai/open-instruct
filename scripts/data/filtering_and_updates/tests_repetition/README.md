# N-gram Repetition Filter Tests

This directory contains test scripts for the `filter_ngram_repetitions.py` script. These tests verify that the exact block repetition detection works correctly for various patterns.

## Test Files

### `test_exact_repetition.py`
Tests the core functionality with the original examples provided:
- **Scooby-Doo repeated lines**: Multiple identical paragraphs
- **Marketing URL conversation**: Repeated conversation exchanges  
- **Normal text**: Ensures no false positives on regular content

### `test_2x_repetitions.py`
Tests that the detection works with minimal (2x) repetitions:
- **2x paragraph repetition**: Exactly 2 repeated paragraphs
- **2x consecutive line repetition**: Exactly 2 consecutive repeated lines
- **Large conversation chunks**: Large blocks repeated exactly 2 times
- **Threshold testing**: Verifies that higher thresholds don't catch 2x repetitions

### `test_consecutive.py`
Tests consecutive vs non-consecutive repetition detection:
- **Consecutive repetitions**: Lines repeated one after another
- **Non-consecutive repetitions**: Same content scattered throughout text

### `test_main_functions.py`
Tests the main filtering pipeline functions:
- **Normal conversation**: Should not be filtered
- **Repetitive examples**: Should be caught by the filter
- **Process example function**: Tests the complete processing pipeline

## Running Tests

Run all tests with `uv run`:

```bash
# Run individual tests
uv run python scripts/data/filtering_and_updates/tests/test_exact_repetition.py
uv run python scripts/data/filtering_and_updates/tests/test_2x_repetitions.py
uv run python scripts/data/filtering_and_updates/tests/test_consecutive.py
uv run python scripts/data/filtering_and_updates/tests/test_main_functions.py

# Run from the repository root
cd /weka/oe-adapt-default/nathanl/open-instruct
uv run python scripts/data/filtering_and_updates/tests/test_exact_repetition.py
```

## What the Filter Detects

The filter specifically looks for:

1. **Exact paragraph repetitions** (separated by double newlines)
   - Minimum 2x repetitions by default
   - Both consecutive and non-consecutive patterns

2. **Exact line repetitions** (separated by single newlines)  
   - Minimum 2x repetitions by default
   - Prioritizes consecutive repetitions

3. **Case-insensitive matching** with whitespace normalization
   - Handles minor formatting differences
   - Focuses on content repetition, not formatting

## Filter Configuration

- **Minimum repetitions**: 2 (configurable via `min_repetitions` parameter)
- **Minimum text length**: 50 characters (for individual detection), 100 characters (for message filtering)
- **Block size thresholds**: 
  - Paragraphs: minimum 20 characters
  - Lines: minimum 15 characters

The filter is designed to catch problematic patterns like:
- Model getting "stuck" repeating the same response
- Training data with duplicated content
- Generation loops in reasoning traces
