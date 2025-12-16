# Filtering Scripts Tests

This directory contains tests for the filtering scripts used in the open-instruct project.

## Test Files

- `test_filter_ngram_repetitions.py` - Comprehensive test suite for the n-gram repetition filtering functionality

## Running Tests

### Run All Tests
```bash
cd scripts/data/filtering_and_updates
python run_tests.py
```

### Run Individual Tests
```bash
cd scripts/data/filtering_and_updates
python test_filter_ngram_repetitions.py
```

## Test Coverage

The `test_filter_ngram_repetitions.py` test suite covers:

### Utility Functions
- `split_into_paragraphs()` - Text paragraph splitting
- `split_into_sentences()` - Text sentence splitting
- `is_math_or_code()` - Math/code pattern detection
- `is_code_import_or_return()` - Code import/return detection
- `is_short_phrase()` - Short phrase detection

### Core Functionality
- `detect_exact_block_repetition()` - Main repetition detection algorithm
- `process_example()` - Example processing with repetition detection
- `should_be_filtered_by_repetition()` - Filtering decision logic

### Test Scenarios
- 2x repetitions (minimum threshold testing)
- Consecutive vs non-consecutive repetitions
- Exact block repetition examples (Scooby-Doo, Marketing URL)
- Edge cases (empty text, single words, code patterns)
- N-gram repetition detection functions

### Test Cases Include
- Normal text (should NOT be flagged)
- Repetitive text (should be flagged)
- Code patterns (should be ignored)
- Math expressions (should be ignored)
- Short phrases (should be handled appropriately)
- Various repetition thresholds and patterns

## Notes

- Tests use lower thresholds than production to ensure functionality works
- Focus is on testing that functions work correctly, not exact production thresholds
- Tests verify both positive cases (repetitions detected) and negative cases (normal text not flagged)
- Edge cases and boundary conditions are covered
