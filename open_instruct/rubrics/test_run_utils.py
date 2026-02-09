#!/usr/bin/env python3
"""Test script for run_utils.py - tests LiteLLM calls and JSON extraction.

Usage:
    # From the repo root:
    uv run pytest open_instruct/rubrics/test_run_utils.py -v
"""

import asyncio
import os
import unittest

from open_instruct.rubrics.run_utils import extract_json_from_response


def _check_litellm_available():
    """Check if litellm is available."""
    try:
        import litellm  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


class TestExtractJsonFromResponse(unittest.TestCase):
    """Test suite for extract_json_from_response."""

    def test_basic_json(self):
        """Test basic JSON extraction."""
        self.assertEqual(extract_json_from_response('{"score": 2}'), {"score": 2})

    def test_json_with_text_before(self):
        """Test JSON with text before."""
        self.assertEqual(extract_json_from_response('The score is {"score": 1}'), {"score": 1})

    def test_json_with_text_after(self):
        """Test JSON with text after."""
        self.assertEqual(extract_json_from_response('{"score": 0} is the result'), {"score": 0})

    def test_json_in_markdown(self):
        """Test JSON in markdown code block."""
        self.assertEqual(extract_json_from_response('```json\n{"score": 2}\n```'), {"score": 2})

    def test_nested_json(self):
        """Test nested JSON."""
        self.assertEqual(extract_json_from_response('{"score": 1, "reason": "good"}'), {"score": 1, "reason": "good"})

    def test_double_braces(self):
        """Test double braces (common LLM mistake)."""
        self.assertEqual(extract_json_from_response('{{"score": 2}}'), {"score": 2})

    def test_no_json(self):
        """Test string with no JSON."""
        self.assertIsNone(extract_json_from_response("This has no JSON"))

    def test_invalid_json(self):
        """Test invalid JSON."""
        self.assertIsNone(extract_json_from_response('{"score": }'))

    def test_multiple_json_objects(self):
        """Test multiple JSON objects (should get last valid one)."""
        self.assertEqual(extract_json_from_response('{"a": 1} more text {"score": 2}'), {"score": 2})


@unittest.skipIf(not _check_litellm_available(), "litellm not installed")
@unittest.skipIf(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")), "No API credentials available"
)
class TestRunLitellmAsync(unittest.TestCase):
    """Test suite for run_litellm_async (requires API)."""

    def test_async_call(self):
        """Test async LiteLLM calls."""
        from open_instruct.rubrics.run_utils import run_litellm_async  # noqa: PLC0415

        model = os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1-mini")

        response = asyncio.run(
            run_litellm_async(
                model_name=model, user_prompt="What is 2+2? Answer with just the number.", temperature=0, max_tokens=10
            )
        )
        self.assertIn(response.strip(), ["4", "4."])


@unittest.skipIf(not _check_litellm_available(), "litellm not installed")
@unittest.skipIf(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")), "No API credentials available"
)
class TestRunLitellmSync(unittest.TestCase):
    """Test suite for run_litellm (sync) (requires API)."""

    def test_sync_call(self):
        """Test sync LiteLLM calls."""
        from open_instruct.rubrics.run_utils import run_litellm  # noqa: PLC0415

        model = os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1-mini")

        response = run_litellm(
            model_name=model, user_prompt="What is 3+3? Answer with just the number.", temperature=0, max_tokens=10
        )
        self.assertIn(response.strip(), ["6", "6."])


if __name__ == "__main__":
    unittest.main()
