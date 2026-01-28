"""Tests for calculate_response_fraction script."""

import logging
import tempfile
import unittest

from scripts.calculate_response_fraction import parse_script_for_args

logging.basicConfig(level=logging.INFO)


class TestParseScriptForArgs(unittest.TestCase):
    """Tests for parse_script_for_args function."""

    def test_parse_mixer_list(self):
        script_content = """#!/bin/bash
uv run python train.py \\
    --mixer_list allenai/dataset1 1.0 allenai/dataset2 0.5 \\
    --max_seq_length 16384
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            f.flush()
            args = parse_script_for_args(f.name)

        self.assertIn("mixer_list", args)
        self.assertEqual(args["mixer_list"], ["allenai/dataset1", "1.0", "allenai/dataset2", "0.5"])
        self.assertEqual(args["max_seq_length"], 16384)

    def test_parse_max_seq_length(self):
        script_content = """#!/bin/bash
uv run python train.py --max_seq_length 8192 --other_arg value
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            f.flush()
            args = parse_script_for_args(f.name)

        self.assertEqual(args.get("max_seq_length"), 8192)

    def test_parse_tokenizer_name(self):
        script_content = """#!/bin/bash
uv run python train.py --tokenizer_name_or_path allenai/OLMo-2-1124-7B
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            f.flush()
            args = parse_script_for_args(f.name)

        self.assertEqual(args.get("tokenizer_name_or_path"), "allenai/OLMo-2-1124-7B")

    def test_parse_model_name(self):
        script_content = """#!/bin/bash
uv run python train.py --model_name_or_path /path/to/model
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            f.flush()
            args = parse_script_for_args(f.name)

        self.assertEqual(args.get("model_name_or_path"), "/path/to/model")

    def test_parse_chat_template_name(self):
        script_content = """#!/bin/bash
uv run python train.py --chat_template_name olmo123
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            f.flush()
            args = parse_script_for_args(f.name)

        self.assertEqual(args.get("chat_template_name"), "olmo123")

    def test_parse_multiline_script(self):
        script_content = """#!/bin/bash
MODEL_NAME=/weka/model/path
uv run python train.py \\
    --model_name_or_path $MODEL_NAME \\
    --mixer_list allenai/dataset1 125000 \\
        allenai/dataset2 125000 \\
    --max_seq_length 16384 \\
    --seed 123
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            f.flush()
            args = parse_script_for_args(f.name)

        self.assertIn("mixer_list", args)
        self.assertEqual(args["max_seq_length"], 16384)

    def test_skips_comments(self):
        script_content = """#!/bin/bash
# This is a comment
# --max_seq_length 1234
uv run python train.py --max_seq_length 5678
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            f.flush()
            args = parse_script_for_args(f.name)

        self.assertEqual(args.get("max_seq_length"), 5678)


if __name__ == "__main__":
    unittest.main()
