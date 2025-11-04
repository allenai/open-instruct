import unittest

import parameterized

import mason


class TestMason(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (
                "remove_arg_without_value",
                ["python", "script.py", "--with_tracking", "--output", "out.txt"],
                {"--with_tracking": False},
                ["python", "script.py", "--output", "out.txt"],
            ),
            (
                "remove_arg_with_value",
                ["python", "script.py", "--checkpoint_state_dir", "/path/to/dir", "--output", "out.txt"],
                {"--checkpoint_state_dir": True},
                ["python", "script.py", "--output", "out.txt"],
            ),
            (
                "remove_multiple_args",
                ["python", "script.py", "--with_tracking", "--checkpoint_state_dir", "/path", "--output", "out.txt"],
                {"--with_tracking": False, "--checkpoint_state_dir": True},
                ["python", "script.py", "--output", "out.txt"],
            ),
            (
                "arg_not_present",
                ["python", "script.py", "--output", "out.txt"],
                {"--nonexistent": True},
                ["python", "script.py", "--output", "out.txt"],
            ),
            ("empty_command", [], {"--with_tracking": False}, []),
            (
                "empty_args_to_remove",
                ["python", "script.py", "--output", "out.txt"],
                {},
                ["python", "script.py", "--output", "out.txt"],
            ),
            (
                "remove_all_cache_excluded_args",
                [
                    "python",
                    "open_instruct/grpo_fast.py",
                    "--with_tracking",
                    "--checkpoint_state_freq",
                    "200",
                    "--checkpoint_state_dir",
                    "/weka/path",
                    "--gs_checkpoint_state_dir",
                    "gs://bucket",
                    "--output",
                    "out.txt",
                ],
                mason.CACHE_EXCLUDED_ARGS,
                ["python", "open_instruct/grpo_fast.py", "--output", "out.txt"],
            ),
            (
                "arg_at_end_without_value",
                ["python", "script.py", "--output", "out.txt", "--with_tracking"],
                {"--with_tracking": False},
                ["python", "script.py", "--output", "out.txt"],
            ),
            (
                "arg_at_end_with_value",
                ["python", "script.py", "--output", "out.txt", "--checkpoint_dir", "/path"],
                {"--checkpoint_dir": True},
                ["python", "script.py", "--output", "out.txt"],
            ),
        ]
    )
    def test_build_command_without_args(self, name, command, args_to_remove, expected):
        result = mason.build_command_without_args(command, args_to_remove)
        self.assertEqual(result, expected)

    @parameterized.parameterized.expand(
        [
            ([], []),
            (["python", "script.py"], ["python", "script.py"]),
            (
                ["python", "script.py", "--arg", '{"key": "value"}'],
                ["python", "script.py", "--arg", '\'{"key": "value"}\''],
            ),
            (
                ["python", "--dataset_mixer", '{"trl-internal-testing/sentiment-trl-style": 1.0}'],
                ["python", "--dataset_mixer", "'{\"trl-internal-testing/sentiment-trl-style\": 1.0}'"],
            ),
            (
                ["python", "--arg1", '{"nested": {"key": "value"}}', "--arg2", "normal"],
                ["python", "--arg1", '\'{"nested": {"key": "value"}}\'', "--arg2", "normal"],
            ),
            (["python", "--config", "{a:1,b:2}", "--flag"], ["python", "--config", "'{a:1,b:2}'", "--flag"]),
            (["echo", "no braces here"], ["echo", "no braces here"]),
            (
                ["python", "--json", '{"a":1}', "--json2", '{"b":2}'],
                ["python", "--json", "'{\"a\":1}'", "--json2", "'{\"b\":2}'"],
            ),
        ]
    )
    def test_escape_strings(self, input_command, expected_output):
        result = mason.escape_strings(input_command.copy())
        self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
