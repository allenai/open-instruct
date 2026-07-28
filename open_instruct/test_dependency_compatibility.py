"""Focused tests for dependency APIs exercised by Open Instruct."""

import unittest

from datasets import Dataset


class TestDependencyCompatibility(unittest.TestCase):
    def test_datasets_torch_format(self):
        dataset = Dataset.from_dict({"tokens": [[1, 2], [3]]}).with_format("torch")

        self.assertEqual(dataset[0]["tokens"].tolist(), [1, 2])
        self.assertEqual(dataset[1]["tokens"].tolist(), [3])


if __name__ == "__main__":
    unittest.main()
