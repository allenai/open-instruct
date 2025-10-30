import unittest

import parameterized

import open_instruct.dataset_processor


class TestDatasetProcessor(unittest.TestCase):
    @parameterized.parameterized.expand(
        [("too_little_data", 296, 120, 1), ("optimal", 1500, 120, 3), ("too_much_data", 1000000, 120, 120)]
    )
    def test_get_num_proc(self, name, num_examples, max_workers, expected):
        result = open_instruct.dataset_processor.get_num_proc(
            num_examples, max_workers, open_instruct.dataset_processor.APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
