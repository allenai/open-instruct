import unittest

import open_instruct.dataset_processor


class TestDatasetProcessor(unittest.TestCase):
    def test_get_num_proc_too_little_data(self):
        result = open_instruct.dataset_processor.get_num_proc(
            296, 120, open_instruct.dataset_processor.APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU
        )
        self.assertEqual(result, 1)

    def test_get_num_proc_optimal(self):
        result = open_instruct.dataset_processor.get_num_proc(
            1500, 120, open_instruct.dataset_processor.APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU
        )
        self.assertEqual(result, 3)

    def test_get_num_proc_too_much_data(self):
        result = open_instruct.dataset_processor.get_num_proc(
            1000000, 120, open_instruct.dataset_processor.APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU
        )
        self.assertEqual(result, 120)


if __name__ == "__main__":
    unittest.main()
