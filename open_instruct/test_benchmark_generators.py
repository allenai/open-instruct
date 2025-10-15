import unittest

import parameterized

from open_instruct import utils


class TestBenchmark(unittest.TestCase):
    @parameterized.parameterized.expand(
        [("NVIDIA H100 80GB HBM3", "h100"), ("NVIDIA L40S", "l40s"), ("NVIDIA RTX A6000", "a6000")]
    )
    def test_get_device_name(self, device_name, expected):
        result = utils.get_device_name(device_name)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
