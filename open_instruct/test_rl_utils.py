import unittest

import open_instruct.rl_utils


class TestRLUtils(unittest.TestCase):
    def test_pack_sequences(self):
        open_instruct.rl_utils.test_pack_sequences()

    def test_calculate_advantages_packed(self):
        open_instruct.rl_utils.test_calculate_advantages_packed()

    def test_pack_sequences_logits(self):
        open_instruct.rl_utils.test_pack_sequences_logits()


if __name__ == "__main__":
    unittest.main()
