import types
import unittest

from open_instruct import finetune


class TestReservedSlotTokensWithLora(unittest.TestCase):
    def test_lora_with_reserved_slot_tokens_raises_before_anything_loads(self):
        # The stubs carry only the two fields read before the check; reaching any later line
        # would raise AttributeError instead of the ValueError asserted here.
        args = types.SimpleNamespace(use_lora=True)
        tc = types.SimpleNamespace(reserved_slot_tokens=["<think>", "</think>"])
        with self.assertRaisesRegex(ValueError, "not supported with --use_lora"):
            finetune.main(args, tc)


if __name__ == "__main__":
    unittest.main()
