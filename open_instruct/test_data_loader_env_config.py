import unittest

from open_instruct.data_loader import _merge_env_config, add_prompt_to_generator
from open_instruct.dataset_transformation import INPUT_IDS_PROMPT_KEY


class _QueueStub:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class TestEnvConfigMerging(unittest.TestCase):
    def test_reject_legacy_single_env_base(self):
        base = {"max_steps": 5, "env_name": "counter", "is_text_env": False}
        with self.assertRaises(ValueError):
            _merge_env_config(base, None)

    def test_merge_multi_env_base(self):
        base = {
            "max_steps": 7,
            "env_configs": [
                {"env_name": "counter", "is_text_env": False},
                {"env_name": "guess_number", "is_text_env": False},
            ],
        }
        merged = _merge_env_config(base, None)
        self.assertEqual(merged, base)

    def test_sample_env_configs_merge_with_base_defaults(self):
        base = {
            "max_steps": 9,
            "env_configs": [
                {"env_name": "counter", "is_text_env": False},
                {"env_name": "guess_number", "is_text_env": False},
            ],
        }
        sample = {"env_configs": [{"env_name": "counter", "target": 11}]}
        merged = _merge_env_config(base, sample)
        self.assertEqual(merged["max_steps"], 9)
        self.assertEqual(merged["env_configs"], [{"env_name": "counter", "is_text_env": False, "target": 11}])

    def test_legacy_sample_env_config_rejected(self):
        base = {
            "max_steps": 9,
            "env_configs": [
                {"env_name": "counter", "is_text_env": False},
                {"env_name": "guess_number", "is_text_env": False},
            ],
        }
        with self.assertRaises(ValueError):
            _merge_env_config(base, {"env_name": "counter", "target": 3})

    def test_sample_env_without_name_raises(self):
        base = {"max_steps": 9, "env_configs": [{"env_name": "counter", "is_text_env": False}]}
        with self.assertRaises(ValueError):
            _merge_env_config(base, {"env_configs": [{"target": 3}]})

    def test_add_prompt_to_generator_emits_normalized_env_payload(self):
        queue = _QueueStub()
        base = {
            "max_steps": 4,
            "env_configs": [
                {"env_name": "counter", "is_text_env": False},
                {"env_name": "guess_number", "is_text_env": False},
            ],
        }
        example = {
            "index": 0,
            INPUT_IDS_PROMPT_KEY: [1, 2, 3],
            "env_config": {"env_configs": [{"env_name": "counter", "target": 2}]},
        }

        add_prompt_to_generator(
            example=example,
            epoch_number=0,
            param_prompt_Q=queue,
            generation_config=object(),
            is_eval=False,
            base_env_config=base,
        )

        self.assertEqual(len(queue.items), 1)
        request = queue.items[0]
        self.assertEqual(request.env_config["max_steps"], 4)
        self.assertEqual(
            request.env_config["env_configs"], [{"env_name": "counter", "is_text_env": False, "target": 2}]
        )


if __name__ == "__main__":
    unittest.main()
