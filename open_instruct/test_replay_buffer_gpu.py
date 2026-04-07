from __future__ import annotations

import tempfile
import unittest

import yaml

from open_instruct import data_loader, utils


class TestReplayBufferConfig(unittest.TestCase):
    def test_default_values(self):
        config = data_loader.StreamingDataLoaderConfig()
        self.assertEqual(config.replay_buffer, data_loader.ReplayBufferConfig())

    def test_custom_replay_buffer_config(self):
        rb = data_loader.ReplayBufferConfig(
            capacity=100, sampler="prioritized", remover="lifo", max_times_sampled=3, min_size=10
        )
        config = data_loader.StreamingDataLoaderConfig(replay_buffer=rb)
        self.assertEqual(config.replay_buffer, rb)

    def test_argument_parser_parses_yaml(self):
        yaml_content = {
            "replay_buffer": {
                "capacity": 50,
                "sampler": "lifo",
                "remover": "prioritized",
                "max_times_sampled": 5,
                "min_size": 10,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            parser = utils.ArgumentParserPlus((data_loader.StreamingDataLoaderConfig,))
            (config,) = parser.parse_yaml_file(f.name)
        self.assertEqual(
            config.replay_buffer,
            data_loader.ReplayBufferConfig(
                capacity=50, sampler="lifo", remover="prioritized", max_times_sampled=5, min_size=10
            ),
        )


if __name__ == "__main__":
    unittest.main()
