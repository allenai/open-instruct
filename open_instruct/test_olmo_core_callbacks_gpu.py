import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import parameterized


class MockCallback:
    _trainer = None
    _step = 0

    @property
    def trainer(self):
        return self._trainer

    @property
    def step(self):
        return self._step


_MOCKED_MODULES = [
    "olmo_core",
    "olmo_core.distributed",
    "olmo_core.distributed.utils",
    "olmo_core.train",
    "olmo_core.train.callbacks",
    "olmo_core.train.callbacks.callback",
    "olmo_core.train.callbacks.comet",
    "olmo_core.train.callbacks.wandb",
    "olmo_core.train.common",
    "open_instruct.utils",
]
_original_modules = {k: sys.modules.get(k) for k in _MOCKED_MODULES}

mock_olmo_core_distributed = MagicMock()
mock_olmo_core_distributed.utils.get_rank = Mock(return_value=0)
mock_callback = MagicMock()

mock_utils = MagicMock()

sys.modules["olmo_core"] = MagicMock()
sys.modules["olmo_core.distributed"] = MagicMock()
sys.modules["olmo_core.distributed.utils"] = mock_olmo_core_distributed
sys.modules["olmo_core.train"] = MagicMock()
sys.modules["olmo_core.train.callbacks"] = MagicMock()
sys.modules["olmo_core.train.callbacks.callback"] = mock_callback
sys.modules["olmo_core.train.callbacks.comet"] = MagicMock()
sys.modules["olmo_core.train.callbacks.wandb"] = MagicMock()
sys.modules["olmo_core.train.common"] = MagicMock()
sys.modules["open_instruct.utils"] = mock_utils

mock_callback.Callback = MockCallback
sys.modules["olmo_core.train.callbacks.comet"].CometCallback = type("CometCallback", (), {"priority": 100})
sys.modules["olmo_core.train.callbacks.wandb"].WandBCallback = type("WandBCallback", (), {"priority": 100})

from open_instruct.olmo_core_callbacks import BeakerCallbackV2  # noqa: E402

for _key in _MOCKED_MODULES:
    if _original_modules[_key] is None:
        sys.modules.pop(_key, None)
    else:
        sys.modules[_key] = _original_modules[_key]


class TestBeakerCallbackPreTrain(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pre_train_saves_files(self):
        callback = BeakerCallbackV2()
        callback.enabled = True
        callback.config = {"key": "value", "nested": {"a": 1}}
        callback.result_dir = self.temp_dir.name

        trainer_mock = Mock()
        trainer_mock.callbacks = {}
        callback._trainer = trainer_mock

        with (
            patch("open_instruct.olmo_core_callbacks.get_rank", return_value=0),
            patch.dict("os.environ", {"BEAKER_WORKLOAD_ID": "test-workload-123"}),
            patch("subprocess.run") as mock_subprocess,
        ):
            callback.pre_train()
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            self.assertEqual(call_args[0][0], ["uv", "pip", "freeze"])

        config_path = f"{self.temp_dir.name}/olmo-core/config.json"
        with open(config_path) as f:
            saved_config = json.load(f)
        self.assertEqual(saved_config, {"key": "value", "nested": {"a": 1}})

        requirements_path = f"{self.temp_dir.name}/olmo-core/requirements.txt"
        with open(requirements_path) as f:
            content = f.read()
        self.assertIn("# python=", content)

    def test_pre_train_gets_tracking_url(self):
        callback = BeakerCallbackV2()
        callback.enabled = True
        callback.result_dir = self.temp_dir.name

        trainer_mock = Mock()
        trainer_mock.callbacks = {}
        callback._trainer = trainer_mock

        with (
            patch("open_instruct.olmo_core_callbacks.get_rank", return_value=0),
            patch.dict("os.environ", {"BEAKER_WORKLOAD_ID": "test-workload-123"}),
            patch("subprocess.run"),
            patch.object(callback, "_get_tracking_url", return_value="https://wandb.ai/test/run/123"),
        ):
            callback.pre_train()

        self.assertEqual(callback._url, "https://wandb.ai/test/run/123")

    def test_pre_train_skips_when_disabled(self):
        callback = BeakerCallbackV2()
        callback.enabled = False
        callback.config = {"key": "value"}
        callback.result_dir = self.temp_dir.name

        trainer_mock = Mock()
        callback._trainer = trainer_mock

        with patch("open_instruct.olmo_core_callbacks.get_rank", return_value=0):
            callback.pre_train()

        self.assertFalse(os.path.exists(f"{self.temp_dir.name}/olmo-core"))


class TestBeakerCallbackPostStep(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("throttled", True, 100, 0, False),
            ("after_throttle", True, 100, -15, True),
            ("not_on_interval", True, 50, None, False),
            ("disabled", False, 100, None, False),
        ]
    )
    def test_post_step(self, name, enabled, step, last_update_offset, expected_called):
        callback = BeakerCallbackV2()
        callback.enabled = enabled
        if last_update_offset is not None:
            callback._last_update = time.perf_counter() + last_update_offset
        else:
            callback._last_update = None

        trainer_mock = Mock()
        trainer_mock.metrics_collect_interval = 100
        callback._trainer = trainer_mock
        callback._step = step

        with (
            patch("open_instruct.olmo_core_callbacks.get_rank", return_value=0),
            patch.object(callback, "_update") as mock_update,
        ):
            callback.post_step()
            if expected_called:
                mock_update.assert_called_once()
            else:
                mock_update.assert_not_called()


class TestBeakerCallbackPostTrain(unittest.TestCase):
    @parameterized.parameterized.expand([("enabled", True, True), ("disabled", False, False)])
    def test_post_train(self, name, enabled, expected_called):
        callback = BeakerCallbackV2()
        callback.enabled = enabled

        trainer_mock = Mock()
        callback._trainer = trainer_mock

        with (
            patch("open_instruct.olmo_core_callbacks.get_rank", return_value=0),
            patch.object(callback, "_update") as mock_update,
        ):
            callback.post_train()
            if expected_called:
                mock_update.assert_called_once()
            else:
                mock_update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
