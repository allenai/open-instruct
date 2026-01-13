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


mock_olmo_core_distributed = MagicMock()
mock_olmo_core_distributed.utils.get_rank = Mock(return_value=0)
mock_callback = MagicMock()

sys.modules["olmo_core"] = MagicMock()
sys.modules["olmo_core.data"] = MagicMock()
sys.modules["olmo_core.data.data_loader"] = MagicMock()
sys.modules["olmo_core.distributed"] = MagicMock()
sys.modules["olmo_core.distributed.utils"] = mock_olmo_core_distributed
sys.modules["olmo_core.train"] = MagicMock()
sys.modules["olmo_core.train.callbacks"] = MagicMock()
sys.modules["olmo_core.train.callbacks.callback"] = mock_callback
sys.modules["olmo_core.train.callbacks.comet"] = MagicMock()
sys.modules["olmo_core.train.callbacks.wandb"] = MagicMock()
sys.modules["olmo_core.train.common"] = MagicMock()

mock_callback.Callback = MockCallback
sys.modules["olmo_core.train.callbacks.comet"].CometCallback = type("CometCallback", (), {"priority": 100})
sys.modules["olmo_core.train.callbacks.wandb"].WandBCallback = type("WandBCallback", (), {"priority": 100})

from open_instruct.beaker_callback import BeakerCallbackV2  # noqa: E402


def setup_beaker_mocks(mock_beaker_from_env, mock_is_beaker_job, initial_description):
    """Shared mock setup for beaker tests."""
    mock_is_beaker_job.return_value = True

    mock_client = MagicMock()
    mock_beaker_from_env.return_value = mock_client

    mock_workload = MagicMock()
    mock_client.workload.get.return_value = mock_workload

    mock_spec = MagicMock()
    mock_spec.description = initial_description
    mock_client.experiment.get_spec.return_value = mock_spec

    description_history = []

    def track_description(workload, description=None):
        if description is not None:
            description_history.append(description)

    mock_client.workload.update.side_effect = track_description

    return mock_client, mock_spec, description_history


class TestBeakerCallbackPreTrain(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("os.environ.get")
    @patch("beaker.Beaker.from_env")
    @patch("open_instruct.utils.is_beaker_job")
    def test_pre_train_saves_files(self, mock_is_beaker_job, mock_beaker_from_env, mock_environ_get):
        env_values = {"BEAKER_WORKLOAD_ID": "test-workload-123", "GIT_COMMIT": "abc123", "GIT_BRANCH": "main"}
        mock_environ_get.side_effect = lambda key, default=None: env_values.get(key, default)
        mock_client, mock_spec, description_history = setup_beaker_mocks(
            mock_beaker_from_env, mock_is_beaker_job, "Initial description"
        )

        callback = BeakerCallbackV2()
        callback.enabled = True
        callback.config = {"key": "value", "nested": {"a": 1}}
        callback.result_dir = self.temp_dir.name

        trainer_mock = Mock()
        trainer_mock.callbacks = {}
        trainer_mock.run_bookkeeping_op = Mock(side_effect=lambda fn, *args, **kwargs: fn(*args))
        trainer_mock.training_progress = Mock()
        trainer_mock.training_progress.current_step = 0
        trainer_mock.training_progress.total_steps = 100
        callback._trainer = trainer_mock

        with (
            patch("open_instruct.beaker_callback.get_rank", return_value=0),
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

        self.assertGreaterEqual(len(description_history), 1)

    @patch("os.environ.get")
    @patch("beaker.Beaker.from_env")
    @patch("open_instruct.utils.is_beaker_job")
    def test_pre_train_gets_tracking_url(self, mock_is_beaker_job, mock_beaker_from_env, mock_environ_get):
        env_values = {"BEAKER_WORKLOAD_ID": "test-workload-123", "GIT_COMMIT": "abc123", "GIT_BRANCH": "main"}
        mock_environ_get.side_effect = lambda key, default=None: env_values.get(key, default)
        setup_beaker_mocks(mock_beaker_from_env, mock_is_beaker_job, "Initial description")

        callback = BeakerCallbackV2()
        callback.enabled = True
        callback.result_dir = self.temp_dir.name

        trainer_mock = Mock()
        trainer_mock.callbacks = {}
        trainer_mock.run_bookkeeping_op = Mock(side_effect=lambda fn, *args, **kwargs: fn(*args))
        trainer_mock.training_progress = Mock()
        trainer_mock.training_progress.current_step = 0
        trainer_mock.training_progress.total_steps = 100
        callback._trainer = trainer_mock

        with (
            patch("open_instruct.beaker_callback.get_rank", return_value=0),
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

        with patch("open_instruct.beaker_callback.get_rank", return_value=0):
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
            patch("open_instruct.beaker_callback.get_rank", return_value=0),
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
            patch("open_instruct.beaker_callback.get_rank", return_value=0),
            patch.object(callback, "_update") as mock_update,
        ):
            callback.post_train()
            if expected_called:
                mock_update.assert_called_once()
            else:
                mock_update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
