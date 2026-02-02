import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import datasets
import parameterized
from transformers import AutoTokenizer

from open_instruct import data_loader, dpo_utils


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

from open_instruct.olmo_core_callbacks import BeakerCallbackV2, PerfCallback  # noqa: E402

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


def mock_training_run(callback, loader: data_loader.HFDataLoader, num_steps: int = 10):
    """Mock a training run with the given callback and data_loader.

    Returns:
        dict: The metrics recorded during training.
    """
    recorded_metrics = {}

    def mock_record_metric(name, value, reduce_type=None):
        recorded_metrics[name] = value

    trainer_mock = Mock()
    trainer_mock.metrics_collect_interval = 1
    trainer_mock.record_metric = mock_record_metric
    trainer_mock.data_loader = loader

    callback._trainer = trainer_mock
    callback.pre_train()

    for step, batch in enumerate(loader):
        if step >= num_steps:
            break

        token_count = loader.global_num_tokens_in_batch(batch)
        trainer_mock.get_metric = Mock(return_value=Mock(item=lambda tc=token_count: tc))

        callback._step = step + 1
        callback._last_step = step
        time.sleep(0.01)
        callback.post_step()

    return recorded_metrics


class TestPerfCallbackMFU(unittest.TestCase):
    """Test that PerfCallback MFU calculation uses correct token counts."""

    def test_mfu_with_different_padding(self, batch_size: int = 2, real_tokens: int = 5):
        """Verify that different padding amounts don't affect MFU."""
        dataset = datasets.Dataset.from_dict(
            {
                "chosen_input_ids": [list(range(real_tokens)) for _ in range(batch_size * 10)],
                "chosen_attention_mask": [[1] * real_tokens for _ in range(batch_size * 10)],
                "chosen_labels": [list(range(real_tokens)) for _ in range(batch_size * 10)],
                "rejected_input_ids": [list(range(real_tokens)) for _ in range(batch_size * 10)],
                "rejected_attention_mask": [[1] * real_tokens for _ in range(batch_size * 10)],
                "rejected_labels": [list(range(real_tokens)) for _ in range(batch_size * 10)],
                "index": list(range(batch_size * 10)),
            }
        )

        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")

        mock_model_dims = MagicMock()
        mock_model_dims.approximate_learner_utilization.return_value = {"mfu": 50.0}

        callback = PerfCallback(
            model_dims=mock_model_dims,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            num_training_gpus=1,
        )

        mock_time = [0.0]

        def mock_perf_counter():
            mock_time[0] += 1.0
            return mock_time[0]

        for max_length in [real_tokens, real_tokens + 10, real_tokens + 50, real_tokens + 100]:
            collator = dpo_utils.DataCollatorForSeq2SeqDPO(
                tokenizer=tokenizer, model=None, padding="longest", max_length=max_length
            )
            loader = data_loader.HFDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                seed=42,
                dp_rank=0,
                dp_world_size=1,
                work_dir=tempfile.gettempdir(),
                collator=collator,
            )

            mock_time[0] = 0.0
            with patch("time.perf_counter", mock_perf_counter):
                metrics = mock_training_run(callback, loader, num_steps=1)

            self.assertEqual(metrics["perf/mfu"], 50.0, f"MFU mismatch at max_length={max_length}")


if __name__ == "__main__":
    unittest.main()
