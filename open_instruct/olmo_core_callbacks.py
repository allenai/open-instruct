"""OLMo-core callbacks for training.

This module contains custom callbacks for OLMo-core training, including:
- BeakerCallbackV2: Beaker v2 integration (will be deleted once OLMo-core switches to Beaker v2)
- PerfCallback: MFU and tokens_per_second metrics calculation
"""

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from olmo_core.distributed.utils import get_rank
from olmo_core.train.callbacks.callback import Callback
from olmo_core.train.callbacks.comet import CometCallback
from olmo_core.train.callbacks.wandb import WandBCallback
from olmo_core.train.common import TrainingProgress

from open_instruct import logger_utils, utils
from open_instruct.utils import maybe_update_beaker_description

logger = logger_utils.setup_logger(__name__)

BEAKER_WORKLOAD_ID_ENV_VAR = "BEAKER_WORKLOAD_ID"
BEAKER_RESULT_DIR = "/results"


@dataclass
class BeakerCallbackV2(Callback):
    """
    Adds metadata to the Beaker experiment description when running as a Beaker batch job.
    Compatible with beaker-py 2.x (uses workload.update instead of experiment.set_description).
    """

    priority: ClassVar[int] = min(CometCallback.priority - 1, WandBCallback.priority - 1)
    enabled: bool | None = None
    config: dict[str, Any] | None = None
    result_dir: str = BEAKER_RESULT_DIR

    _url: str | None = field(default=None, repr=False)
    _last_update: float | None = field(default=None, repr=False)
    _start_time: float | None = field(default=None, repr=False)

    def post_attach(self) -> None:
        if self.enabled is None and BEAKER_WORKLOAD_ID_ENV_VAR in os.environ:
            self.enabled = True

    def pre_train(self) -> None:
        if self.enabled and get_rank() == 0:
            workload_id = os.environ.get(BEAKER_WORKLOAD_ID_ENV_VAR)
            if workload_id is None:
                logger.warning(f"BeakerCallbackV2: {BEAKER_WORKLOAD_ID_ENV_VAR} not set, disabling")
                self.enabled = False
                return

            self._start_time = time.perf_counter()
            logger.info(f"Running in Beaker workload {workload_id}")

            result_dir = Path(self.result_dir) / "olmo-core"
            result_dir.mkdir(parents=True, exist_ok=True)

            if self.config is not None:
                config_path = result_dir / "config.json"
                with config_path.open("w") as config_file:
                    logger.info(f"Saving config to '{config_path}'")
                    json.dump(self.config, config_file)

            requirements_path = result_dir / "requirements.txt"
            with requirements_path.open("w") as requirements_file:
                requirements_file.write(f"# python={platform.python_version()}\n")
                subprocess.run(["uv", "pip", "freeze"], stdout=requirements_file, check=True, timeout=10)

            self._url = self._get_tracking_url()
            self._update()

    def post_step(self) -> None:
        should_update = (
            self.enabled
            and get_rank() == 0
            and self.step % self.trainer.metrics_collect_interval == 0
            and (self._last_update is None or (time.perf_counter() - self._last_update) > 10)
        )
        if should_update:
            self._update()

    def post_train(self) -> None:
        if self.enabled and get_rank() == 0:
            self._update()

    def _get_tracking_url(self) -> str | None:
        for callback in self.trainer.callbacks.values():
            if isinstance(callback, WandBCallback) and callback.enabled and callback.run is not None:
                return callback.run.get_url()
            elif isinstance(callback, CometCallback) and callback.enabled and callback.exp is not None:
                return callback.exp.url
        return None

    def _update(self) -> None:
        self.trainer.run_bookkeeping_op(
            self._set_description,
            self.trainer.training_progress,
            op_name="beaker_set_description",
            allow_multiple=False,
            distributed=False,
        )
        self._last_update = time.perf_counter()

    def _set_description(self, progress: TrainingProgress) -> None:
        maybe_update_beaker_description(
            current_step=progress.current_step,
            total_steps=progress.total_steps,
            start_time=self._start_time,
            wandb_url=self._url,
        )


@dataclass
class PerfCallback(Callback):
    """Calculates MFU and tokens_per_second using same formula as dpo_tune_cache.py."""

    model_dims: utils.ModelDims
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_training_gpus: int

    _start_time: float = field(default=0.0, repr=False)
    _interval_start_time: float = field(default=0.0, repr=False)
    _step_start_time: float = field(default=0.0, repr=False)
    _total_tokens_processed: int = field(default=0, repr=False)
    _mfu_sum: float = field(default=0.0, repr=False)
    _last_step: int = field(default=0, repr=False)
    _batch_load_start: float = field(default=0.0, repr=False)
    _batch_load_time: float = field(default=0.0, repr=False)
    _wall_clock_step_start: float = field(default=0.0, repr=False)
    _prev_wall_clock_step_start: float = field(default=0.0, repr=False)

    def pre_train(self) -> None:
        self._start_time = time.perf_counter()
        self._interval_start_time = self._start_time
        self._total_tokens_processed = 0
        self._mfu_sum = 0.0
        self._last_step = 0

    def pre_load_batch(self) -> None:
        self._batch_load_start = time.perf_counter()
        self._prev_wall_clock_step_start = self._wall_clock_step_start
        self._wall_clock_step_start = self._batch_load_start

    def pre_step(self, batch: dict[str, Any]) -> None:
        del batch
        self._batch_load_time = time.perf_counter() - self._batch_load_start
        self._step_start_time = time.perf_counter()

    def post_step(self) -> None:
        if self.step % self.trainer.metrics_collect_interval != 0:
            return
        if self.step == self._last_step:
            return

        token_count_metric = self.trainer.get_metric("train/token_count")
        if token_count_metric is None:
            return
        total_tokens_step = int(token_count_metric.item())

        interval_end = time.perf_counter()
        training_time = interval_end - self._interval_start_time
        total_time_elapsed = interval_end - self._start_time

        tokens_per_second = total_tokens_step / training_time if training_time > 0 else 0
        self._total_tokens_processed += total_tokens_step
        tokens_per_second_avg = self._total_tokens_processed / total_time_elapsed

        logging_steps = self.trainer.metrics_collect_interval
        num_sequences = (
            self.per_device_train_batch_size
            * self.num_training_gpus
            * self.gradient_accumulation_steps
            * logging_steps
            * 2  # * 2 for chosen + rejected
        )
        avg_sequence_length = total_tokens_step / num_sequences if num_sequences > 0 else 0

        mfu_result = self.model_dims.approximate_learner_utilization(
            total_tokens=total_tokens_step,
            avg_sequence_length=avg_sequence_length,
            training_time=training_time,
            num_training_gpus=self.num_training_gpus,
        )

        self._mfu_sum += mfu_result["mfu"]
        mfu_avg = self._mfu_sum / (self.step // logging_steps)

        seconds_per_step = interval_end - self._step_start_time

        self.trainer.record_metric("perf/mfu", mfu_result["mfu"], reduce_type=None)
        self.trainer.record_metric("perf/mfu_avg", mfu_avg, reduce_type=None)
        self.trainer.record_metric("perf/seconds_per_step", seconds_per_step, reduce_type=None)
        self.trainer.record_metric("perf/tokens_per_second", tokens_per_second, reduce_type=None)
        self.trainer.record_metric("perf/tokens_per_second_avg", tokens_per_second_avg, reduce_type=None)
        self.trainer.record_metric("perf/total_tokens", self._total_tokens_processed, reduce_type=None)
        self.trainer.record_metric("perf/data_loading_seconds", self._batch_load_time, reduce_type=None)

        if self._prev_wall_clock_step_start > 0:
            wall_clock_per_step = self._wall_clock_step_start - self._prev_wall_clock_step_start
            self.trainer.record_metric("perf/wall_clock_per_step", wall_clock_per_step, reduce_type=None)
            if wall_clock_per_step > 0:
                data_loading_pct = 100 * self._batch_load_time / wall_clock_per_step
                self.trainer.record_metric("perf/data_loading_pct", data_loading_pct, reduce_type=None)
                step_overhead_pct = 100 * (wall_clock_per_step - seconds_per_step) / wall_clock_per_step
                self.trainer.record_metric("perf/step_overhead_pct", step_overhead_pct, reduce_type=None)

        self._interval_start_time = interval_end
        self._last_step = self.step
