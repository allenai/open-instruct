"""This module only exists because Olmo-core still uses Beaker v1, so we have to write our own BeakerCallback that works with v2.

This module will be deleted once Olmo-core switches to Beaker v2."""

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

from open_instruct import logger_utils
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
