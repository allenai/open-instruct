# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vLLM-Ascend compatibility layer.

This module is only imported from code paths that have already detected an NPU
accelerator, so GPU processes never import ``vllm_ascend``. Imports stay
guarded so importing on a machine without ``vllm_ascend`` still works; the
engine getters raise at call time instead of at import time.
"""

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

try:
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLTrainerSendWeightsArgs,
        HCCLWeightTransferEngine,
    )
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
        NPUIPCTrainerSendWeightsArgs,
        NPUIPCWeightTransferEngine,
    )
except ImportError:
    HCCLTrainerSendWeightsArgs = None
    HCCLWeightTransferEngine = None
    NPUIPCTrainerSendWeightsArgs = None
    NPUIPCWeightTransferEngine = None

try:
    from vllm_ascend.sample.sampler import AscendSampler
    from vllm_ascend.worker.worker import NPUWorker
except ImportError:
    AscendSampler = None
    NPUWorker = None

if NPUWorker is not None and AscendSampler is not None:

    class OpenInstructNPUWorker(NPUWorker):
        """Pass vLLM's requested logprob mode to the vLLM-Ascend sampler.

        vLLM passes ``logprobs_mode`` through ``ModelConfig``.  The affected
        vLLM-Ascend v1 runner creates ``AscendSampler()`` without that value,
        which silently falls back to ``raw_logprobs``.  The worker is resolved
        by its qualified name inside the EngineCore worker process, so this
        keeps the compatibility fix in Open-Instruct rather than modifying an
        installed package.

        TODO(npu): remove once vLLM-Ascend's runner forwards
        ``ModelConfig.logprobs_mode`` into ``AscendSampler`` itself.
        """

        def init_device(self):
            super().init_device()

            if getattr(self, "use_v2_model_runner", False):
                return

            logprobs_mode = getattr(self.vllm_config.model_config, "logprobs_mode", None)
            sampler = getattr(self.model_runner, "sampler", None)
            if logprobs_mode is None or sampler is None or sampler.logprobs_mode == logprobs_mode:
                return

            assert AscendSampler is not None
            self.model_runner.sampler = AscendSampler(logprobs_mode=logprobs_mode)

            rejection_sampler = getattr(self.model_runner, "rejection_sampler", None)
            if rejection_sampler is not None:
                self.model_runner.rejection_sampler = type(rejection_sampler)(self.model_runner.sampler)

            logger.info(
                "Recreated vLLM-Ascend sampler with logprobs_mode=%s (was %s)", logprobs_mode, sampler.logprobs_mode
            )

else:
    OpenInstructNPUWorker = None

WORKER_CLS_PATH = f"{__name__}.OpenInstructNPUWorker"


def get_collective_weight_transfer():
    """Return the (trainer args class, engine class) HCCL pair."""
    if HCCLTrainerSendWeightsArgs is None or HCCLWeightTransferEngine is None:
        raise RuntimeError("vLLM-Ascend HCCL weight transfer is unavailable")
    return HCCLTrainerSendWeightsArgs, HCCLWeightTransferEngine


def get_ipc_weight_transfer():
    """Return the (trainer args class, engine class) NPU IPC pair."""
    if NPUIPCTrainerSendWeightsArgs is None or NPUIPCWeightTransferEngine is None:
        raise RuntimeError("vLLM-Ascend NPU IPC weight transfer is unavailable")
    return NPUIPCTrainerSendWeightsArgs, NPUIPCWeightTransferEngine
