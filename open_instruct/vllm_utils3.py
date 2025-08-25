# Taken and modified from https://github.com/huggingface/trl
# Copyright 2024 The AllenAI Team. All rights reserved.
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

"""This file is copied from https://github.com/OpenRLHF/OpenRLHF"""

import collections
import copy
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import ray
import torch
import torch.distributed
import vllm
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)

from open_instruct import logger_utils
from open_instruct.queue_types import GenerationResult, RequestInfo
from open_instruct.tool_utils.tool_vllm import MaxCallsExceededTool, Tool
from open_instruct.utils import ray_get_with_progress

logger = logger_utils.setup_logger(__name__)


def _init_tool_tracking():
    """Initialize tracking variables for tool mode."""
    return {
        "num_calls": defaultdict(int),
        "timeout": defaultdict(bool),
        "tool_error": defaultdict(str),
        "tool_output": defaultdict(str),
        "tool_runtime": defaultdict(float),
        "tool_called": defaultdict(bool),
        "concat_outputs": {},
        "masks": defaultdict(list),
        "pending_tool_futures": {},
    }


def _handle_output(output, tools, tracking, sampling_params, max_tool_calls, executor):
    """
    Handle a finished output. Returns the output if it should be added to results,
    or None if it's being held for tool processing.

    This is a free function to keep the processing logic separate from the actor state.
    """
    if not tools:
        return output

    assert len(output.outputs) <= 1  # In tool mode, sampling_params.n == 1
    o = output.outputs[0]

    # Update concatenated outputs
    if output.request_id in tracking["concat_outputs"]:
        tracking["concat_outputs"][output.request_id].outputs[0].token_ids.extend(o.token_ids)
    else:
        tracking["concat_outputs"][output.request_id] = output

    tracking["masks"][output.request_id].extend([1] * len(o.token_ids))

    # Check for tool calls
    for stop_str in sampling_params.stop:
        if stop_str in tools and o.text.endswith(stop_str):
            if tracking["num_calls"][output.request_id] < max_tool_calls.get(stop_str, 0):
                tool = tools[stop_str]
            else:
                tool = MaxCallsExceededTool(start_str="<tool>", end_str="</tool>")

            future = executor.submit(tool, o.text)
            tracking["pending_tool_futures"][output.request_id] = (future, o, output)

            return None  # Output is being held for tool processing

    return output


def _process_outputs(
    outputs: List[vllm.RequestOutput],
    dataset_index: Optional[List[int]] = None,
    prompt_tokens: Optional[int] = None,
    generation_tokens: Optional[int] = None,
    generation_time: Optional[float] = None,
) -> "GenerationResult":
    """Process vLLM RequestOutputs into GenerationResult format."""
    response_ids = [list(out.token_ids) for output in outputs for out in output.outputs]
    finish_reasons = [out.finish_reason for output in outputs for out in output.outputs]

    masks = [[1] * len(resp) for resp in response_ids]
    num_calls = [0] * len(response_ids)
    timeouts = [0] * len(response_ids)
    tool_errors = [""] * len(response_ids)
    tool_outputs = [""] * len(response_ids)
    tool_runtimes = [0] * len(response_ids)
    tool_calleds = [False] * len(response_ids)

    request_info = RequestInfo(
        num_calls=num_calls,
        timeouts=timeouts,
        tool_errors=tool_errors,
        tool_outputs=tool_outputs,
        tool_runtimes=tool_runtimes,
        tool_calleds=tool_calleds,
    )

    result = GenerationResult(
        responses=response_ids,
        finish_reasons=finish_reasons,
        masks=masks,
        request_info=request_info,
        dataset_index=dataset_index,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        generation_time=generation_time,
    )

    return result


def _process_outputs_with_tools(
    outputs: List[vllm.RequestOutput],
    dataset_index: Optional[List[int]] = None,
    prompt_tokens: Optional[int] = None,
    generation_tokens: Optional[int] = None,
    generation_time: Optional[float] = None,
) -> "GenerationResult":
    """Process vLLM RequestOutputs into GenerationResult format with tool information."""
    response_ids = [list(out.token_ids) for output in outputs for out in output.outputs]
    finish_reasons = [out.finish_reason for output in outputs for out in output.outputs]

    masks = [out.mask for output in outputs for out in output.outputs]
    num_calls = [out.num_calls for output in outputs for out in output.outputs]
    timeouts = [out.timeout for output in outputs for out in output.outputs]
    tool_errors = [out.tool_error for output in outputs for out in output.outputs]
    tool_outputs = [out.tool_output for output in outputs for out in output.outputs]
    tool_runtimes = [out.tool_runtime for output in outputs for out in output.outputs]
    tool_calleds = [out.tool_called for output in outputs for out in output.outputs]

    request_info = RequestInfo(
        num_calls=num_calls,
        timeouts=timeouts,
        tool_errors=tool_errors,
        tool_outputs=tool_outputs,
        tool_runtimes=tool_runtimes,
        tool_calleds=tool_calleds,
    )

    result = GenerationResult(
        responses=response_ids,
        finish_reasons=finish_reasons,
        masks=masks,
        request_info=request_info,
        dataset_index=dataset_index,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        generation_time=generation_time,
    )

    return result


def _finalize_outputs(
    outputs, tracking, dataset_index, tools, prompt_tokens=None, generation_tokens=None, generation_time=None
):
    """Prepare final outputs based on whether tools were used."""
    if not tools:
        outputs.sort(key=lambda x: int(x.request_id.split("_")[-1]))
        return _process_outputs(
            outputs,
            dataset_index=dataset_index,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            generation_time=generation_time,
        )

    # Tool mode: add metadata and merge completions
    for req_id in tracking["masks"]:
        assert req_id in tracking["concat_outputs"], f"req_id {req_id} not in concat_outputs!"
        output = tracking["concat_outputs"][req_id].outputs[0]
        setattr(output, "mask", tracking["masks"][req_id])
        setattr(output, "num_calls", tracking["num_calls"][req_id])
        setattr(output, "timeout", tracking["timeout"][req_id])
        setattr(output, "tool_error", tracking["tool_error"][req_id])
        setattr(output, "tool_output", tracking["tool_output"][req_id])
        setattr(output, "tool_runtime", tracking["tool_runtime"][req_id])
        setattr(output, "tool_called", tracking["tool_called"][req_id])

    # Merge n completions into the same outputs
    merged_outputs = {}
    for req_id in tracking["concat_outputs"]:
        real_req_id, _ = req_id.split("-")
        if real_req_id not in merged_outputs:
            merged_outputs[real_req_id] = tracking["concat_outputs"][req_id]
        else:
            merged_outputs[real_req_id].outputs.append(tracking["concat_outputs"][req_id].outputs[0])

    final_outputs = sorted(
        merged_outputs.values(), key=lambda x: (int(x.request_id.split("-")[0]), int(x.request_id.split("-")[1]))
    )

    return _process_outputs_with_tools(
        final_outputs,
        dataset_index=dataset_index,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        generation_time=generation_time,
    )


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


@ray.remote
class ActorManager:
    """Centralized manager for controlling evaluation and weight updates across all LLMRayActors."""

    def __init__(self, queues=None):
        self._should_stop = False
        self._last_updated = datetime.now()
        self._dashboard_port = int(os.environ.get("DASHBOARD_PORT", 8080))
        self._queues = queues or {}
        self._queue_sizes = {}
        self._queue_info = {}
        self._sample_window = 100  # Keep last 100 samples for moving averages
        self._token_history = collections.deque(maxlen=self._sample_window)  # Keep last 100 token entries
        self._total_prefill_tokens = 0
        self._total_decode_tokens = 0
        # New timing metrics
        self._training_step_history = collections.deque(
            maxlen=self._sample_window
        )  # Keep last 100 training step times
        self._generation_batch_history = collections.deque(
            maxlen=self._sample_window
        )  # Keep last 100 batch generation times
        # KV cache metrics
        self._kv_cache_max_concurrency = None
        self._setup_queue_monitoring()
        self._start_dashboard()

    def _setup_queue_monitoring(self):
        """Setup queue monitoring with background polling thread."""
        # Initialize queue info
        for queue_name, q in self._queues.items():
            self._queue_info[queue_name] = {"maxsize": q.maxsize if hasattr(q, "maxsize") else 0, "queue": q}
            self._queue_sizes[queue_name] = 0

        # Start background polling thread
        if self._queues:
            self._polling_active = True
            self._poll_thread = threading.Thread(target=self._poll_queue_sizes, daemon=True)
            self._poll_thread.start()

    def _poll_queue_sizes(self):
        """Background thread to poll queue sizes."""
        while self._polling_active:
            for queue_name, info in self._queue_info.items():
                try:
                    # Get current size from the queue
                    current_size = info["queue"].size()
                    self._queue_sizes[queue_name] = current_size
                except Exception:
                    # If we can't get the size, keep the last known value
                    pass
            # Small sleep to avoid excessive polling
            time.sleep(0.5)

    def _start_dashboard(self):
        """Start the FastAPI dashboard server in a background thread."""
        import socket

        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse

        app = FastAPI(title="ActorManager Dashboard")

        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the HTML dashboard."""
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>ActorManager Dashboard</title>
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        max-width: 800px;
                        margin: 50px auto;
                        padding: 20px;
                        background: #f5f5f5;
                    }
                    .container {
                        background: white;
                        border-radius: 8px;
                        padding: 30px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }
                    h1 {
                        color: #333;
                        margin-bottom: 30px;
                    }
                    .status-card {
                        background: #f8f9fa;
                        border-radius: 6px;
                        padding: 20px;
                        margin: 20px 0;
                        border-left: 4px solid #ccc;
                    }
                    .status-card.active {
                        border-left-color: #dc3545;
                        background: #fff5f5;
                    }
                    .status-card.inactive {
                        border-left-color: #28a745;
                        background: #f5fff5;
                    }
                    .status-label {
                        font-size: 14px;
                        color: #666;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                        margin-bottom: 10px;
                    }
                    .status-value {
                        font-size: 24px;
                        font-weight: bold;
                        margin: 10px 0;
                    }
                    .status-value.active {
                        color: #dc3545;
                    }
                    .status-value.inactive {
                        color: #28a745;
                    }
                    .timestamp {
                        font-size: 14px;
                        color: #999;
                        margin-top: 10px;
                    }
                    .refresh-indicator {
                        position: fixed;
                        top: 20px;
                        right: 20px;
                        width: 10px;
                        height: 10px;
                        background: #28a745;
                        border-radius: 50%;
                        animation: pulse 2s infinite;
                    }
                    @keyframes pulse {
                        0% { opacity: 1; }
                        50% { opacity: 0.3; }
                        100% { opacity: 1; }
                    }
                    .queue-section {
                        margin-top: 30px;
                    }
                    .queue-card {
                        background: #f8f9fa;
                        border-radius: 6px;
                        padding: 15px;
                        margin: 15px 0;
                    }
                    .queue-name {
                        font-size: 16px;
                        font-weight: 600;
                        color: #444;
                        margin-bottom: 10px;
                    }
                    .queue-stats {
                        font-size: 14px;
                        color: #666;
                        margin-bottom: 8px;
                    }
                    .progress-bar-container {
                        width: 100%;
                        height: 30px;
                        background: #e9ecef;
                        border-radius: 4px;
                        overflow: hidden;
                        position: relative;
                    }
                    .progress-bar {
                        height: 100%;
                        background: linear-gradient(90deg, #28a745, #20c997);
                        transition: width 0.3s ease;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-weight: 600;
                        font-size: 14px;
                    }
                    .progress-bar.warning {
                        background: linear-gradient(90deg, #ffc107, #fd7e14);
                    }
                    .progress-bar.danger {
                        background: linear-gradient(90deg, #dc3545, #c82333);
                    }
                    .progress-text {
                        position: absolute;
                        width: 100%;
                        text-align: center;
                        line-height: 30px;
                        color: #333;
                        font-weight: 600;
                        font-size: 14px;
                        z-index: 1;
                    }
                    h2 {
                        color: #444;
                        margin-top: 30px;
                        margin-bottom: 20px;
                        font-size: 20px;
                        border-bottom: 2px solid #e9ecef;
                        padding-bottom: 10px;
                    }
                    .token-section {
                        margin-top: 30px;
                    }
                    .token-card {
                        background: #f0f8ff;
                        border-radius: 6px;
                        padding: 20px;
                        margin: 15px 0;
                        border-left: 4px solid #2196F3;
                    }
                    .token-grid {
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 20px;
                        margin-top: 15px;
                    }
                    .token-stat {
                        background: white;
                        padding: 15px;
                        border-radius: 4px;
                    }
                    .token-label {
                        font-size: 12px;
                        color: #666;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    }
                    .token-value {
                        font-size: 24px;
                        font-weight: bold;
                        color: #2196F3;
                        margin-top: 5px;
                    }
                    .token-rate {
                        font-size: 14px;
                        color: #999;
                        margin-top: 3px;
                    }
                </style>
            </head>
            <body>
                <div class="refresh-indicator"></div>
                <div class="container">
                    <h1>🎛️ ActorManager Dashboard</h1>
                    <div id="status-container">
                        <div class="status-card">
                            <div class="status-label">Loading...</div>
                        </div>
                    </div>
                    <h2>📊 Queue Status</h2>
                    <div id="queue-container">
                        <div class="queue-card">
                            <div class="queue-name">Loading queues...</div>
                        </div>
                    </div>
                    <h2>⚡ Token Throughput</h2>
                    <div id="token-container">
                        <div class="token-card">
                            <div class="token-grid">
                                <div class="token-stat">
                                    <div class="token-label">Loading...</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <h2>⏱️ Performance Metrics</h2>
                    <div id="timing-container">
                        <div class="token-card" style="border-left-color: #9C27B0;">
                            <div class="token-grid">
                                <div class="token-stat">
                                    <div class="token-label">Loading...</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <h2>💾 KV Cache</h2>
                    <div id="kv-cache-container">
                        <div class="token-card" style="border-left-color: #FF9800;">
                            <div class="token-grid">
                                <div class="token-stat">
                                    <div class="token-label">Loading...</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <script>
                    async function updateStatus() {
                        const response = await fetch('/api/status');
                        const data = await response.json();

                        const statusClass = data.should_stop ? 'active' : 'inactive';
                        const statusText = data.should_stop ? 'STOPPED' : 'RUNNING';

                        document.getElementById('status-container').innerHTML = `
                            <div class="status-card ${statusClass}">
                                <div class="status-label">Should Stop Status</div>
                                <div class="status-value ${statusClass}">${statusText}</div>
                                <div class="timestamp">Last updated: ${data.last_updated}</div>
                            </div>
                        `;

                        // Update queue visualizations
                        if (data.queues) {
                            let queueHtml = '';
                            for (const [queueName, queueInfo] of Object.entries(data.queues)) {
                                const percentage = queueInfo.maxsize > 0
                                    ? Math.round((queueInfo.current / queueInfo.maxsize) * 100)
                                    : 0;

                                let barClass = '';
                                if (percentage >= 90) {
                                    barClass = 'danger';
                                } else if (percentage >= 70) {
                                    barClass = 'warning';
                                }

                                queueHtml += `
                                    <div class="queue-card">
                                        <div class="queue-name">${queueName}</div>
                                        <div class="queue-stats">
                                            Current: ${queueInfo.current} / Max: ${queueInfo.maxsize}
                                            (${percentage}% full)
                                        </div>
                                        <div class="progress-bar-container">
                                            <div class="progress-text">${queueInfo.current} / ${queueInfo.maxsize}</div>
                                            <div class="progress-bar ${barClass}" style="width: ${percentage}%;">
                                            </div>
                                        </div>
                                    </div>
                                `;
                            }

                            if (queueHtml === '') {
                                queueHtml = `
                                    <div class="queue-card">
                                        <div class="queue-name">No queues configured</div>
                                    </div>
                                `;
                            }

                            document.getElementById('queue-container').innerHTML = queueHtml;
                        }

                        // Update token statistics
                        if (data.token_stats) {
                            const stats = data.token_stats;
                            const formatNumber = (num) => {
                                if (num >= 1000000) return (num / 1000000).toFixed(2) + 'M';
                                if (num >= 1000) return (num / 1000).toFixed(2) + 'K';
                                return num.toFixed(0);
                            };

                            document.getElementById('token-container').innerHTML = `
                                <div class="token-card">
                                    <div class="token-grid">
                                        <div class="token-stat">
                                            <div class="token-label">Prefill Tokens/Sec</div>
                                            <div class="token-value">${stats.prefill_tokens_per_sec.toFixed(1)}</div>
                                            <div class="token-rate">Avg over ${stats.sample_count} samples | Total: ${formatNumber(stats.total_prefill_tokens)}</div>
                                        </div>
                                        <div class="token-stat">
                                            <div class="token-label">Decode Tokens/Sec</div>
                                            <div class="token-value">${stats.decode_tokens_per_sec.toFixed(1)}</div>
                                            <div class="token-rate">Avg over ${stats.sample_count} samples | Total: ${formatNumber(stats.total_decode_tokens)}</div>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }

                        // Update timing statistics
                        if (data.timing_stats) {
                            const timing = data.timing_stats;
                            const formatTime = (seconds) => {
                                if (seconds >= 60) return (seconds / 60).toFixed(2) + ' min';
                                return seconds.toFixed(2) + ' sec';
                            };

                            document.getElementById('timing-container').innerHTML = `
                                <div class="token-card" style="border-left-color: #9C27B0;">
                                    <div class="token-grid">
                                        <div class="token-stat">
                                            <div class="token-label">Avg Training Step Time</div>
                                            <div class="token-value" style="color: #9C27B0;">${formatTime(timing.avg_training_step_time)}</div>
                                            <div class="token-rate">Moving avg: last ${timing.training_step_count} of 100 steps</div>
                                        </div>
                                        <div class="token-stat">
                                            <div class="token-label">Avg Batch Generation Time</div>
                                            <div class="token-value" style="color: #9C27B0;">${formatTime(timing.avg_batch_generation_time)}</div>
                                            <div class="token-rate">Moving avg: last ${timing.batch_generation_count} of 100 batches</div>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }

                        // Update KV cache statistics
                        if (data.kv_cache_max_concurrency !== null && data.kv_cache_max_concurrency !== undefined) {
                            document.getElementById('kv-cache-container').innerHTML = `
                                <div class="token-card" style="border-left-color: #FF9800;">
                                    <div class="token-grid">
                                        <div class="token-stat">
                                            <div class="token-label">Max Theoretical Generation Concurrency</div>
                                            <div class="token-value" style="color: #FF9800;">${data.kv_cache_max_concurrency}</div>
                                            <div class="token-rate">Maximum concurrent generations supported by KV cache configuration</div>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }
                    }

                    // Update immediately and then every second
                    updateStatus();
                    setInterval(updateStatus, 1000);
                </script>
            </body>
            </html>
            """
            return html_content

        @app.get("/api/status")
        async def api_status():
            """Return the current status as JSON."""
            # Prepare queue information
            queues_data = {}
            for queue_name, info in self._queue_info.items():
                queues_data[queue_name] = {"current": self._queue_sizes.get(queue_name, 0), "maxsize": info["maxsize"]}

            # Get token statistics
            token_stats = self.get_token_stats()

            # Get timing statistics
            timing_stats = self.get_timing_stats()

            return {
                "should_stop": self._should_stop,
                "last_updated": self._last_updated.isoformat(),
                "queues": queues_data,
                "token_stats": token_stats,
                "timing_stats": timing_stats,
                "kv_cache_max_concurrency": self._kv_cache_max_concurrency,
            }

        # Store app reference for potential cleanup
        self._app = app

        # Run server in background thread
        def run_server():
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=self._dashboard_port,
                log_level="error",  # Minimize logging
            )

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        # Get the actual hostname/IP for logging
        hostname = socket.gethostname()
        try:
            # Try to get the actual IP address
            host_ip = socket.gethostbyname(hostname)
        except socket.error:
            # Fallback to localhost if we can't resolve
            host_ip = "127.0.0.1"

        logger = logger_utils.setup_logger(__name__)
        logger.info(f"Dashboard server started at http://{host_ip}:{self._dashboard_port} (hostname: {hostname})")

    def set_should_stop(self, should_stop: bool):
        """Set whether actors should stop processing."""
        self._should_stop = should_stop
        self._last_updated = datetime.now()

    def should_stop(self) -> bool:
        """Check if actors should stop processing."""
        return self._should_stop

    def report_token_stats(self, prompt_tokens: int, generation_tokens: int):
        """Report token statistics from main thread."""
        current_time = time.time()

        # Update totals
        self._total_prefill_tokens += prompt_tokens
        self._total_decode_tokens += generation_tokens

        # Add to history for moving average (deque automatically maintains last N samples)
        self._token_history.append(
            {"timestamp": current_time, "prompt_tokens": prompt_tokens, "generation_tokens": generation_tokens}
        )

    def report_training_step_time(self, duration: float):
        """Report the time taken for a training step."""
        # Add to history (deque automatically maintains last N samples)
        self._training_step_history.append(duration)

    def report_batch_generation_time(self, duration: float):
        """Report the time taken to generate a batch of data."""
        # Add to history (deque automatically maintains last N samples)
        self._generation_batch_history.append(duration)

    def set_kv_cache_max_concurrency(self, max_concurrency: int):
        """Set the KV cache max concurrency value."""
        self._kv_cache_max_concurrency = max_concurrency

    def get_token_stats(self):
        """Calculate and return current token statistics."""
        if not self._token_history:
            return {
                "total_prefill_tokens": self._total_prefill_tokens,
                "total_decode_tokens": self._total_decode_tokens,
                "prefill_tokens_per_sec": 0,
                "decode_tokens_per_sec": 0,
                "sample_count": 0,
            }

        current_time = time.time()

        # Calculate total tokens in the sample window
        window_prompt_tokens = 0
        window_generation_tokens = 0
        oldest_timestamp = self._token_history[0]["timestamp"]

        for entry in self._token_history:
            window_prompt_tokens += entry["prompt_tokens"]
            window_generation_tokens += entry["generation_tokens"]

        # Calculate time span of samples
        time_span = current_time - oldest_timestamp if len(self._token_history) > 1 else 1

        # Calculate tokens per second based on actual time span of samples
        prompt_tokens_per_sec = window_prompt_tokens / time_span if time_span > 0 else 0
        generation_tokens_per_sec = window_generation_tokens / time_span if time_span > 0 else 0

        return {
            "total_prefill_tokens": self._total_prefill_tokens,
            "total_decode_tokens": self._total_decode_tokens,
            "prefill_tokens_per_sec": prompt_tokens_per_sec,
            "decode_tokens_per_sec": generation_tokens_per_sec,
            "sample_count": len(self._token_history),
        }

    def get_timing_stats(self):
        """Calculate and return current timing statistics."""
        # Calculate average training step time from last N samples
        avg_training_step_time = (
            sum(self._training_step_history) / len(self._training_step_history) if self._training_step_history else 0
        )

        # Calculate average batch generation time from last N samples
        avg_batch_generation_time = (
            sum(self._generation_batch_history) / len(self._generation_batch_history)
            if self._generation_batch_history
            else 0
        )

        return {
            "avg_training_step_time": avg_training_step_time,
            "avg_batch_generation_time": avg_batch_generation_time,
            "training_step_count": len(self._training_step_history),
            "batch_generation_count": len(self._generation_batch_history),
        }


class LLMRayActor:
    """Ray actor for LLM generation with optional tool support."""

    def __init__(
        self,
        *args,
        tools: Optional[Dict[str, Tool]] = None,
        max_tool_calls: Optional[Dict[str, int]] = None,
        bundle_indices: list = None,
        prompt_queue=None,
        results_queue=None,
        eval_results_queue=None,
        actor_manager=None,
        **kwargs,
    ):
        self.logger = logger_utils.setup_logger(__name__)
        self.tools = tools or {}
        self.max_tool_calls = max_tool_calls or {}
        self.request_metadata = {}

        if self.tools:
            self.executor = ThreadPoolExecutor(max_workers=20)
        else:
            self.executor = None

        noset_visible_devices = kwargs.pop("noset_visible_devices")
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            self.logger.info(f"creating LLM with bundle_indices={bundle_indices}")

        self.llm_engine = vllm.LLMEngine.from_engine_args(vllm.EngineArgs(*args, **kwargs))

        self.prompt_queue = prompt_queue
        self.results_queue = results_queue
        self.eval_results_queue = eval_results_queue
        self.actor_manager = actor_manager

    def process_from_queue(self, timeout: float = 60.0):
        """Run generation loop using LLMEngine directly, with optional tool support.

        Returns:
            int: Number of requests processed (0 or 1)
        """
        while True:
            # Non-blocking check for should_stop using ray.wait
            should_stop_ref = self.actor_manager.should_stop.remote()
            ready_refs, _ = ray.wait([should_stop_ref], timeout=0.1)
            if ready_refs and ray.get(ready_refs[0]):
                return 0

            try:
                request = self.prompt_queue.get(timeout=timeout)
            except queue.Empty:
                return 0

            result = self._process_request(request)

            try:
                if request.is_eval:
                    self.eval_results_queue.put(result, timeout=10)
                else:
                    self.results_queue.put(result, timeout=10)
                return 1  # Successfully processed one request
            except queue.Full:
                self.logger.warning("Results queue is full, discarding result.")
                return 0

    def _process_request(self, request):
        """Unified processing for both tool and non-tool generation."""
        import time

        prompts = request.prompts
        sampling_params = request.generation_config

        self.logger.info(f"[LLMRayActor] Processing request with {len(prompts)} prompts, tools={bool(self.tools)}")

        if self.tools:
            # Need n=1 for individual tool tracking
            sampling_params = copy.deepcopy(sampling_params)
            original_n = request.generation_config.n
            sampling_params.n = 1
            tracking = _init_tool_tracking()
            tokenizer = self.llm_engine.tokenizer
        else:
            original_n = 1
            tracking = None
            tokenizer = None

        self._add_initial_requests(prompts, sampling_params, original_n, request.training_step)

        outputs = []
        iteration = 0

        while True:
            iteration += 1

            # Poll tool futures first (matching ToolUseLLM order)
            if tracking and tracking.get("pending_tool_futures"):
                self._poll_tool_futures(tracking, sampling_params, tokenizer)

            # Process engine steps - ONLY if there are unfinished requests (matching ToolUseLLM)
            if self.llm_engine.has_unfinished_requests():
                step_outputs = list(self.llm_engine.step())
                for output in step_outputs:
                    if output.finished:
                        result = _handle_output(
                            output, self.tools, tracking, sampling_params, self.max_tool_calls, self.executor
                        )
                        if result is not None:
                            outputs.append(result)

            # Check termination condition (matching ToolUseLLM exactly)
            pending_count = len(tracking["pending_tool_futures"]) if tracking else 0
            if not self.llm_engine.has_unfinished_requests() and pending_count == 0:
                self.logger.info(f"[LLMRayActor] Terminating after {iteration} iterations with {len(outputs)} outputs")
                break

        # Calculate token statistics
        end_time = time.time()
        total_prompt_tokens = 0
        total_generation_tokens = 0
        earliest_start_time = float("inf")

        # Collect stats from all processed requests
        for output in outputs:
            request_id = output.request_id
            if request_id in self.request_metadata:
                metadata = self.request_metadata[request_id]
                total_prompt_tokens += metadata["prompt_tokens"]
                earliest_start_time = min(earliest_start_time, metadata["start_time"])

                # Calculate generation tokens from output
                for completion in output.outputs:
                    total_generation_tokens += len(completion.token_ids)

        generation_time = end_time - earliest_start_time if earliest_start_time != float("inf") else 0

        # Clean up metadata for processed requests
        for output in outputs:
            self.request_metadata.pop(output.request_id, None)

        result = _finalize_outputs(
            outputs,
            tracking,
            request.dataset_index,
            self.tools,
            prompt_tokens=total_prompt_tokens,
            generation_tokens=total_generation_tokens,
            generation_time=generation_time,
        )
        return result

    def _add_initial_requests(self, prompts, sampling_params, n_samples, training_step):
        """Add initial requests to the engine."""
        import time

        for i, prompt in enumerate(prompts):
            if self.tools:
                # Create individual requests for each sample when using tools
                for j in range(n_samples):
                    request_id = f"{training_step}_{i}-{j}"
                    self.request_metadata[request_id] = {"start_time": time.time(), "prompt_tokens": len(prompt)}
                    tokens_prompt = vllm.TokensPrompt(prompt_token_ids=prompt)
                    self.llm_engine.add_request(request_id, tokens_prompt, sampling_params)
            else:
                # Standard request format for non-tool mode
                request_id = f"batch_{training_step}_{i}"
                self.request_metadata[request_id] = {"start_time": time.time(), "prompt_tokens": len(prompt)}
                tokens_prompt = vllm.TokensPrompt(prompt_token_ids=prompt)
                self.llm_engine.add_request(request_id, tokens_prompt, sampling_params)

    def _poll_tool_futures(self, tracking, sampling_params, tokenizer):
        """Poll and handle completed tool executions."""
        if not self.tools or not tracking["pending_tool_futures"]:
            return

        dict_keys_to_delete = []

        for req_id, (future, last_o, last_output) in tracking["pending_tool_futures"].items():
            if not future.done():
                continue

            # Tool future is done, process it
            tool_result = future.result()  # Get the tool result

            last_prompt_token_ids = last_output.prompt_token_ids
            last_token_ids = last_o.token_ids
            tool_output_token_ids = tokenizer.encode(
                "<output>\n" + tool_result.output + "</output>\n", add_special_tokens=False
            )
            tracking["timeout"][req_id] = tool_result.timeout
            tracking["tool_error"][req_id] += "" if tool_result.error is None else tool_result.error
            tracking["tool_output"][req_id] += tool_result.output
            tracking["tool_runtime"][req_id] += tool_result.runtime
            tracking["tool_called"][req_id] = True

            # Edge case 1: clip against model context length
            prompt_and_tool_output_token = last_prompt_token_ids + last_token_ids + tool_output_token_ids
            tracking["num_calls"][req_id] += 1
            excess = len(prompt_and_tool_output_token) - self.llm_engine.model_config.max_model_len
            if excess > 0:
                tool_output_token_ids = tool_output_token_ids[:-excess]
                can_make_new_request = False
            else:
                can_make_new_request = True

            # Edge case 2: clip against per-request max_tokens
            remaining = sampling_params.max_tokens - len(tracking["masks"][req_id])
            if remaining <= 0:
                tool_output_token_ids = []
            elif len(tool_output_token_ids) > remaining:
                tool_output_token_ids = tool_output_token_ids[:remaining]

            tracking["concat_outputs"][req_id].outputs[0].token_ids.extend(tool_output_token_ids)
            tracking["masks"][req_id].extend([0] * len(tool_output_token_ids))
            new_sample_tokens = sampling_params.max_tokens - len(tracking["masks"][req_id])
            can_make_new_request = can_make_new_request and new_sample_tokens > 0

            if can_make_new_request:
                new_sampling_params = copy.deepcopy(sampling_params)
                new_sampling_params.max_tokens = new_sample_tokens

                try:
                    self.llm_engine.add_request(
                        req_id, vllm.TokensPrompt(prompt_token_ids=prompt_and_tool_output_token), new_sampling_params
                    )
                except Exception as e:
                    # Match original ToolUseLLM behavior - just log and continue
                    self.logger.error(f"[_poll_tool_futures] Error adding request {req_id}: {e}")

            dict_keys_to_delete.append(req_id)

        for req_id in dict_keys_to_delete:
            if req_id in tracking["pending_tool_futures"]:
                del tracking["pending_tool_futures"][req_id]

    def init_process_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend,
        use_ray=False,
        timeout_minutes=120,
    ):
        return self.llm_engine.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray, timeout_minutes),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm_engine.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm_engine.collective_rpc(
            "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
        )

    def reset_prefix_cache(self):
        self.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm_engine.sleep(level=level)

    def wake_up(self, tags: Optional[list[str]] = None):
        self.llm_engine.wake_up(tags)

    def ready(self):
        return True

    def get_kv_cache_info(self):
        """Get KV cache max concurrency from the vLLM engine."""
        # Calculate maximum theoretical concurrency
        max_concurrency = vllm.v1.core.kv_cache_utils.get_max_concurrency_for_kv_cache_config(
            self.llm_engine.engine_config, self.llm_engine.cache_config
        )

        return int(max_concurrency)


def get_cuda_arch_list() -> str:
    """Get CUDA compute capabilities and format them for TORCH_CUDA_ARCH_LIST."""
    if not torch.cuda.is_available():
        return ""

    cuda_capabilities = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_capabilities.append(f"{major}.{minor}")

    # Remove duplicates and sort
    cuda_capabilities = sorted(set(cuda_capabilities))
    cuda_arch_list = ";".join(cuda_capabilities)
    logger.info(
        f"Detected CUDA compute capabilities: {cuda_capabilities}, setting TORCH_CUDA_ARCH_LIST={cuda_arch_list}"
    )
    return cuda_arch_list


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    enforce_eager: bool,
    tokenizer_name_or_path: str,
    pretrain: str,
    revision: str,
    seed: int,
    enable_prefix_caching: bool,
    max_model_len: int,
    vllm_gpu_memory_utilization: float = 0.9,
    single_gpu_mode: bool = False,
    pg: Optional[ray.util.placement_group] = None,
    vllm_enable_sleep=False,
    tools: Optional[Dict[str, Tool]] = None,
    max_tool_calls: List[int] = [5],
    prompt_queue=None,
    results_queue=None,
    eval_results_queue=None,
    actor_manager=None,
) -> list[LLMRayActor]:
    # Convert max_tool_calls to a dict mapping tool end strings to their limits
    if tools:
        assert len(max_tool_calls) == 1 or len(max_tool_calls) == len(tools), (
            "max_tool_calls must have length 1 (applies to all tools) or same length as tools (per-tool limit)"
        )
        # tool key is the end_str
        if len(max_tool_calls) == 1:
            max_tool_calls_dict = {end_str: max_tool_calls[0] for end_str in tools.keys()}
        else:
            max_tool_calls_dict = {end_str: limit for end_str, limit in zip(tools.keys(), max_tool_calls)}
    else:
        max_tool_calls_dict = {}

    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1 and single_gpu_mode:
        # every worker will use 0.5 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        num_gpus = 0.5

    logger.info(f"num_gpus: {num_gpus}")

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=i * tensor_parallel_size,
        )

        vllm_engines.append(
            ray.remote(LLMRayActor)
            .options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                # VLLM v1 multiprocessing is required due to https://github.com/vllm-project/vllm/issues/15349
                runtime_env=ray.runtime_env.RuntimeEnv(
                    env_vars={"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TORCH_CUDA_ARCH_LIST": get_cuda_arch_list()}
                ),
            )
            .remote(
                model=pretrain,
                revision=revision,
                tokenizer=tokenizer_name_or_path,
                tokenizer_revision=revision,
                trust_remote_code=True,
                worker_extension_cls="open_instruct.vllm_utils_workerwrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=enforce_eager,
                dtype="bfloat16",
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                enable_prefix_caching=enable_prefix_caching,
                max_model_len=max_model_len,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.2 if use_hybrid_engine else 1,
                enable_sleep_mode=vllm_enable_sleep,
                noset_visible_devices=ray_noset_visible_devices(),
                prompt_queue=prompt_queue,
                results_queue=results_queue,
                eval_results_queue=eval_results_queue,
                actor_manager=actor_manager,
                tools=tools,
                max_tool_calls=max_tool_calls_dict,
            )
        )

    # Verify engines initialized successfully
    try:
        ray_get_with_progress(
            [engine.ready.remote() for engine in vllm_engines], "Initializing vLLM engines", timeout=300
        )
    except TimeoutError as e:
        logger.error(f"vLLM engines failed to initialize: {e}")
        # Kill partially initialized actors before raising
        for engine in vllm_engines:
            ray.kill(engine)
        raise RuntimeError(f"vLLM engine initialization timed out: {e}")

    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep", rank_0_only=False)

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """
    Batch call a method on multiple vLLM engines.
    Args:
        engines: List of vLLM engine instances
        method_name: Name of the method to call
        rank_0_only: Only execute on rank 0 if True
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
    Returns:
        List of results from ray.get() if on rank 0, None otherwise
    """
    import torch

    if rank_0_only and torch.distributed.get_rank() != 0:
        return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)


if __name__ == "__main__":
    num_engines = 1
    tensor_parallel_size = 1
    world_size = num_engines * tensor_parallel_size + 1
    vllm_engines = create_vllm_engines(
        num_engines=num_engines,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        pretrain="facebook/opt-125m",
        revision="main",
        seed=42,
        enable_prefix_caching=False,
        max_model_len=1024,
    )
    llm = vllm_engines[0]
    from vllm.utils import get_ip, get_open_port

    master_address = get_ip()
    master_port = get_open_port()
    backend = "gloo"

    refs = [
        engine.init_process_group.remote(
            master_address, master_port, i * tensor_parallel_size + 1, world_size, "openrlhf", backend=backend
        )
        for i, engine in enumerate(vllm_engines)
    ]
    model_update_group = init_process_group(
        backend=backend,
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name="openrlhf",
    )
    ray.get(refs)
    output = ray.get(llm.generate.remote("San Franciso is a"))
    logger.info(f"output: {output}")
