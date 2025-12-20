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

"""ActorManager for controlling evaluation and weight updates across all LLMRayActors."""

import collections
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from open_instruct import logger_utils


def find_free_port():
    """Find and return a free port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class ActorManager:
    """Centralized manager for controlling evaluation and weight updates across all LLMRayActors."""

    def __init__(self, queues: dict, args):
        self._should_stop = False
        self._last_updated = datetime.now()
        self._dashboard_port: int | None = None
        self._queues = queues or {}
        self._queue_sizes = {}
        self._queue_info = {}
        self._sample_window = 100
        self._token_history = collections.deque(maxlen=self._sample_window)
        self._total_prefill_tokens = 0
        self._total_decode_tokens = 0
        self._training_step_history = collections.deque(maxlen=self._sample_window)
        self._generation_batch_history = collections.deque(maxlen=self._sample_window)
        self._kv_cache_max_concurrency = None
        self._args = args
        if self._args.enable_queue_dashboard:
            self._setup_queue_monitoring()
            self._start_dashboard()

    def _setup_queue_monitoring(self):
        """Setup queue monitoring with background polling thread."""
        for queue_name, q in self._queues.items():
            self._queue_info[queue_name] = {"maxsize": q.maxsize if hasattr(q, "maxsize") else 0, "queue": q}
            self._queue_sizes[queue_name] = 0

        self._polling_active = True
        self._poll_thread = threading.Thread(target=self._poll_queue_sizes, daemon=True)
        self._poll_thread.start()

    def _poll_queue_sizes(self):
        """Background thread to poll queue sizes."""
        while self._polling_active:
            for queue_name, info in self._queue_info.items():
                current_size = info["queue"].size()
                self._queue_sizes[queue_name] = current_size
            time.sleep(0.5)

    def _start_dashboard(self):
        """Start the FastAPI dashboard server in a background thread."""
        if self._args.queue_dashboard_port is None:
            self._dashboard_port = find_free_port()
        else:
            self._dashboard_port = self._args.queue_dashboard_port
        assert self._dashboard_port is not None
        app = FastAPI(title="ActorManager Dashboard")

        static_dir = Path(__file__).parent / "static"
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the HTML dashboard."""
            html_path = Path(__file__).parent / "static" / "dashboard.html"
            with open(html_path) as f:
                return f.read()

        @app.get("/api/status")
        async def api_status():
            """Return the current status as JSON."""
            queues_data = {
                queue_name: {"current": self._queue_sizes.get(queue_name, 0), "maxsize": info["maxsize"]}
                for queue_name, info in self._queue_info.items()
            }

            return {
                "should_stop": self._should_stop,
                "last_updated": self._last_updated.isoformat(),
                "queues": queues_data,
                "token_stats": self.get_token_stats(),
                "timing_stats": self.get_timing_stats(),
                "concurrency_per_engine": self._kv_cache_max_concurrency,
                "total_concurrency": self._kv_cache_max_concurrency * self._args.vllm_num_engines,
                "batch_size": self._args.num_unique_prompts_rollout * self._args.num_samples_per_prompt_rollout,
            }

        def run_server():
            assert self._dashboard_port is not None
            uvicorn.run(app, host="0.0.0.0", port=self._dashboard_port, log_level="error")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        hostname = socket.getfqdn()

        logger = logger_utils.setup_logger(__name__)
        logger.info(f"Dashboard server started at http://{hostname}:{self._dashboard_port}")

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

        self._total_prefill_tokens += prompt_tokens
        self._total_decode_tokens += generation_tokens

        self._token_history.append(
            {"timestamp": current_time, "prompt_tokens": prompt_tokens, "generation_tokens": generation_tokens}
        )

    def report_token_statistics(self, token_stats):
        """Report token statistics using TokenStatistics object."""
        current_time = time.time()

        self._total_prefill_tokens += token_stats.num_prompt_tokens
        self._total_decode_tokens += token_stats.num_response_tokens

        self._token_history.append(
            {
                "timestamp": current_time,
                "prompt_tokens": token_stats.num_prompt_tokens,
                "generation_tokens": token_stats.num_response_tokens,
            }
        )

        self._generation_batch_history.append(token_stats.generation_time)

    def report_training_step_time(self, duration: float):
        """Report the time taken for a training step."""
        self._training_step_history.append(duration)

    def report_batch_generation_time(self, duration: float):
        """Report the time taken to generate a batch of data."""
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

        window_prompt_tokens = 0
        window_generation_tokens = 0
        oldest_timestamp = self._token_history[0]["timestamp"]

        for entry in self._token_history:
            window_prompt_tokens += entry["prompt_tokens"]
            window_generation_tokens += entry["generation_tokens"]

        time_span = current_time - oldest_timestamp if len(self._token_history) > 1 else 1

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
        avg_training_step_time = (
            sum(self._training_step_history) / len(self._training_step_history) if self._training_step_history else 0
        )

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

    def get_dashboard_port(self):
        """Get the port number where the dashboard is running."""
        return self._dashboard_port

    def cleanup(self):
        """Clean up resources including stopping the polling thread."""
        logger = logger_utils.setup_logger(__name__)

        # Stop the polling thread if dashboard was enabled
        if self._args.enable_queue_dashboard:
            logger.info("Stopping queue polling thread...")
            self._polling_active = False
            # Wait for the thread to finish with a timeout
            self._poll_thread.join(timeout=2.0)
