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

    def __init__(self, queues: dict, args, vllm_engines=None):
        self._should_stop = False
        self._last_updated = datetime.now()
        self._dashboard_port = None
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
        self._vllm_engines = vllm_engines or []
        self._last_metrics_collection_time = 0
        # Cache for static token rates (updated only on new batch completion)
        self._cached_token_rates = {"prefill_tokens_per_sec": 0, "decode_tokens_per_sec": 0, "last_update_count": 0}
        # Training progress tracking
        self._current_training_step = 0
        self._total_training_steps = getattr(args, "num_training_steps", None)
        self._training_start_time = None
        # MFU/MBU tracking
        self._model_utilization_history = collections.deque(maxlen=self._sample_window)
        self._memory_usage_stats = {"total_gpu_memory_used": 0, "average_kv_cache_size": 0, "peak_memory_usage": 0}
        # Actor status tracking
        self._actor_status = {}  # actor_id -> {unfinished_requests, inference_batch_size, last_update}
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
        """Background thread to poll queue sizes and collect vLLM metrics."""
        while self._polling_active:
            # Poll queue sizes
            for queue_name, info in self._queue_info.items():
                current_size = info["queue"].size()
                self._queue_sizes[queue_name] = current_size

            # Collect vLLM metrics every 10 seconds
            current_time = time.time()
            if (current_time - self._last_metrics_collection_time) >= 10.0:
                self._collect_vllm_metrics()
                self._last_metrics_collection_time = current_time

            time.sleep(0.5)

    def _collect_vllm_metrics(self):
        """Collect metrics from all vLLM engines."""
        if not self._vllm_engines:
            return

        try:
            # Collect metrics from all engines asynchronously
            import ray

            metrics_futures = []
            for engine in self._vllm_engines:
                try:
                    future = engine.get_engine_metrics.remote()
                    metrics_futures.append(future)
                except Exception as e:
                    logger = logger_utils.setup_logger(__name__)
                    logger.warning(f"Error getting metrics from engine: {e}")

            if metrics_futures:
                # Get all metrics with a short timeout to avoid blocking
                try:
                    all_metrics = ray.get(metrics_futures, timeout=5.0)

                    # Aggregate metrics across all engines
                    total_gpu_memory = 0
                    total_kv_cache_memory = 0
                    total_mfu = 0
                    total_mbu = 0
                    valid_engines = 0

                    for metrics in all_metrics:
                        if metrics and isinstance(metrics, dict):
                            total_gpu_memory += metrics.get("gpu_memory_reserved_gb", 0)
                            total_kv_cache_memory += metrics.get("gpu_memory_allocated_gb", 0)  # Approximation
                            total_mfu += metrics.get("mfu_estimate", 0)
                            total_mbu += metrics.get("mbu_estimate", 0)
                            valid_engines += 1

                    if valid_engines > 0:
                        # Report aggregated metrics
                        avg_mfu = total_mfu / valid_engines
                        avg_mbu = total_mbu / valid_engines
                        self.report_model_utilization(avg_mfu, avg_mbu)
                        self.report_memory_usage(total_gpu_memory, total_kv_cache_memory)

                except ray.exceptions.GetTimeoutError:
                    logger = logger_utils.setup_logger(__name__)
                    logger.warning("Timeout collecting vLLM metrics")
                except Exception as e:
                    logger = logger_utils.setup_logger(__name__)
                    logger.warning(f"Error processing vLLM metrics: {e}")

        except Exception as e:
            logger = logger_utils.setup_logger(__name__)
            logger.warning(f"Error in _collect_vllm_metrics: {e}")

    def _start_dashboard(self):
        """Start the FastAPI dashboard server in a background thread."""
        if self._args.queue_dashboard_port is None:
            self._dashboard_port = find_free_port()
        else:
            self._dashboard_port = self._args.queue_dashboard_port
        app = FastAPI(title="ActorManager Dashboard")

        static_dir = Path(__file__).parent / "static"
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the HTML dashboard."""
            html_path = Path(__file__).parent / "static" / "dashboard.html"
            with open(html_path, "r") as f:
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
                "training_progress": self.get_training_progress(),
                "utilization_stats": self.get_utilization_stats(),
                "memory_stats": self.get_memory_stats(),
                "kv_cache_max_concurrency": self._kv_cache_max_concurrency,
                # This is less confusing to users.
                "inference_batch_size": self._args.inference_batch_size * self._args.num_samples_per_prompt_rollout,
                "actor_status": self.get_actor_status(),
            }

        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=self._dashboard_port, log_level="error")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        hostname = socket.getfqdn()

        logger = logger_utils.setup_logger(__name__)
        logger.info(f"Dashboard server started at http://{hostname}:{self._dashboard_port}")
        
        # Give server a moment to start, then test if it's responding
        import time
        time.sleep(1)
        try:
            import socket as sock
            test_sock = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
            test_sock.settimeout(2)
            result = test_sock.connect_ex(("127.0.0.1", self._dashboard_port))
            test_sock.close()
            if result != 0:
                logger.warning("❌ Dashboard server is not responding on localhost")
        except Exception as e:
            logger.warning(f"Could not test dashboard server connectivity: {e}")

    def set_should_stop(self, should_stop: bool):
        """Set whether actors should stop processing."""
        self._should_stop = should_stop
        self._last_updated = datetime.now()

    def should_stop(self) -> bool:
        """Check if actors should stop processing."""
        return self._should_stop

    def report_actor_status(self, actor_id: str, unfinished_requests: int, inference_batch_size: int):
        """Report status from an individual actor."""
        current_time = time.time()
        self._actor_status[actor_id] = {
            "unfinished_requests": unfinished_requests,
            "inference_batch_size": inference_batch_size,
            "last_update": current_time,
        }

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

        # Report batch generation time (avoid double reporting via report_batch_generation_time)
        # Add validation to prevent extreme outliers (e.g., > 300 seconds)
        if 0 < token_stats.generation_time < 300:
            self._generation_batch_history.append(token_stats.generation_time)

    def report_training_step_time(self, duration: float):
        """Report the time taken for a training step."""
        self._training_step_history.append(duration)

    def update_training_step(self, step: int):
        """Update the current training step."""
        if self._training_start_time is None:
            self._training_start_time = time.time()
        self._current_training_step = step

    def report_batch_generation_time(self, duration: float):
        """Report the time taken to generate a batch of data."""
        # Add validation to prevent extreme outliers (e.g., > 300 seconds)
        if 0 < duration < 300:
            self._generation_batch_history.append(duration)

    def set_kv_cache_max_concurrency(self, max_concurrency: int):
        """Set the KV cache max concurrency value."""
        self._kv_cache_max_concurrency = max_concurrency

    def set_vllm_engines(self, vllm_engines):
        """Set the vLLM engines for metrics collection."""
        self._vllm_engines = vllm_engines or []

    def get_token_stats(self):
        """Calculate and return current token statistics."""
        if not self._token_history:
            return {
                "total_prefill_tokens": self._total_prefill_tokens,
                "total_decode_tokens": self._total_decode_tokens,
                "prefill_tokens_per_sec": self._cached_token_rates["prefill_tokens_per_sec"],
                "decode_tokens_per_sec": self._cached_token_rates["decode_tokens_per_sec"],
                "sample_count": 0,
            }

        # Only update rates if we have new token history entries
        current_sample_count = len(self._token_history)
        if current_sample_count > self._cached_token_rates["last_update_count"]:
            current_time = time.time()

            window_prompt_tokens = 0
            window_generation_tokens = 0
            oldest_timestamp = self._token_history[0]["timestamp"]

            for entry in self._token_history:
                window_prompt_tokens += entry["prompt_tokens"]
                window_generation_tokens += entry["generation_tokens"]

            time_span = current_time - oldest_timestamp if len(self._token_history) > 1 else 1

            # Update cached rates
            self._cached_token_rates["prefill_tokens_per_sec"] = (
                window_prompt_tokens / time_span if time_span > 0 else 0
            )
            self._cached_token_rates["decode_tokens_per_sec"] = (
                window_generation_tokens / time_span if time_span > 0 else 0
            )
            self._cached_token_rates["last_update_count"] = current_sample_count

        return {
            "total_prefill_tokens": self._total_prefill_tokens,
            "total_decode_tokens": self._total_decode_tokens,
            "prefill_tokens_per_sec": self._cached_token_rates["prefill_tokens_per_sec"],
            "decode_tokens_per_sec": self._cached_token_rates["decode_tokens_per_sec"],
            "sample_count": current_sample_count,
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

    def get_training_progress(self):
        """Calculate and return training progress and ETA."""
        if not self._total_training_steps or self._current_training_step <= 0:
            return {
                "current_step": self._current_training_step,
                "total_steps": self._total_training_steps,
                "progress_percent": 0,
                "eta_seconds": None,
                "eta_formatted": "N/A",
            }

        progress_percent = (self._current_training_step / self._total_training_steps) * 100
        eta_seconds = None
        eta_formatted = "N/A"

        if self._training_start_time and self._current_training_step > 0:
            elapsed_time = time.time() - self._training_start_time
            avg_time_per_step = elapsed_time / self._current_training_step
            remaining_steps = self._total_training_steps - self._current_training_step
            eta_seconds = remaining_steps * avg_time_per_step

            if eta_seconds > 0:
                hours = int(eta_seconds // 3600)
                minutes = int((eta_seconds % 3600) // 60)
                if hours > 0:
                    eta_formatted = f"{hours}h {minutes}m"
                else:
                    eta_formatted = f"{minutes}m"

        return {
            "current_step": self._current_training_step,
            "total_steps": self._total_training_steps,
            "progress_percent": progress_percent,
            "eta_seconds": eta_seconds,
            "eta_formatted": eta_formatted,
        }

    def report_model_utilization(self, mfu: float, mbu: float):
        """Report MFU (Model FLOPs Utilization) and MBU (Memory Bandwidth Utilization)."""
        current_time = time.time()
        # Validate and clamp values to reasonable ranges
        mfu = max(0, min(100, mfu))
        mbu = max(0, min(100, mbu))

        self._model_utilization_history.append({"timestamp": current_time, "mfu": mfu, "mbu": mbu})

    def report_memory_usage(self, gpu_memory_used: float, kv_cache_size: float):
        """Report memory usage statistics."""
        self._memory_usage_stats["total_gpu_memory_used"] = gpu_memory_used
        self._memory_usage_stats["average_kv_cache_size"] = kv_cache_size
        self._memory_usage_stats["peak_memory_usage"] = max(
            self._memory_usage_stats["peak_memory_usage"], gpu_memory_used
        )

    def get_utilization_stats(self):
        """Calculate and return current utilization statistics."""
        if not self._model_utilization_history:
            return {"mfu": 0, "mbu": 0, "sample_count": 0}

        # Calculate averages over the sample window
        total_mfu = sum(entry["mfu"] for entry in self._model_utilization_history)
        total_mbu = sum(entry["mbu"] for entry in self._model_utilization_history)
        count = len(self._model_utilization_history)

        return {
            "mfu": total_mfu / count if count > 0 else 0,
            "mbu": total_mbu / count if count > 0 else 0,
            "sample_count": count,
        }

    def get_memory_stats(self):
        """Return current memory usage statistics."""
        return self._memory_usage_stats.copy()

    def get_actor_status(self):
        """Return current actor status information."""
        current_time = time.time()
        # Filter out stale actor data (older than 60 seconds)
        active_actors = {}
        for actor_id, status in self._actor_status.items():
            if current_time - status["last_update"] < 60:
                active_actors[actor_id] = {
                    "actor_id_short": actor_id[:8],  # Short version for display
                    "unfinished_requests": status["unfinished_requests"],
                    "inference_batch_size": status["inference_batch_size"],
                    "last_update": status["last_update"],
                    "is_active": status["unfinished_requests"] > 0,
                }
        return active_actors

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
