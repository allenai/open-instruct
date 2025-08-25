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
import os
import threading
import time
from datetime import datetime

from open_instruct import logger_utils


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
        self._inference_batch_size = None
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
                            let kvCacheHtml = `
                                <div class="token-card" style="border-left-color: #FF9800;">
                                    <div class="token-grid">
                                        <div class="token-stat">
                                            <div class="token-label">Max Theoretical Generation Concurrency</div>
                                            <div class="token-value" style="color: #FF9800;">${data.kv_cache_max_concurrency}</div>
                                            <div class="token-rate">Maximum concurrent generations supported by KV cache configuration</div>
                                        </div>
                            `;

                            if (data.inference_batch_size !== null && data.inference_batch_size !== undefined) {
                                kvCacheHtml += `
                                        <div class="token-stat">
                                            <div class="token-label">Inference Batch Size</div>
                                            <div class="token-value" style="color: #FF9800;">${data.inference_batch_size}</div>
                                            <div class="token-rate">Number of prompts processed per vLLM engine batch</div>
                                        </div>
                                `;
                            }

                            kvCacheHtml += `
                                    </div>
                                </div>
                            `;

                            document.getElementById('kv-cache-container').innerHTML = kvCacheHtml;
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
                "inference_batch_size": self._inference_batch_size,
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

    def report_token_statistics(self, token_stats):
        """Report token statistics using TokenStatistics object."""
        current_time = time.time()

        # Update totals
        self._total_prefill_tokens += token_stats.num_prompt_tokens
        self._total_decode_tokens += token_stats.num_response_tokens

        # Add to history for moving average (deque automatically maintains last N samples)
        self._token_history.append(
            {
                "timestamp": current_time,
                "prompt_tokens": token_stats.num_prompt_tokens,
                "generation_tokens": token_stats.num_response_tokens,
            }
        )

        # Also report batch generation time
        self._generation_batch_history.append(token_stats.generation_time)

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

    def set_inference_batch_size(self, batch_size: int):
        """Set the inference batch size value."""
        self._inference_batch_size = batch_size

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
