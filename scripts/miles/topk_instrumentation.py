"""Opt-in benchmark timing hooks; no per-token CUDA synchronization.

CUDA event spans include stream idle/launch gaps, not just kernel execution.
Nested spans (sampler contains top-k) must never be added together.
"""

import functools
import importlib
import json
import mmap
import os
import time
from collections import deque
from pathlib import Path

_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    torch = importlib.import_module("torch")
    with open(os.environ["TOPK_BENCH_CONTROL"], "r+b") as control_file:
        control = mmap.mmap(control_file.fileno(), 256)
    directory = Path(os.environ["TOPK_BENCH_OUTPUT"])
    output = None
    output_pid = None
    pending = deque()

    def emit(row):
        nonlocal output, output_pid
        if output_pid != os.getpid():
            output_pid = os.getpid()
            output = os.open(directory / f"stages-{output_pid}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.write(output, (json.dumps(row) + "\n").encode())

    def marker():
        return control[:].split(b"\0", 1)[0].decode()

    def drain():
        while pending and pending[0][1].query():
            start, end, row = pending.popleft()
            row["cuda_ms"] = start.elapsed_time(end)
            emit(row)

    def wrap_gpu(owner, name, label):
        original = getattr(owner, name)
        calls = 0

        @functools.wraps(original)
        def timed(*args, **kwargs):
            nonlocal calls
            calls += 1
            # Sample fixed frequency, across all conditions, without synchronizing.
            if calls % 32 or not torch.cuda.is_initialized() or torch.cuda.is_current_stream_capturing():
                return original(*args, **kwargs)
            drain()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            block = marker()
            host_start = time.perf_counter()
            start.record()
            result = original(*args, **kwargs)
            end.record()
            pending.append(
                (
                    start,
                    end,
                    {
                        "block": block,
                        "stage": label,
                        "host_ms": (time.perf_counter() - host_start) * 1000,
                        "sample_every": 32,
                    },
                )
            )
            return result

        setattr(owner, name, timed)

    runner = importlib.import_module("sglang.srt.model_executor.model_runner")
    sampler = importlib.import_module("sglang.srt.layers.sampler")
    logprob = importlib.import_module("sglang.srt.layers.logprob_processor")
    results = importlib.import_module("sglang.srt.managers.utils")
    wrap_gpu(runner.ModelRunner, "forward", "model_forward_stream")
    wrap_gpu(sampler.Sampler, "forward", "sampler_including_logprobs_stream")
    wrap_gpu(logprob, "get_top_logprobs", "topk_extraction_stream")
    wrap_gpu(results.GenerationBatchResult, "copy_to_cpu", "result_d2h_stream")
    if hasattr(sampler, "compute_mc_scores"):
        wrap_gpu(sampler, "compute_mc_scores", "mc32_extraction_stream")

    def wrap_host(owner, name, label):
        original = getattr(owner, name)
        calls = 0

        @functools.wraps(original)
        def timed(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls % 32:
                return original(*args, **kwargs)
            block = marker()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            emit({"block": block, "stage": label, "host_ms": (time.perf_counter() - start) * 1000, "sample_every": 32})
            return result

        setattr(owner, name, timed)

    batch_results = importlib.import_module("sglang.srt.managers.scheduler_components.batch_result_processor")
    streamer = importlib.import_module("sglang.srt.managers.scheduler_components.output_streamer")
    wrap_host(batch_results.SchedulerBatchResultProcessor, "move_logprobs_to_cpu", "scheduler_score_conversion")
    wrap_host(streamer._GenerationStreamAccumulator, "accept", "scheduler_request_accumulation")

    transport = importlib.import_module("sglang.srt.managers.io_struct")
    wrap_host(transport, "msgpack_encode", "ipc_pack")
    wrap_host(transport, "msgpack_decode", "ipc_unpack")

    manager = importlib.import_module("sglang.srt.managers.tokenizer_manager")
    original_pack = manager.TokenizerManager.add_logprob_to_meta_info

    @functools.wraps(original_pack)
    def packed(self, meta_info, state, *args, **kwargs):
        start = time.perf_counter()
        result = original_pack(self, meta_info, state, *args, **kwargs)
        elapsed = time.perf_counter() - start
        state.benchmark_pack_s = getattr(state, "benchmark_pack_s", 0.0) + elapsed
        meta_info["benchmark_pack_s"] = state.benchmark_pack_s
        return result

    manager.TokenizerManager.add_logprob_to_meta_info = packed
    responses = importlib.import_module("sglang.srt.utils.json_response")
    original_render = responses.SGLangORJSONResponse.render

    @functools.wraps(original_render)
    def rendered(self, content):
        start = time.perf_counter()
        result = original_render(self, content)
        self.benchmark_json_s = time.perf_counter() - start
        return result

    original_init = responses.SGLangORJSONResponse.__init__

    @functools.wraps(original_init)
    def initialized(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.headers["x-benchmark-json-seconds"] = str(self.benchmark_json_s)

    responses.SGLangORJSONResponse.render = rendered
    responses.SGLangORJSONResponse.__init__ = initialized
    emit({"event": "instrumentation_installed", "pid": os.getpid()})
