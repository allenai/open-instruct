"""Run the pinned MILES producer and serving fleet with a discard consumer."""

import asyncio
import hashlib
import json
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import httpx
import ray
from miles.backends.core_utils.rollout import pipeline_observer
from miles.ray import wiring
from miles.ray.rollout.inference_controller import InferenceController
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.inference_rollout.compatibility import load_rollout_function
from miles.rollout.inference_rollout.inference_rollout_train import get_worker_urls
from miles.utils.compiler_cache import startup_cache
from miles.utils.function_registry import load_function
from miles.utils.http_utils import init_http_client
from scripts.miles import snapshot_options

from open_instruct import logger_utils
from open_instruct.miles.execution import benchmark_resources, workflow
from open_instruct.miles.rollout import benchmark_metrics, benchmark_replay

logger = logger_utils.setup_logger(__name__)


async def no_reward(args, sample, **kwargs):
    """Keep the generation contract; benchmark validation disables reward filters."""
    return [0.0] * len(sample) if isinstance(sample, list) else 0.0


def inference_specs(args, specs):
    configured = startup_cache.configure_specs(args, specs)
    return [spec for spec in configured if not spec.name.startswith("trainer-")]


async def server_info(args):
    async with httpx.AsyncClient(timeout=30) as client:
        result = []
        for url in await get_worker_urls(args):
            response = await client.get(url + "/get_server_info")
            response.raise_for_status()
            result.append({"url": url, "info": response.json()})
        return result


async def engine_metrics(args):
    """Retain raw counters and scrape bounds; failures remain missing observations."""
    async with httpx.AsyncClient(timeout=10) as client:

        async def scrape(url):
            row = {"url": url, "started_unix": time.time(), "started": time.monotonic()}
            try:
                response = await client.get(url + "/metrics")
                response.raise_for_status()
                row["text"] = response.text
            except Exception as error:
                row["error"] = str(error)
                logger.warning("Benchmark engine metric scrape failed: %s", error)
            row.update(finished_unix=time.time(), finished=time.monotonic())
            return row

        return await asyncio.gather(*(scrape(url) for url in await get_worker_urls(args)))


async def consume(producer, args, settings, root):
    batches = []
    captured_indices = set()
    started = time.monotonic()
    phase = "warmup" if settings["warmup_seconds"] else "measure"
    measure_started = started
    boundaries = {"start": pipeline_observer.snapshot(producer)}
    counters = {}
    if phase == "measure":
        counters["measure_start"] = await engine_metrics(args)
        workflow.write_json(root / "benchmark-engine-counters.json", counters)
    capture_root = root / "benchmark-captures"
    capture_root.mkdir()
    with (
        (root / "benchmark-batches.jsonl").open("w", buffering=1) as batch_file,
        (root / "benchmark-responses.jsonl").open("w", buffering=1) as response_file,
    ):
        while True:
            batch_started = time.monotonic()
            output = await asyncio.wait_for(
                producer(RolloutFnTrainInput(rollout_id=len(batches), weight_version=0)),
                timeout=settings["collection_timeout"],
            )
            rows = []
            for group in output.samples:
                if len(group) != args.n_samples_per_prompt:
                    raise ValueError("Incomplete benchmark prompt group")
                for sample in group:
                    row = benchmark_metrics.sample_record(sample, require_routes=args.use_rollout_routing_replay)
                    rows.append(row)
                    response_file.write(json.dumps({"phase": phase, "batch": len(batches), **row}) + "\n")
                    if (
                        sample.index is not None
                        and 0 <= sample.index < settings["capture_samples"]
                        and sample.index not in captured_indices
                    ):
                        routes = sample.rollout_routed_experts
                        (capture_root / f"{sample.index:08d}.json").write_text(
                            json.dumps(
                                {
                                    **row,
                                    "tokens": sample.tokens,
                                    "logprobs": sample.rollout_log_probs,
                                    "route_sha256": hashlib.sha256(routes.tobytes()).hexdigest()
                                    if routes is not None
                                    else None,
                                    "route_shape": list(routes.shape) if routes is not None else None,
                                }
                            )
                            + "\n"
                        )
                        captured_indices.add(sample.index)
            finished = time.monotonic()
            batch = {
                "phase": phase,
                "started": batch_started,
                "finished": finished,
                "finished_unix": time.time(),
                "groups": len(output.samples),
                "samples": len(rows),
                "response_tokens": sum(row["response_tokens"] for row in rows),
                "route_bytes": sum(row["route_bytes"] for row in rows),
                "metrics": output.metrics,
            }
            batches.append(batch)
            batch_file.write(json.dumps(batch) + "\n")
            logger.info("Benchmark collection: %s", {key: value for key, value in batch.items() if key != "metrics"})
            del output, rows
            if phase == "warmup" and finished - started >= settings["warmup_seconds"]:
                phase, measure_started = "measure", finished
                boundaries["measure_start"] = pipeline_observer.snapshot(producer)
                counters["measure_start"] = await engine_metrics(args)
                workflow.write_json(root / "benchmark-engine-counters.json", counters)
            elif phase == "measure" and finished - measure_started >= settings["measure_seconds"]:
                break
    boundaries["measure_end"] = pipeline_observer.snapshot(producer)
    counters["measure_end"] = await engine_metrics(args)
    workflow.write_json(root / "benchmark-engine-counters.json", counters)
    workflow.write_json(root / "benchmark-boundaries.json", boundaries)
    return benchmark_metrics.summarize(batches, args.rollout_num_gpus)


async def run(args, settings, root):
    if settings["score_reference"]:
        observer = asyncio.create_task(benchmark_resources.observe_resources(root))
        try:
            score = load_function("open_instruct.miles.execution.benchmark_core_reference.run")
            result = await asyncio.to_thread(score, args, settings, root)
            workflow.write_json(root / "benchmark-summary.json", result)
            return result
        except BaseException as error:
            workflow.write_json(root / "benchmark-failure.json", {"type": type(error).__name__, "error": str(error)})
            raise
        finally:
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
    # Native parsing still checks the production serving contract. Only this
    # explicit execution path enables inference-only placement and omits trainers.
    args.debug_rollout_only = True
    args.start_rollout_id = 0
    if settings["reward_mode"] == "none":
        args.custom_rm_path = "open_instruct.miles.execution.benchmark.no_reward"
    started = time.monotonic()
    workers = inference = producer = source = engine_observer = None
    observer = asyncio.create_task(benchmark_resources.observe_resources(root))
    failure = None
    result = None
    try:
        workflow.write_json(
            root / "benchmark-model-config.json", json.loads((Path(args.hf_checkpoint) / "config.json").read_text())
        )
        workflow.write_json(root / "benchmark-native-options.json", snapshot_options.snapshot())
        startup_cache.prepare(args, application_root=Path(__file__).resolve().parents[3])
        workers = wiring.launch_worker_manager(args, transform_specs=partial(inference_specs, args))
        inference = InferenceController(args)
        await inference.init()
        init_http_client(args)
        workflow.write_json(root / "benchmark-server-start.json", await server_info(args))
        startup_seconds = time.monotonic() - started
        if settings["replay_reference"]:
            engine_observer = asyncio.create_task(
                pipeline_observer.observe_engines(SimpleNamespace(args=args), get_worker_urls)
            )
            result = await benchmark_replay.run(args, settings, root)
        else:
            source = load_function(args.data_source_path)(args)
            producer = load_rollout_function(
                RolloutFnConstructorInput(args=args, data_source=source), args.rollout_function_path
            )
            startup_seconds = time.monotonic() - started
            result = await consume(producer, args, settings, root)
        result["startup_seconds"] = startup_seconds
        result["reward_mode"] = settings["reward_mode"]
        # Draining long generations can outlive the job timeout. Preserve the
        # completed measurement independently of successful worker teardown.
        workflow.write_json(
            root / "benchmark-measurement.json", {"status": "measurement_complete_shutdown_pending", "summary": result}
        )
        workflow.write_json(root / "benchmark-server-end.json", await server_info(args))
    except BaseException as error:
        failure = error
        workflow.write_json(root / "benchmark-failure.json", {"type": type(error).__name__, "error": str(error)})
        raise
    finally:
        drain_started = time.monotonic()
        if engine_observer is not None:
            engine_observer.cancel()
            await asyncio.gather(engine_observer, return_exceptions=True)
        cleanup_error = None
        for component, operation, timeout in (
            (producer, lambda: producer.shutdown(), settings["drain_timeout"]),
            (inference, lambda: inference.dispose(), 60),
            (workers, lambda: workers.dispose.remote(), 120),
        ):
            if component is not None:
                try:
                    await asyncio.wait_for(operation(), timeout=timeout)
                except BaseException as error:
                    cleanup_error = cleanup_error or error
                    logger.exception("Benchmark cleanup failed")
        observer.cancel()
        observed = await asyncio.gather(observer, return_exceptions=True)
        if observed and isinstance(observed[0], Exception):
            logger.warning("Resource observations unavailable: %s", observed[0])
        if source is not None:
            source.close()
        if producer is not None:
            workflow.write_json(root / "benchmark-shutdown.json", pipeline_observer.snapshot(producer))
        await startup_cache.finish(args, success=failure is None and cleanup_error is None)
        ray.shutdown()
        if failure is None and cleanup_error is not None:
            raise cleanup_error
        if result is not None:
            result["drain_seconds"] = time.monotonic() - drain_started
            workflow.write_json(root / "benchmark-summary.json", result)
    return result
