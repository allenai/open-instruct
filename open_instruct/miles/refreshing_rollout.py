"""Keep unfinished single-turn generations alive across direct policy publication."""

import asyncio
import time
import uuid

import httpx
from miles.rollout.base_types import GenerateFnOutput, RolloutFnEvalOutput
from miles.rollout.generate_hub.single_turn import generate as single_turn_generate
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.utils.http_utils import post

from open_instruct import logger_utils
from open_instruct.miles import pipeline_observer, policy_refresh, sibling_timing
from open_instruct.miles.async_rollout import ManagedFullyAsyncRolloutFn

logger = logger_utils.setup_logger(__name__)


class RefreshingRolloutFn(ManagedFullyAsyncRolloutFn):
    """Publication gates new groups; evaluation and teardown drain owned requests.

    SGLang owns request retraction, state invalidation and continuation. Core's
    actor publishes while source weights are fixed and reopens engines only
    after every bucket and cache flush succeeds. No CPU snapshot is involved.
    """

    def __init__(self, input):
        super().__init__(input)
        self.state.generate_function = self._generate_response
        self._refreshing = False
        self._boundary_capacity = None
        self._refresh_started = None

    async def begin_refresh(self):
        if self._refreshing or self._publication_paused or self._stopping or self._interrupted():
            raise RuntimeError("Cannot start policy refresh while paused, stopping, or recovering")
        self._refreshing = True
        self._refresh_started = time.monotonic()
        self._producer_resumed.clear()
        # Do not join/cancel active requests or requeue their prompts. The actor
        # pauses engines at the scheduler boundary after this acknowledgement.
        logger.info("Policy refresh admission closed: owned_groups=%d", len(self._producing_groups))

    async def end_refresh(self):
        if not self._refreshing or self._interrupted():
            raise RuntimeError("Cannot reopen policy refresh before successful publication")
        self._refreshing = False
        self._resume_if_buffer_allows()
        logger.info(
            "Policy refresh admission reopened: seconds=%.3f owned_groups=%d",
            time.monotonic() - self._refresh_started,
            len(self._producing_groups),
        )

    async def _generate_group(self, prompt_group):
        if any(s.generate_function_path or s.response_length or s.multimodal_inputs for s in prompt_group):
            raise ValueError("Policy refresh supports fresh, single-turn text requests without custom generators")
        observed = getattr(self.args.olmo_core, "pipeline_observation_interval", 0) > 0
        records = sibling_timing.start_group(prompt_group) if observed else None
        outcome = "completed"
        try:
            return await super()._generate_group(prompt_group)
        except BaseException as error:
            outcome = type(error).__name__
            raise
        finally:
            if records is not None:
                sibling_timing.write_group(self.args, records, outcome)

    async def _generate_response(self, input):
        sibling_timing.admitted(input.sample)
        outcome = "completed"
        try:
            return await self._generate_response_impl(input)
        except BaseException as error:
            outcome = type(error).__name__
            raise
        finally:
            sibling_timing.finished(input.sample, outcome)

    async def _generate_response_impl(self, input):
        sample = input.sample
        payload, halt = compute_request_payload(
            input.args,
            input_ids=compute_prompt_ids_from_sample(input.state, sample),
            sampling_params=input.sampling_params,
        )
        if payload is None:
            raise ValueError(f"Policy refresh prompt has no response-token budget ({halt}); increase context length")
        payload["rid"] = uuid.uuid4().hex
        timing = sibling_timing.sample_record(sample)
        if timing is not None:
            timing["request_id"] = payload["rid"]
        # With the qualified MILES router the complete response metadata survives.
        # Ambiguous failures must not transparently resample under newer weights.
        url = f"http://{input.args.sglang_router_ip}:{input.args.sglang_router_port}/generate"
        timeout = self.args.olmo_core.refresh_request_timeout
        started = time.monotonic()
        logger.info(
            "Policy refresh request submitted: request=%s group=%s sample=%s url=%s prompt_tokens=%d",
            payload["rid"],
            sample.group_index,
            sample.index,
            url,
            len(payload.get("input_ids", [])),
        )
        try:
            output = await asyncio.wait_for(
                post(url, payload, max_retries=1, headers={"x-miles-request-id": payload["rid"]}), timeout
            )
        except httpx.HTTPError as error:
            logger.exception(
                "Policy refresh HTTP failure: request=%s group=%s sample=%s url=%s elapsed_seconds=%.3f "
                "error=%s status=%s attempts=1 delivery=unknown; correlate request with miles_router logs. "
                "No automatic resampling; propagating failure to the producer.",
                payload["rid"],
                sample.group_index,
                sample.index,
                url,
                time.monotonic() - started,
                type(error).__name__,
                getattr(getattr(error, "response", None), "status_code", None),
            )
            raise
        except TimeoutError as error:
            raise TimeoutError(
                f"Policy refresh request {payload['rid']} exceeded core.refresh_request_timeout={timeout:g}s. "
                "This covers serving queue time, generation and refresh pauses. Check engine progress and "
                "admission pressure; increase the request timeout for deliberately long responses. "
                "core.engine_drain_timeout only controls eval/export/shutdown draining."
            ) from error
        sibling_timing.response_received(sample, payload["rid"], output.get("meta_info", {}))
        if output.get("meta_info", {}).get("finish_reason", {}).get("type") not in ("stop", "length"):
            raise RuntimeError("Policy refresh request did not finish; refusing a partial training sample")
        await update_sample_from_response(input.args, sample, payload, output)
        record = policy_refresh.record_response(sample, output["meta_info"])
        logger.info(
            "Policy refresh response: request=%s tokens=%d spans=%s replay_version=%d",
            payload["rid"],
            sample.response_length,
            record["spans"],
            record["replay_version"],
        )
        return GenerateFnOutput(samples=sample)

    async def prepare_publication(self):
        """Quiesce for eval/export/teardown, allowing only owned completions to drain."""
        if self._refreshing:
            raise RuntimeError("Policy refresh did not finish; engines must not be reused")
        self._producer_resumed.clear()
        self._publication_paused = True
        if self._output is not None and self._boundary_capacity is None:
            self._boundary_capacity = await self._output.reserve_drain_capacity(len(self._producing_groups))
        if self._worker is not None:
            await asyncio.wait_for(self._wait_until_idle(), self.args.olmo_core.engine_drain_timeout)
        return []

    async def finish_publication(self):
        if self._refreshing or self._interrupted():
            raise RuntimeError("Cannot resume generation before successful publication")
        if self._boundary_capacity is not None:
            await self._output.restore_capacity(self._boundary_capacity)
            self._boundary_capacity = None
        self._publication_paused = False
        self._resume_if_buffer_allows()

    def _resume_if_buffer_allows(self):
        if self._publication_paused or self._refreshing or self._stopping:
            return
        delegate = self._output._delegate if self._output is not None else None
        if delegate is None or len(delegate._buffer) <= delegate._capacity:
            self._producer_resumed.set()

    async def _next_group(self, current_version):
        # Discarding an over-age completion also frees capacity, so poll while
        # the delegate waits for a replacement after a lifecycle drain.
        pending = asyncio.create_task(super()._next_group(current_version))
        try:
            while not pending.done():
                self._resume_if_buffer_allows()
                await asyncio.wait({pending}, timeout=0.25)
            result = pending.result()
            self._resume_if_buffer_allows()
            return result
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    async def _call_eval(self, input):
        await self.prepare_publication()
        generator = self.state.generate_function
        self.state.generate_function = single_turn_generate
        try:
            results = await run_eval_datasets(self.state, self._eval_prompt_dataset_cache)
            return RolloutFnEvalOutput(data=results)
        finally:
            self.state.generate_function = generator
            await self.finish_publication()

    async def shutdown(self):
        if self._shutdown_complete:
            return
        pipeline_observer.write_lifecycle(self, "shutdown_start")
        quiesced = False
        try:
            await self.prepare_publication()
            quiesced = True
        finally:
            # On failed publication, cancel HTTP ownership without reopening any
            # engine. Driver teardown terminates servers after transport cleanup.
            self._stopping = True
            self._stop_requested.set()
            if self._worker is not None:
                self._worker.cancel()
                await asyncio.gather(self._worker, return_exceptions=True)
            self._shutdown_complete = True
            pipeline_observer.write_lifecycle(self, "shutdown_complete" if quiesced else "shutdown_incomplete")
