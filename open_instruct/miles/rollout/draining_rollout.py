"""MILES producer with group-pinned dispatch and independently drained engines."""

import asyncio
import json
import os
from pathlib import Path

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
from open_instruct.miles.publication import policy_versions
from open_instruct.miles.publication.engine_drain import Engine, EngineDrain
from open_instruct.miles.rollout import pipeline_observer
from open_instruct.miles.rollout.async_rollout import ManagedFullyAsyncRolloutFn

logger = logger_utils.setup_logger(__name__)


class DrainingRolloutFn(ManagedFullyAsyncRolloutFn):
    def __init__(self, input):
        super().__init__(input)
        self.controller = None
        self._assignments = {}
        self._boundary_capacity = None
        self._urls = {}
        self._deliveries = {}
        # Qualification-only: hold one reserved request before HTTP admission.
        # This deliberately exercises queued/unsent ownership, not slow grading.
        self._probe_delay = int(os.environ.get("OI_MILES_ENGINE_DRAIN_TEST_DELAY_SECONDS", "0"))
        if not 0 <= self._probe_delay <= 120:
            raise ValueError("OI_MILES_ENGINE_DRAIN_TEST_DELAY_SECONDS must be an integer in [0, 120]")
        self._probe_used = False
        self._probe_after_version = 0

    def _event(self, record):
        logger.info("Core engine drain: %s", json.dumps(record, sort_keys=True))
        if self.args.save:
            path = Path(self.args.save) / "engine_drain.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")

    async def engine_drain_control(self, operation, **kwargs):
        if operation == "initialize":
            if self.controller is not None:
                raise RuntimeError("engine drain is already initialized")
            self._probe_after_version = kwargs["version"]
            self._urls = kwargs["urls"]
            self._deliveries = kwargs["deliveries"]
            core = self.args.olmo_core
            self.controller = EngineDrain(
                [Engine(identity, kwargs["incarnations"][identity], kwargs["version"]) for identity in self._urls],
                self._deliver,
                max_lag=core.max_policy_lag,
                capacity=core.snapshot_capacity,
                drain_timeout=core.engine_drain_timeout,
                update_timeout=core.engine_update_timeout,
                event=self._event,
            )
            self.state.generate_function = self._generate_response
        elif operation == "capacity":
            await self.controller.wait_capacity()
        elif operation == "publish":
            self.controller.set_step(kwargs["snapshot"].version)
            self.controller.publish(kwargs["snapshot"])
        elif operation == "step":
            self.controller.set_step(kwargs["version"])
        elif operation == "barrier":
            await self.controller.barrier()
        elif operation == "quiesce":
            await self.prepare_publication()
        elif operation == "resume":
            await self.finish_publication()
        else:
            raise ValueError(f"unknown engine drain operation: {operation}")
        return self.controller.status()

    async def _deliver(self, identity, snapshot):
        result = await self._deliveries[identity].deliver.remote(snapshot)
        self.controller.emit("delivery_measured", engine=identity, **result)
        return result["version"]

    async def _generate_group(self, prompt_group):
        assignment = await self.controller.reserve(prompt_group[0].group_index, len(prompt_group))
        for sample, request in zip(prompt_group, assignment.requests):
            if sample.generate_function_path or sample.response_length:
                error = ValueError("engine drain supports fresh single-turn requests without per-sample generators")
                self.controller.fail(assignment.engine, error)
                raise error
            self._assignments[id(sample)] = (assignment, request)
        try:
            result = await super()._generate_group(prompt_group)
            self.controller.graded(assignment)
            return result
        except BaseException as error:
            self.controller.fail(assignment.engine, error)
            raise
        finally:
            for sample in prompt_group:
                self._assignments.pop(id(sample), None)

    async def _generate_response(self, input):
        sample = input.sample
        assignment, request = self._assignments[id(sample)]
        prompt_ids = compute_prompt_ids_from_sample(input.state, sample)
        payload, halt = compute_request_payload(
            input.args,
            input_ids=prompt_ids,
            sampling_params=input.sampling_params,
            multimodal_inputs=sample.multimodal_inputs,
        )
        if payload is None:
            # No engine call took place; this cannot produce a training response.
            raise ValueError(f"engine-drain prompt has no response-token budget ({halt}); increase context length")
        payload["rid"] = request
        if (
            self._probe_delay
            and not self._probe_used
            and assignment.engine == "1"
            and assignment.version > self._probe_after_version
        ):
            self._probe_used = True
            await self._delay_until_draining(assignment, request)
        self.controller.emit(
            "request_sent",
            engine=assignment.engine,
            group=assignment.group,
            request=request,
            version=assignment.version,
        )
        # Bypass the load-balancing router: it has no policy-version reservations.
        # Never retry an ambiguous HTTP outcome under a different policy/engine.
        try:
            output = await asyncio.wait_for(
                post(f"{self._urls[assignment.engine]}/generate", payload, max_retries=1),
                self.args.olmo_core.engine_drain_timeout,
            )
        except BaseException as error:
            self.controller.emit(
                "request_outcome_unknown",
                engine=assignment.engine,
                request=request,
                error=type(error).__name__,
                client_cancelled=isinstance(error, asyncio.CancelledError),
                generated_tokens=None,
            )
            raise
        if output.get("meta_info", {}).get("finish_reason", {}).get("type") not in ("stop", "length"):
            self.controller.emit(
                "request_failed",
                engine=assignment.engine,
                request=request,
                finish_reason=output.get("meta_info", {}).get("finish_reason"),
            )
            raise RuntimeError("engine-drain request did not complete; partial-policy continuation is unsupported")
        await update_sample_from_response(input.args, sample, payload, output)
        if set(policy_versions.versions(sample.weight_versions)) != {assignment.version}:
            raise ValueError(f"engine returned unexpected behavior versions: {sample.weight_versions}")
        self.controller.decoded(assignment, request, version=assignment.version, tokens=sample.response_length)
        sample.metadata = {
            **(sample.metadata or {}),
            "engine_drain": {
                "engine": assignment.engine,
                "incarnation": assignment.incarnation,
                "request": request,
                "version": assignment.version,
                "attempt": assignment.attempt,
            },
        }
        return GenerateFnOutput(samples=sample)

    async def _delay_until_draining(self, assignment, request):
        self.controller.emit(
            "qualification_hold_reserved", engine=assignment.engine, request=request, version=assignment.version
        )
        # Cold compilation can outlast a fixed delay started at admission. Gate
        # the qualification hold on actual admission closure, so the test really
        # leaves one owned request behind while the peer receives newer weights.
        async with asyncio.timeout(
            self.args.olmo_core.engine_drain_timeout + self.args.olmo_core.engine_update_timeout
        ):
            while self.controller.engines[assignment.engine].state == "serving":
                self.controller.check()
                await asyncio.sleep(0.1)
        self.controller.check()
        self.controller.emit(
            "qualification_delay_started",
            engine=assignment.engine,
            request=request,
            seconds=self._probe_delay,
            version=assignment.version,
        )
        await asyncio.sleep(self._probe_delay)
        self.controller.emit("qualification_delay_finished", engine=assignment.engine, request=request)

    async def _next_group(self, current_version):
        self.controller.check()
        # A dead publication must interrupt a waiting dequeue, even if no request
        # remains to expose that error through the producer's worker task.
        pending = asyncio.create_task(super()._next_group(current_version))
        try:
            while not pending.done():
                self.controller.check()
                # The delegate can discard stale groups without returning an
                # entry. That still frees capacity and must wake the producer.
                self._resume_if_buffer_allows()
                await asyncio.wait({pending}, timeout=0.25)
            result = pending.result()
            self._resume_if_buffer_allows()
            return result
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    async def prepare_publication(self):
        if self.controller is None:
            return await super().prepare_publication()
        self.controller.check()
        self._producer_resumed.clear()
        # Stop admission at the producer first. Expand only enough to store all
        # already-owned groups, including completed tasks awaiting buffer insertion.
        # _active_tasks excludes those completions; the ownership ledger does not.
        if self._output is not None and self._boundary_capacity is None:
            self._boundary_capacity = await self._output.reserve_drain_capacity(len(self._producing_groups))
        if self._worker is not None:
            await asyncio.wait_for(
                self._wait_until_idle(),
                self.args.olmo_core.engine_drain_timeout + self.args.olmo_core.engine_update_timeout,
            )
        await self.controller.pause()
        self._publication_paused = True
        self.controller.emit("lifecycle_quiesced", **self.controller.status())
        return []

    async def finish_publication(self):
        if self.controller is None:
            return await super().finish_publication()
        if self._boundary_capacity is not None:
            await self._output.restore_capacity(self._boundary_capacity)
            self._boundary_capacity = None
        self.controller.resume()
        self._publication_paused = False
        self._resume_if_buffer_allows()

    def _resume_if_buffer_allows(self):
        # A boundary may retain more completions than the normal queue capacity.
        # Consume that excess before admitting another generation wave, otherwise
        # frequent checkpoints could grow the retained queue on every boundary.
        if self._publication_paused:
            return
        delegate = self._output._delegate if self._output is not None else None
        if delegate is None or len(delegate._buffer) <= delegate._capacity:
            self._producer_resumed.set()

    async def _call_eval(self, input):
        if self.controller is None:
            return await super()._call_eval(input)
        await self.prepare_publication()
        try:
            # All engines now acknowledge the same snapshot. The ordinary eval
            # generator/router is safe here; training admission remains closed.
            return await self._quiescent_eval(input)
        finally:
            await self.finish_publication()

    async def _quiescent_eval(self, input):
        generator = self.state.generate_function
        self.state.generate_function = single_turn_generate
        try:
            results = await run_eval_datasets(input.generate_state or self.state, self._eval_prompt_dataset_cache)
            return RolloutFnEvalOutput(data=results)
        finally:
            self.state.generate_function = generator

    async def shutdown(self):
        if self.controller is None:
            return await super().shutdown()
        if self._shutdown_complete:
            return
        pipeline_observer.write_lifecycle(self, "shutdown_start")
        await self.prepare_publication()
        await self.controller.close()
        self._stopping = True
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
        self._shutdown_complete = True
        pipeline_observer.write_lifecycle(self, "shutdown_complete")
