"""Driver lifecycle for the opt-in engine-drain publisher."""

import asyncio
import uuid
from typing import Any, cast

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from open_instruct.miles.publication.engine_delivery import EngineDelivery


class RollingPublication:
    def __init__(self, args, learner, manager, inference):
        self.args, self.learner, self.manager = args, learner, manager
        self.inference = inference
        self._window_task = None
        self._publication_lock = asyncio.Lock()
        self._publication_serial = 0
        self.deliveries = {}
        self.version: int | None = None

    async def initialize(self):
        info = await self.inference.start_update_weights()
        try:
            await self._initialize(info)
        except BaseException:
            await self.inference.abort_update_weights()
            raise
        await self.inference.end_update_weights(info.snapshot_cell_id_to_hashes)
        if self.args.check_weight_update_equal:
            await self.inference.check_weights(
                action="compare",
                allow_quant_error=self.args.check_weight_update_allow_quant_error,
                selector=self.args.check_weight_update_selector,
                skip_list=self.args.check_weight_update_skip_list,
            )
        await self.inference.prepare_rollout(0)

    async def _initialize(self, info):
        if not info.rollout_engines or any(count != 1 for count in info.engine_gpu_counts):
            raise ValueError("engine drain requires resident TP1 engines")
        locations = await self.learner.execute_workers("delivery_location")
        source = locations[0]
        self.version = source["version"]
        # The legacy initial publication verifies the model mapping. Retire its
        # fleet communicator before any independent update groups are created.
        await self.learner.execute_workers("close_weight_transport")
        urls, incarnations = {}, {}
        for index, engine in enumerate(info.rollout_engines):
            identity = str(index)
            urls[identity] = engine.server_url
            incarnations[identity] = uuid.uuid4().hex
            self.deliveries[identity] = (
                cast(Any, EngineDelivery)
                .options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(source["node_id"], soft=False),
                    runtime_env={
                        "env_vars": {
                            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                            "CUDA_VISIBLE_DEVICES": source["cuda_visible_devices"],
                        }
                    },
                )
                .remote(
                    engine,
                    source["cuda_visible_devices"],
                    source["device_index"],
                    self.args.olmo_core.engine_update_timeout,
                )
            )
        timeout = self.args.olmo_core.engine_update_timeout
        await asyncio.wait_for(
            asyncio.gather(*(worker.connect.remote() for worker in self.deliveries.values())), timeout
        )
        snapshot = (await self.learner.execute_workers("capture_weight_snapshot"))[0]
        await asyncio.wait_for(
            asyncio.gather(*(w.deliver.remote(snapshot) for w in self.deliveries.values())), timeout
        )
        await self.manager.core_engine_drain.remote(
            "initialize", urls=urls, incarnations=incarnations, deliveries=self.deliveries, version=self.version
        )

    async def optimizer_step_completed(self):
        # The consuming clock advances before snapshot backpressure/capture, so
        # admission never mistakes a delayed snapshot for a delayed optimizer.
        if self.version is None:
            raise RuntimeError("rolling publication has not been initialized")
        self.version += 1
        return await self.manager.core_engine_drain.remote("step", version=self.version)

    async def publish(self):
        await self.manager.core_engine_drain.remote("capacity")
        snapshot = (await self.learner.execute_workers("capture_weight_snapshot"))[0]
        if snapshot.version != self.version:
            raise RuntimeError("captured snapshot disagrees with the completed optimizer step")
        async with self._publication_lock:
            if self._window_task is not None and self._window_task.done():
                await self._window_task
                self._window_task = None
            info = await self.inference.start_update_weights() if self._window_task is None else None
            try:
                result = await self.manager.core_engine_drain.remote("publish", snapshot=snapshot)
            except BaseException:
                if info is not None:
                    await self.inference.abort_update_weights()
                raise
            self._publication_serial += 1
            if info is not None:
                self._window_task = asyncio.create_task(self._finish_window(info))
            return result

    async def _finish_window(self, info):
        # Overlapping deliveries share one window: health checks must remain
        # paused until every engine has finished loading. Call quiesce() before
        # any controller operation that needs its lock (evaluation/offload/etc.).
        # A failed update is terminal; abort releases the lock but must not
        # resume health checks or admit an engine with incomplete weights.
        released = False
        try:
            while True:
                serial = self._publication_serial
                await self.manager.core_engine_drain.remote("barrier")
                async with self._publication_lock:
                    if serial == self._publication_serial:
                        released = True
                        await self.inference.end_update_weights(info.snapshot_cell_id_to_hashes)
                        await self.inference.prepare_rollout(0)
                        return
        except BaseException:
            if not released:
                await self.inference.abort_update_weights()
            raise

    async def quiesce(self):
        result = await self.manager.core_engine_drain.remote("quiesce")
        if self._window_task is not None:
            await self._window_task
            self._window_task = None
        return result

    async def resume(self):
        return await self.manager.core_engine_drain.remote("resume")

    async def close(self, *, failed):
        try:
            if not failed:
                await self.quiesce()
                await asyncio.wait_for(
                    asyncio.gather(*(w.close.remote() for w in self.deliveries.values())),
                    self.args.olmo_core.engine_update_timeout,
                )
        finally:
            if self._window_task is not None:
                self._window_task.cancel()
                await asyncio.gather(self._window_task, return_exceptions=True)
                self._window_task = None
            # Failed receivers are never reopened. Terminating the delivery
            # processes also bounds any remote RPC still running after timeout.
            for worker in self.deliveries.values():
                ray.kill(worker, no_restart=True)
