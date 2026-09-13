"""Driver lifecycle for the opt-in engine-drain publisher."""

import asyncio
import uuid
from typing import Any, cast

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from open_instruct.miles.engine_delivery import EngineDelivery


class RollingPublication:
    def __init__(self, args, learner, manager):
        self.args, self.learner, self.manager = args, learner, manager
        self.deliveries = {}
        self.version = None

    async def initialize(self):
        info = await self.manager.get_updatable_engines_and_lock.remote()
        if not info.rollout_engines or any(count != 1 for count in info.engine_gpu_counts):
            raise ValueError("engine drain requires resident TP1 engines")
        locations = await self.learner._broadcast("delivery_location")
        source = locations[0]
        self.version = source["version"]
        # The legacy initial publication verifies the model mapping. Retire its
        # fleet communicator before any independent update groups are created.
        await self.learner._broadcast("close_weight_transport")
        urls, incarnations = {}, {}
        for index, engine in enumerate(info.rollout_engines):
            identity = str(index)
            topology = await engine.get_topology_info.remote()
            if topology["node_rank"] != 0:
                raise ValueError("engine drain requires one TP1 worker per engine")
            urls[identity] = topology["url"]
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
        snapshot = (await self.learner._broadcast("capture_weight_snapshot"))[0]
        await asyncio.wait_for(
            asyncio.gather(*(w.deliver.remote(snapshot) for w in self.deliveries.values())), timeout
        )
        if self.args.check_weight_update_equal:
            await self.manager.check_weights.remote(
                action="compare",
                allow_quant_error=self.args.check_weight_update_allow_quant_error,
                selector=self.args.check_weight_update_selector,
                skip_list=self.args.check_weight_update_skip_list,
            )
        await self.manager.core_engine_drain.remote(
            "initialize", urls=urls, incarnations=incarnations, deliveries=self.deliveries, version=self.version
        )

    async def publish(self):
        await self.manager.core_engine_drain.remote("capacity")
        snapshot = (await self.learner._broadcast("capture_weight_snapshot"))[0]
        self.version = snapshot.version
        return await self.manager.core_engine_drain.remote("publish", snapshot=snapshot)

    async def quiesce(self):
        return await self.manager.core_engine_drain.remote("quiesce")

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
            # Failed receivers are never reopened. Terminating the delivery
            # processes also bounds any remote RPC still running after timeout.
            for worker in self.deliveries.values():
                ray.kill(worker, no_restart=True)
