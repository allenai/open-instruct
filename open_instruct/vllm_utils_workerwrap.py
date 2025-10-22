class WorkerWrap:
    def init_process_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
        use_ray=False,
        timeout_minutes=120,
        group_specs=None,
    ):
        """Init torch process group for model weights update"""
        from datetime import timedelta

        import torch

        from open_instruct.vllm_utils import init_process_group

        print("init_process_group")
        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=rank, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            print("init_process_group else")
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
                timeout=timedelta(minutes=timeout_minutes),
            )
        self._model_update_with_ray = use_ray
        self._global_rank = rank
        self._inter_node_group = None
        self._intra_node_group = None
        self._intra_group_root = None
        self._hierarchical_broadcast_enabled = False
        self._is_node_leader = False
        if not use_ray and group_specs:
            for spec in group_specs:
                members = spec.get("members", [])
                if self._global_rank not in members:
                    continue
                subgroup_rank = members.index(self._global_rank)
                subgroup_world_size = len(members)
                subgroup = init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=subgroup_world_size,
                    rank=subgroup_rank,
                    group_name=spec.get("group_name"),
                    timeout=timedelta(minutes=timeout_minutes),
                )
                self._hierarchical_broadcast_enabled = True
                if spec.get("tag") == "inter_node":
                    self._inter_node_group = subgroup
                    if self._global_rank != 0:
                        self._is_node_leader = True
                elif spec.get("tag") == "intra_node":
                    self._intra_node_group = subgroup
                    leader_rank = spec.get("leader_rank", members[0] if members else None)
                    self._intra_group_root = members.index(leader_rank) if leader_rank in members else 0
                    if leader_rank == self._global_rank:
                        self._is_node_leader = True
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        import torch

        assert str(dtype) == str(self.model_config.dtype), (
            f"mismatch dtype: src {dtype}, dst {str(self.model_config.dtype)}"
        )
        weight = torch.empty(shape, dtype=self.model_config.dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            if getattr(self, "_hierarchical_broadcast_enabled", False) and (
                self._inter_node_group is not None or self._intra_node_group is not None
            ):
                if self._inter_node_group is not None and self._is_node_leader:
                    torch.distributed.broadcast(weight, 0, group=self._inter_node_group)
                    if self._intra_node_group is not None:
                        root = self._intra_group_root if self._intra_group_root is not None else self._global_rank
                        torch.distributed.broadcast(weight, root, group=self._intra_node_group)
                elif self._intra_node_group is not None and self._intra_group_root is not None:
                    torch.distributed.broadcast(weight, self._intra_group_root, group=self._intra_node_group)
                else:
                    torch.distributed.broadcast(weight, 0, group=self._model_update_group)
            else:
                torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        import torch

        from open_instruct.vllm_utils import get_physical_gpu_id

        assert str(dtype) == str(self.model_config.dtype), (
            f"mismatch dtype: src {dtype}, dst {str(self.model_config.dtype)}"
        )
        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        func, args = handle
        list_args = list(args)
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
        weight = func(*list_args)
        self.model_runner.model.load_weights(weights=[(name, weight)])
        torch.cuda.synchronize()
