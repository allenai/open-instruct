class WorkerWrap:
    def __init__(self):
        self._expected_weight_dtypes = {}
        self._original_model_dtype = None

    def _resolve_torch_dtype(self, dtype):
        import torch

        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            candidate = dtype.split(".", 1)[1] if dtype.startswith("torch.") else dtype
            if hasattr(torch, candidate):
                return getattr(torch, candidate)
        raise ValueError(f"Unsupported dtype received for weight update: {dtype!r}")

    def _update_expected_dtype(self, name, dtype):
        prev_dtype = self._expected_weight_dtypes.get(name)
        if prev_dtype is not None and prev_dtype != dtype:
            print(
                f"[vLLM Worker] dtype change detected for param '{name}': "
                f"{prev_dtype} -> {dtype}. Updating expected dtype."
            )
        self._expected_weight_dtypes[name] = dtype
        if self._original_model_dtype is None and hasattr(self, "model_config"):
            self._original_model_dtype = self.model_config.dtype

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
    ):
        """Init torch process group for model weights update"""
        from datetime import timedelta

        import torch

        from open_instruct.vllm_utils3 import init_process_group

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
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        import torch

        resolved_dtype = self._resolve_torch_dtype(dtype)
        self._update_expected_dtype(name, resolved_dtype)
        weight = torch.empty(shape, dtype=resolved_dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        import torch

        from open_instruct.vllm_utils3 import get_physical_gpu_id

        resolved_dtype = self._resolve_torch_dtype(dtype)
        self._update_expected_dtype(name, resolved_dtype)
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
