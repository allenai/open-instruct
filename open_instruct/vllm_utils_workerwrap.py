# This class runs in vLLM worker processes; imports are inside methods.


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
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        import os

        import torch

        # Allow fp32 dtype for lm_head when fp32 mode is enabled (permanent mode)
        fp32_mode = os.environ.get("OPEN_INSTRUCT_FP32_LM_HEAD", "")
        is_lm_head = name.startswith("lm_head.")
        expected_dtype = "torch.float32" if (is_lm_head and fp32_mode == "2") else str(self.model_config.dtype)

        # Also allow model dtype for lm_head (backward compatibility with cache mode)
        if str(dtype) != expected_dtype and not (is_lm_head and str(dtype) == str(self.model_config.dtype)):
            raise AssertionError(f"mismatch dtype: src {dtype}, expected {expected_dtype}")

        # Use the actual dtype from trainer for the receive buffer
        recv_dtype = torch.float32 if dtype == "torch.float32" else self.model_config.dtype
        weight = torch.empty(shape, dtype=recv_dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        self.model_runner.model.load_weights(weights=[(name, weight)])
        self._sync_fp32_lm_head(name)

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        import os

        import torch

        from open_instruct.vllm_utils import get_physical_gpu_id

        # Allow fp32 dtype for lm_head when fp32 mode is enabled (permanent mode)
        fp32_mode = os.environ.get("OPEN_INSTRUCT_FP32_LM_HEAD", "")
        is_lm_head = name.startswith("lm_head.")
        expected_dtype = "torch.float32" if (is_lm_head and fp32_mode == "2") else str(self.model_config.dtype)

        # Also allow model dtype for lm_head (backward compatibility with cache mode)
        if str(dtype) != expected_dtype and not (is_lm_head and str(dtype) == str(self.model_config.dtype)):
            raise AssertionError(f"mismatch dtype: src {dtype}, expected {expected_dtype}")

        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        func, args = handle
        list_args = list(args)
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
        weight = func(*list_args)
        self.model_runner.model.load_weights(weights=[(name, weight)])
        self._sync_fp32_lm_head(name)
        torch.cuda.synchronize()

    def _sync_fp32_lm_head(self, name):
        """Sync fp32 lm_head weights after weight broadcast from trainer.

        Called after weight updates to keep the fp32 LM head in sync.
        Two modes controlled by OPEN_INSTRUCT_FP32_LM_HEAD env var:
        - "1" (cache mode): Keep bf16 weights, maintain separate fp32 cache
        - "2" (permanent mode): Convert lm_head weight to fp32 in-place
                               (falls back to cache mode if weights are tied)
        """
        import os

        import torch

        fp32_mode = os.environ.get("OPEN_INSTRUCT_FP32_LM_HEAD", "")
        if fp32_mode not in ("1", "2"):
            return
        if not isinstance(name, str) or not name.endswith(".weight"):
            return

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner is not None else None
        if model is None:
            return

        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            return

        weight = getattr(lm_head, "weight", None)
        if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
            return

        quant_method = getattr(lm_head, "quant_method", None)
        if quant_method is not None and quant_method.__class__.__name__ != "UnquantizedEmbeddingMethod":
            return

        try:
            module_name, param_name = name.rsplit(".", 1)
            module = model.get_submodule(module_name)
            param = getattr(module, param_name, None)
        except Exception:
            return

        if param is not weight:
            return

        # Check if weights are tied with embed_tokens (common in LLMs)
        # If tied, we can't convert in-place as it would affect the embedding layer
        weights_are_tied = self._check_weights_tied(model, lm_head)

        if fp32_mode == "2" and not weights_are_tied:
            # Permanent mode: convert lm_head weight to fp32 in-place
            # Only safe when weights are not tied to embed_tokens
            if weight.dtype != torch.float32:
                lm_head.weight.data = weight.float()
        else:
            # Cache mode (or permanent mode with tied weights):
            # Maintain separate fp32 cache to avoid affecting embed_tokens
            fp32_weight = getattr(lm_head, "_open_instruct_fp32_weight", None)
            if (
                isinstance(fp32_weight, torch.Tensor)
                and fp32_weight.shape == weight.shape
                and fp32_weight.device == weight.device
            ):
                fp32_weight.copy_(weight)
            else:
                lm_head._open_instruct_fp32_weight = weight.float()

    def _check_weights_tied(self, model, lm_head):
        """Check if lm_head weights are tied with the input embedding layer."""
        import torch

        lm_head_weight = getattr(lm_head, "weight", None)
        if not isinstance(lm_head_weight, torch.Tensor):
            return False

        # Try common paths for embed_tokens in various model architectures
        embed_paths = [
            "model.embed_tokens",  # Qwen, Llama, Mistral
            "transformer.wte",  # GPT-2 style
            "embed_tokens",  # Some models
        ]

        for path in embed_paths:
            try:
                parts = path.split(".")
                module = model
                for part in parts:
                    module = getattr(module, part, None)
                    if module is None:
                        break
                if module is not None:
                    embed_weight = getattr(module, "weight", None)
                    # Check if they share the same underlying storage
                    if isinstance(embed_weight, torch.Tensor) and lm_head_weight.data_ptr() == embed_weight.data_ptr():
                        return True
            except Exception:
                continue

        return False
