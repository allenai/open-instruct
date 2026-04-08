"""GPU tests for FSDP2 weight broadcasting in vllm_utils.

To run:
    ./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_tests.sh
"""

import os
import socket
import unittest

import datasets
import ray
import torch
from olmo_core.nn.transformer import TransformerConfig
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

from open_instruct import logger_utils, vllm_utils
from open_instruct.ground_truth_utils import RewardConfig
from open_instruct.test_grpo_fast import TestGrpoFastBase
from open_instruct.utils import maybe_update_beaker_description
from open_instruct.vllm_utils import create_vllm_engines

logger = logger_utils.setup_logger(__name__)

maybe_update_beaker_description()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestFSDP2BroadcastWithVLLM(TestGrpoFastBase):
    """Integration test for FSDP2 weight broadcast with real vLLM engine on single GPU."""

    @unittest.skip("FSDP2 single-GPU broadcast has zero-storage params; needs multi-GPU")
    def test_broadcast_olmo_core_fsdp2_weights_to_vllm(self):
        """Test broadcasting OLMo-core FSDP2 model weights to a real vLLM engine."""
        tokenizer_name = "Qwen/Qwen3-0.6B"
        AutoTokenizer.from_pretrained(tokenizer_name)

        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        os.environ["MASTER_ADDR"] = master_address
        os.environ["MASTER_PORT"] = str(master_port)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)

        config = TransformerConfig.llama_like(d_model=256, vocab_size=1000, n_layers=2, n_heads=4)
        model = config.build(init_device="cuda")
        model.apply_fsdp()

        param_prompt_Q = ray_queue.Queue(maxsize=100)
        inference_results_Q = ray_queue.Queue(maxsize=100)
        eval_results_Q = ray_queue.Queue(maxsize=100)
        self._ray_queues.extend([param_prompt_Q, inference_results_Q, eval_results_Q])

        pg = placement_group([{"GPU": 1, "CPU": 1}], strategy="PACK")
        ray.get(pg.ready())

        train_dataset = datasets.Dataset.from_dict(
            {"ground_truth": [["4"]], "dataset": ["test"], "prompt": ["test"], "index": [0]}
        )

        engines = create_vllm_engines(
            num_engines=1,
            tensor_parallel_size=1,
            enforce_eager=True,
            tokenizer_name_or_path=tokenizer_name,
            pretrain=tokenizer_name,
            revision="main",
            seed=42,
            enable_prefix_caching=False,
            max_model_len=512,
            vllm_gpu_memory_utilization=0.3,
            single_gpu_mode=True,
            pg=pg,
            prompt_queue=param_prompt_Q,
            results_queue=inference_results_Q,
            eval_results_queue=eval_results_Q,
            reward_config=RewardConfig(),
            train_dataset=train_dataset,
        )
        ray.get(engines[0].ready.remote())

        with socket.socket() as sock:
            sock.bind(("", 0))
            weight_transfer_port = sock.getsockname()[1]

        world_size = 2
        ray.get(
            engines[0].init_weight_transfer_engine.remote(
                master_address=master_address, master_port=weight_transfer_port, rank_offset=1, world_size=world_size
            )
        )

        model_update_group = NCCLWeightTransferEngine.trainer_init(
            {"master_address": master_address, "master_port": weight_transfer_port, "world_size": world_size}
        )

        refs = vllm_utils.broadcast_weights_to_vllm(
            model=model,
            vllm_engines=engines,
            model_update_group=model_update_group,
            name_mapper=None,
            gather_whole_model=True,
        )

        self.assertGreater(len(refs), 0)
        ray.get(refs)

        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
