"""GPU test for the weight-sync fingerprint verification path.

Validates that after `broadcast_weights_to_vllm` pushes a learner-side model's
weights to a vLLM engine, `LLMRayActor.get_weight_fingerprint` returns values
that match the fingerprint computed directly on the trainer-side model. This
is the same mechanism that ``_log_weight_sync_fingerprints`` uses on resume
to confirm the vLLM actor weights match the learner's checkpoint-loaded
weights.

To run::

    ./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_tests.sh
"""

import os
import unittest

import datasets
import ray
import torch
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.distributed.weight_transfer.base import WeightTransferInitRequest
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

from open_instruct import logger_utils, utils, vllm_utils
from open_instruct.ground_truth_utils import RewardConfig
from open_instruct.test_grpo_fast import TestGrpoFastBase

logger = logger_utils.setup_logger(__name__)


def _fingerprint_local(model, param_names: list[str]) -> dict[str, float]:
    params_by_name = dict(model.named_parameters())
    out: dict[str, float] = {}
    for name in param_names:
        if name in params_by_name:
            out[name] = float(params_by_name[name].data.detach().float().abs().mean().item())
    return out


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestWeightSyncFingerprintGPU(TestGrpoFastBase):
    def test_fingerprints_match_after_broadcast(self):
        model_name = "Qwen/Qwen3-0.6B"
        AutoTokenizer.from_pretrained(model_name)

        master_address = ray._private.services.get_node_ip_address()
        master_port = utils.find_free_port()
        os.environ["MASTER_ADDR"] = master_address
        os.environ["MASTER_PORT"] = str(master_port)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)

        trainer_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")

        with torch.no_grad():
            for p in trainer_model.parameters():
                p.mul_(1.01).add_(0.0001)

        param_prompt_Q = ray_queue.Queue(maxsize=100)
        inference_results_Q = ray_queue.Queue(maxsize=100)
        eval_results_Q = ray_queue.Queue(maxsize=100)
        self._ray_queues.extend([param_prompt_Q, inference_results_Q, eval_results_Q])

        pg = placement_group([{"GPU": 1, "CPU": 1}], strategy="PACK")
        ray.get(pg.ready())

        train_dataset = datasets.Dataset.from_dict(
            {"ground_truth": [["4"]], "dataset": ["test"], "prompt": ["test"], "index": [0]}
        )

        engines = vllm_utils.create_vllm_engines(
            num_engines=1,
            tensor_parallel_size=1,
            enforce_eager=True,
            tokenizer_name_or_path=model_name,
            pretrain=model_name,
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

        sample_names = [name for name, _ in trainer_model.named_parameters() if "weight" in name][:4]
        self.assertGreater(len(sample_names), 0)

        trainer_fp = _fingerprint_local(trainer_model, sample_names)
        pre_sync_engine_fp = ray.get(engines[0].get_weight_fingerprint.remote(sample_names))
        logger.info(f"Pre-sync: trainer={trainer_fp} engine={pre_sync_engine_fp}")
        for name in sample_names:
            self.assertIn(name, pre_sync_engine_fp)
            self.assertNotAlmostEqual(trainer_fp[name], pre_sync_engine_fp[name], places=4)

        weight_transfer_port = utils.find_free_port()
        world_size = 2
        ray.get(
            engines[0].init_weight_transfer_engine.remote(
                WeightTransferInitRequest(
                    init_info={
                        "master_address": master_address,
                        "master_port": weight_transfer_port,
                        "rank_offset": 1,
                        "world_size": world_size,
                    }
                )
            )
        )
        model_update_group = NCCLWeightTransferEngine.trainer_init(
            {"master_address": master_address, "master_port": weight_transfer_port, "world_size": world_size}
        )
        refs = vllm_utils.broadcast_weights_to_vllm(
            model=trainer_model,
            vllm_engines=engines,
            model_update_group=model_update_group,
            name_mapper=None,
            gather_whole_model=True,
        )
        ray.get(refs)

        post_sync_engine_fp = ray.get(engines[0].get_weight_fingerprint.remote(sample_names))
        logger.info(f"Post-sync: trainer={trainer_fp} engine={post_sync_engine_fp}")
        for name in sample_names:
            self.assertIn(name, post_sync_engine_fp)
            self.assertAlmostEqual(
                trainer_fp[name],
                post_sync_engine_fp[name],
                delta=1e-2 * max(abs(trainer_fp[name]), 1e-8),
                msg=f"fingerprint mismatch for {name!r} after broadcast",
            )

        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
