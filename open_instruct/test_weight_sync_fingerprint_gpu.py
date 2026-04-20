"""GPU test for `LLMRayActor.get_weight_fingerprint`.

Verifies that the fingerprint method (used by `_log_weight_sync_fingerprints` to
confirm vLLM actor weights match the learner on resume from a checkpoint) runs
end-to-end on a real vLLM engine and returns matching abs-mean scalars for a
stable set of parameter names (embeddings + norms).

To run::

    ./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh \\
        open_instruct/test_weight_sync_fingerprint_gpu.py
"""

import os
import unittest

import datasets
import ray
import torch
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    def test_fingerprint_matches_pretrained_weights(self):
        model_name = "Qwen/Qwen3-0.6B"
        AutoTokenizer.from_pretrained(model_name)

        master_address = ray._private.services.get_node_ip_address()
        master_port = utils.find_free_port()
        os.environ["MASTER_ADDR"] = master_address
        os.environ["MASTER_PORT"] = str(master_port)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)

        trainer_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")

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

        candidate_names = [
            name
            for name, _ in trainer_model.named_parameters()
            if name.endswith("embed_tokens.weight")
            or name.endswith("model.norm.weight")
            or name.endswith("input_layernorm.weight")
            or name.endswith("post_attention_layernorm.weight")
        ]
        sample_names = candidate_names[:4]
        self.assertGreater(len(sample_names), 0)

        trainer_fp = _fingerprint_local(trainer_model, sample_names)
        engine_fp = ray.get(engines[0].get_weight_fingerprint.remote(sample_names))
        logger.info(f"trainer={trainer_fp} engine={engine_fp}")
        for name in sample_names:
            self.assertIn(name, engine_fp)
            self.assertAlmostEqual(
                trainer_fp[name],
                engine_fp[name],
                delta=1e-2 * max(abs(trainer_fp[name]), 1e-8),
                msg=f"fingerprint mismatch for {name!r}",
            )

        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
