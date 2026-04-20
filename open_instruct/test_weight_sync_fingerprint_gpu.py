"""GPU test: broadcasting divergent learner weights to vLLM actually updates vLLM.

The bug this guards against: on resume from a checkpoint, the learner loads
step-N weights while vLLM starts from the pretrain weights. Unless we broadcast
before the first rollout, vLLM silently keeps pretrain weights. A previous
version of this test loaded HF and vLLM from the same pretrain and confirmed
fingerprints matched -- toothless, because it never exercised the case where
learner weights differ from vLLM's loaded weights.

This test:
  1. Creates a vLLM engine from the pretrain.
  2. Creates an HF model, perturbs its weights (simulating a trained learner).
  3. Broadcasts via ``broadcast_weights_to_vllm`` (IPC, single-GPU).
  4. Asserts vLLM's fingerprint now matches the *perturbed* HF weights, not
     the pretrain weights.

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


UNFUSED_PARAM_SUFFIXES = (
    "embed_tokens.weight",
    "model.norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)


def _fingerprint_local(model, param_names: list[str]) -> dict[str, float]:
    params_by_name = dict(model.named_parameters())
    return {
        name: float(params_by_name[name].data.detach().float().abs().mean().item())
        for name in param_names
        if name in params_by_name
    }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestWeightSyncBroadcastUpdatesVLLM(TestGrpoFastBase):
    def test_broadcast_divergent_weights_updates_vllm(self):
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

        sample_names = [name for name, _ in trainer_model.named_parameters() if name.endswith(UNFUSED_PARAM_SUFFIXES)][
            :4
        ]
        self.assertGreater(len(sample_names), 0)

        pretrain_fp = _fingerprint_local(trainer_model, sample_names)
        vllm_pretrain_fp = ray.get(engines[0].get_weight_fingerprint.remote(sample_names))
        for name in sample_names:
            self.assertAlmostEqual(
                pretrain_fp[name],
                vllm_pretrain_fp[name],
                delta=1e-2 * max(abs(pretrain_fp[name]), 1e-8),
                msg=f"pretrain fingerprint mismatch for {name!r}",
            )

        with torch.no_grad():
            for _, param in trainer_model.named_parameters():
                param.data.mul_(1.25)

        perturbed_fp = _fingerprint_local(trainer_model, sample_names)
        for name in sample_names:
            self.assertGreater(
                abs(perturbed_fp[name] - pretrain_fp[name]),
                1e-3 * max(abs(pretrain_fp[name]), 1e-8),
                msg=f"perturbation did not change fingerprint for {name!r}",
            )

        refs = vllm_utils.broadcast_weights_to_vllm(
            model=trainer_model, vllm_engines=engines, model_update_group=None, gather_whole_model=True
        )
        ray.get(refs)
        ray.get([engine.wake_up.remote() for engine in engines])

        vllm_after_fp = ray.get(engines[0].get_weight_fingerprint.remote(sample_names))
        logger.info(f"pretrain={pretrain_fp} perturbed={perturbed_fp} vllm_after={vllm_after_fp}")
        for name in sample_names:
            self.assertAlmostEqual(
                perturbed_fp[name],
                vllm_after_fp[name],
                delta=1e-2 * max(abs(perturbed_fp[name]), 1e-8),
                msg=f"vLLM did not pick up perturbed weights for {name!r}",
            )

        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
