"""GPU tests for on-policy distillation utilities.

These tests require CUDA/vLLM and are intended to run through:

    ./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh
"""

import unittest

import ray
import torch
from transformers import AutoTokenizer

from open_instruct import logger_utils, opd_utils
from open_instruct.test_grpo_fast import TestGrpoFastBase
from open_instruct.utils import maybe_update_beaker_description

logger = logger_utils.setup_logger(__name__)

maybe_update_beaker_description()


class TestOPDTeacherScorerGPU(TestGrpoFastBase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_teacher_scorer_scores_exact_supplied_response_tokens(self):
        """Teacher scorer should load a vLLM teacher and score, not regenerate, student response tokens."""
        model_name = "Qwen/Qwen3-0.6B"
        topk = 128
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        query = tokenizer.encode("The capital of France is", add_special_tokens=False)
        response = tokenizer.encode(" Paris", add_special_tokens=False)
        self.assertGreater(len(query), 0)
        self.assertGreater(len(response), 0)

        teacher_scorers = opd_utils.create_teacher_scorers(
            num_engines=1,
            tensor_parallel_size=1,
            enforce_eager=True,
            tokenizer_name_or_path=model_name,
            tokenizer_revision=None,
            model_name_or_path=model_name,
            model_revision="main",
            seed=42,
            enable_prefix_caching=False,
            max_model_len=128,
            gpu_memory_utilization=0.5,
            topk=topk,
            dtype="bfloat16",
            trust_remote_code=False,
            attention_backend=None,
        )
        result = ray.get(teacher_scorers[0].score.remote([query], [response]))

        self.assertEqual(len(result.teacher_topk_token_ids), 1)
        self.assertEqual(len(result.teacher_topk_logprobs), 1)
        token_ids = result.teacher_topk_token_ids[0]
        logprobs = result.teacher_topk_logprobs[0]

        self.assertEqual(token_ids.shape, (len(response), topk))
        self.assertEqual(logprobs.shape, (len(response), topk))
        self.assertTrue(torch.isfinite(logprobs[:, 0]).all())

        finite_logprobs = torch.where(torch.isfinite(logprobs), logprobs, torch.zeros_like(logprobs))
        topk_mass = torch.where(torch.isfinite(logprobs), torch.exp(finite_logprobs), torch.zeros_like(logprobs)).sum(
            dim=-1
        )
        self.assertTrue((topk_mass > 0.25).all(), f"unexpectedly low teacher top-k mass: {topk_mass}")
        self.assertTrue((topk_mass <= 1.001).all(), f"teacher top-k mass exceeded 1: {topk_mass}")

        first_response_token = response[0]
        self.assertIn(
            first_response_token,
            token_ids[0].tolist(),
            "Expected the supplied response token to appear in the teacher top-k support for this prompt.",
        )
        self.assertGreater(result.metrics["time/opd_teacher_scoring"], 0.0)
        self.assertGreater(result.metrics["opd/teacher_tokens_per_second"], 0.0)
        logger.info(
            "OPD teacher scorer diagnostic passed: response_len=%d topk=%d topk_mass=%s",
            len(response),
            topk,
            topk_mass.tolist(),
        )


if __name__ == "__main__":
    unittest.main()
