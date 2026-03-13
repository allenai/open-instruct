import gc
import logging
import unittest

import numpy as np
import parameterized
import torch
import transformers
import vllm
from typing import Dict, List, Union
from transformers import PreTrainedModel

from open_instruct import model_utils
from open_instruct import rl_utils


SEED = 42
DTYPE = "bfloat16"
HYBRID_MODEL = "allenai/Olmo-Hybrid-Instruct-DPO-7B"

logger = logging.getLogger(__name__)

PROMPT = "Explain the theory of general relativity in detail, including its historical development, mathematical foundations, key predictions, and experimental confirmations. Start from Newton's theory of gravity and work your way through the equivalence principle, curved spacetime, Einstein's field equations, and modern applications."


class TestHybridLogprobsPrefill(unittest.TestCase):
    """Test logprob divergence in prefill (scoring) mode — matches production GRPO.

    In production, vLLM generates a response, then the local HF model re-scores
    the full sequence in a single forward pass. The divergence comes from the
    prefill path (chunk_gated_delta_rule), not the decode path.

    This test has vLLM score a pre-generated sequence using prompt_logprobs,
    so both vLLM and HF use their prefill/chunk paths on the same tokens.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([
        ("eager_256", True, None, 256),
        ("eager_512", True, None, 512),
        ("eager_1024", True, None, 1024),
        ("eager_2048", True, None, 2048),
        ("eager_4096", True, None, 4096),
        ("eager_8192", True, None, 8192),
        ("eager_16384", True, None, 16384),
        ("compiled_256", False, None, 256),
        ("compiled_512", False, None, 512),
        ("compiled_1024", False, None, 1024),
        ("compiled_2048", False, None, 2048),
        ("compiled_4096", False, None, 4096),
        ("compiled_8192", False, None, 8192),
        ("compiled_16384", False, None, 16384),
        ("compiled_fp32_256", False, "float32", 256),
        ("compiled_fp32_512", False, "float32", 512),
        ("compiled_fp32_1024", False, "float32", 1024),
        ("compiled_fp32_2048", False, "float32", 2048),
        ("compiled_fp32_4096", False, "float32", 4096),
        ("compiled_fp32_8192", False, "float32", 8192),
        ("compiled_fp32_16384", False, "float32", 16384),
    ])
    def test_prefill_logprobs(self, _name, enforce_eager, ssm_cache_dtype, seq_len):
        """Score the same token sequence through both vLLM and HF in prefill mode."""
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            HYBRID_MODEL, trust_remote_code=True
        )

        # Generate a fixed sequence of tokens to score.
        # Use vLLM in eager mode to generate, then score with both backends.
        tokens = _generate_fixed_sequence(HYBRID_MODEL, tokenizer, seq_len)
        logger.info("Generated %d tokens to score", len(tokens))

        # Get vLLM logprobs via prompt_logprobs (prefill/scoring path)
        vllm_logprobs = _get_vllm_prompt_logprobs(
            HYBRID_MODEL, tokens,
            enforce_eager=enforce_eager,
            ssm_cache_dtype=ssm_cache_dtype,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # Get HF logprobs via single forward pass
        hf_logprobs = _get_hf_prompt_logprobs(HYBRID_MODEL, tokens)
        gc.collect()
        torch.cuda.empty_cache()

        self.assertEqual(len(vllm_logprobs), len(hf_logprobs))

        vllm_arr = np.array(vllm_logprobs)
        hf_arr = np.array(hf_logprobs)
        abs_diff = np.abs(vllm_arr - hf_arr)

        mode = "eager" if enforce_eager else "compiled"
        if ssm_cache_dtype:
            mode += f"+{ssm_cache_dtype}_state"

        logger.info(
            "RESULT seq_len=%d mode=%s mean_diff=%.4f max_diff=%.4f std_diff=%.4f",
            seq_len, mode, abs_diff.mean(), abs_diff.max(), abs_diff.std(),
        )

        # Log divergence at different positions to see compounding
        positions = [0, seq_len // 8, seq_len // 4, seq_len // 2,
                     3 * seq_len // 4, seq_len - 1]
        for pos in positions:
            if pos < len(abs_diff):
                # Average diff in a window around this position
                window = abs_diff[max(0, pos - 5):pos + 5]
                logger.info(
                    "  position %5d: local_diff=%.4f window_mean=%.4f",
                    pos, abs_diff[pos], window.mean(),
                )


def _generate_fixed_sequence(model_name, tokenizer, target_len):
    """Generate a fixed token sequence using vLLM."""
    prompt_ids = tokenizer(PROMPT)['input_ids']

    llm = vllm.LLM(
        model=model_name,
        seed=SEED,
        enforce_eager=True,
        max_model_len=target_len + len(prompt_ids) + 128,
        dtype=DTYPE,
        disable_cascade_attn=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.45,
    )
    sampling_params = vllm.SamplingParams(
        max_tokens=target_len,
        seed=SEED,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_ids}], sampling_params=sampling_params,
    )
    response_ids = [t for t in outputs[0].outputs[0].token_ids]
    full_sequence = prompt_ids + response_ids

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return full_sequence


def _get_vllm_prompt_logprobs(
    model_name, tokens, enforce_eager=True, ssm_cache_dtype=None,
):
    """Score a token sequence using vLLM's prompt_logprobs (prefill path)."""
    kwargs = {}
    if ssm_cache_dtype is not None:
        kwargs["mamba_ssm_cache_dtype"] = ssm_cache_dtype
    llm = vllm.LLM(
        model=model_name,
        seed=SEED,
        enforce_eager=enforce_eager,
        max_model_len=len(tokens) + 128,
        dtype=DTYPE,
        disable_cascade_attn=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.45,
        **kwargs,
    )
    sampling_params = vllm.SamplingParams(
        max_tokens=1,
        prompt_logprobs=0,
        seed=SEED,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": tokens}], sampling_params=sampling_params,
    )

    # prompt_logprobs[0] is None (no logprob for first token)
    # prompt_logprobs[i] has the logprob of token i given tokens[:i]
    prompt_lps = outputs[0].prompt_logprobs
    logprobs = []
    for i in range(1, len(tokens)):
        token_id = tokens[i]
        if prompt_lps[i] is not None and token_id in prompt_lps[i]:
            logprobs.append(prompt_lps[i][token_id].logprob)
        else:
            logprobs.append(float('nan'))

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return logprobs


def _get_hf_prompt_logprobs(model_name, tokens):
    """Score a token sequence using HF forward pass (prefill path)."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, DTYPE),
        device_map='cuda',
        attn_implementation="flash_attention_2",
        use_cache=False,
        trust_remote_code=True,
    )

    input_ids = torch.tensor([tokens], device='cuda')

    with torch.no_grad():
        output = model(input_ids=input_ids[:, :-1], return_dict=True)
        logits = output.logits.to(torch.float32)
        logprobs = model_utils.log_softmax_and_gather(logits, input_ids[:, 1:])

    result = logprobs.flatten().tolist()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    unittest.main()
