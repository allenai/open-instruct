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
TRANSFORMER_MODEL = "allenai/Olmo-3-1025-7B"

logger = logging.getLogger(__name__)

PROMPTS = [
    "The capital of France is",
    "The weather today is",
    "Machine learning is",
]


class TestHybridLogprobsShort(unittest.TestCase):
    """Short-sequence (20 token) hybrid logprob tests."""

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([
        (f"eager_{i}", HYBRID_MODEL, p, True, None, 20)
        for i, p in enumerate(PROMPTS)
    ] + [
        (f"compiled_{i}", HYBRID_MODEL, p, False, None, 20)
        for i, p in enumerate(PROMPTS)
    ] + [
        (f"compiled_fp32_{i}", HYBRID_MODEL, p, False, "float32", 20)
        for i, p in enumerate(PROMPTS)
    ])
    def test_short(self, _name, model_name, prompt, eager, ssm_dtype, max_tokens):
        _run_comparison(self, model_name, prompt, eager, ssm_dtype, max_tokens)


class TestHybridLogprobsLong(unittest.TestCase):
    """Long-sequence hybrid logprob tests to measure divergence compounding."""

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([
        (f"eager_256_{i}", HYBRID_MODEL, p, True, None, 256)
        for i, p in enumerate(PROMPTS)
    ] + [
        (f"compiled_256_{i}", HYBRID_MODEL, p, False, None, 256)
        for i, p in enumerate(PROMPTS)
    ] + [
        (f"compiled_fp32_256_{i}", HYBRID_MODEL, p, False, "float32", 256)
        for i, p in enumerate(PROMPTS)
    ] + [
        (f"eager_512_{i}", HYBRID_MODEL, p, True, None, 512)
        for i, p in enumerate(PROMPTS)
    ] + [
        (f"compiled_512_{i}", HYBRID_MODEL, p, False, None, 512)
        for i, p in enumerate(PROMPTS)
    ] + [
        (f"compiled_fp32_512_{i}", HYBRID_MODEL, p, False, "float32", 512)
        for i, p in enumerate(PROMPTS)
    ] + [
        (f"eager_1024_{i}", HYBRID_MODEL, p, True, None, 1024)
        for i, p in enumerate(PROMPTS)
    ] + [
        (f"compiled_1024_{i}", HYBRID_MODEL, p, False, None, 1024)
        for i, p in enumerate(PROMPTS)
    ] + [
        (f"compiled_fp32_1024_{i}", HYBRID_MODEL, p, False, "float32", 1024)
        for i, p in enumerate(PROMPTS)
    ])
    def test_long(self, _name, model_name, prompt, eager, ssm_dtype, max_tokens):
        _run_comparison(self, model_name, prompt, eager, ssm_dtype, max_tokens)


def _run_comparison(test_case, model_name, prompt, enforce_eager, ssm_cache_dtype,
                    max_tokens):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    query = tokenizer(prompt)['input_ids']
    pack_length = len(query) + max_tokens + 16

    vllm_output = _get_vllm_logprobs(
        model_name, query,
        enforce_eager=enforce_eager,
        ssm_cache_dtype=ssm_cache_dtype,
        max_tokens=max_tokens,
    )
    gc.collect()
    torch.cuda.empty_cache()
    packed_sequences = rl_utils.pack_sequences(
        queries=[query],
        responses=[vllm_output["response"]],
        masks=[[1] * len(vllm_output["response"])],
        pack_length=pack_length,
        pad_token_id=tokenizer.pad_token_id,
        vllm_logprobs=[vllm_output["logprobs"]],
    )

    hf_logprobs = _get_hf_logprobs(
        model_name, query,
        vllm_output["response"],
        packed_sequences.query_responses[0],
        packed_sequences.attention_masks[0],
        packed_sequences.position_ids[0],
        tokenizer.pad_token_id,
    )
    vllm_logprobs = vllm_output["logprobs"]

    packed_response_tokens = packed_sequences.query_responses[0][
        len(query):len(query) + len(vllm_output['response'])
    ].tolist()

    test_case.assertEqual(len(vllm_logprobs), len(vllm_output["response"]))
    test_case.assertEqual(
        len(vllm_logprobs), len(hf_logprobs),
        f'{vllm_logprobs=}\n{hf_logprobs=}',
    )
    test_case.assertEqual(vllm_output['response'], packed_response_tokens)

    vllm_arr = np.array(vllm_logprobs)
    hf_arr = np.array(hf_logprobs)
    abs_diff = np.abs(vllm_arr - hf_arr)
    mode = "eager" if enforce_eager else "compiled"
    if ssm_cache_dtype:
        mode += f"+{ssm_cache_dtype}_state"
    logger.info(
        "RESULT tokens=%d mode=%s mean_diff=%.4f max_diff=%.4f std_diff=%.4f",
        max_tokens, mode, abs_diff.mean(), abs_diff.max(), abs_diff.std(),
    )

    # Log per-token diffs at intervals for long sequences
    step = max(1, len(vllm_logprobs) // 20)
    for i in range(0, len(vllm_logprobs), step):
        logger.info(
            "  token %4d: vllm=%.4f  hf=%.4f  diff=%.4f",
            i, vllm_logprobs[i], hf_logprobs[i], abs_diff[i],
        )

    # Log divergence at sequence end (last 5 tokens)
    for i in range(max(0, len(vllm_logprobs) - 5), len(vllm_logprobs)):
        logger.info(
            "  token %4d (tail): vllm=%.4f  hf=%.4f  diff=%.4f",
            i, vllm_logprobs[i], hf_logprobs[i], abs_diff[i],
        )


def _get_hf_logprobs(
    model_name: str, query: List[int],
    response: List[int],
    query_response, attention_mask, position_ids, pad_token_id,
) -> List[float]:
    """Get logprobs using HuggingFace transformers."""
    padding_mask = query_response != pad_token_id
    input_ids = torch.masked_fill(query_response, ~padding_mask, 0)

    model: PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, DTYPE),
        device_map='cuda',
        attn_implementation="flash_attention_2",
        use_cache=False,
        trust_remote_code=True,
    )

    with torch.no_grad():
        input_ids = input_ids[None, :].to('cuda')
        attention_mask = attention_mask[None, :].to('cuda')
        position_ids = position_ids[None, :].to('cuda')
        output = model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1].clamp(0, 1),
            position_ids=position_ids[:, :-1],
            return_dict=True,
        )
        logits = output.logits.to(torch.float32)
        logprobs = model_utils.log_softmax_and_gather(logits, input_ids[:, 1:])
        logprobs = logprobs[:, len(query) - 1:]
    result = logprobs.flatten().tolist()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _get_vllm_logprobs(
    model_name: str,
    prompt: List[int],
    enforce_eager: bool = True,
    ssm_cache_dtype: str | None = None,
    max_tokens: int = 20,
) -> Dict[str, Union[List[int], List[float]]]:
    """Get logprobs using vLLM."""
    kwargs = {}
    if ssm_cache_dtype is not None:
        kwargs["mamba_ssm_cache_dtype"] = ssm_cache_dtype
    max_model_len = max(2048, len(prompt) + max_tokens + 128)
    llm = vllm.LLM(
        model=model_name,
        seed=SEED,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        dtype=DTYPE,
        disable_cascade_attn=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        **kwargs,
    )

    sampling_params = vllm.SamplingParams(
        max_tokens=max_tokens,
        logprobs=0,
        seed=SEED,
    )

    outputs = llm.generate(
        [{"prompt_token_ids": prompt}], sampling_params=sampling_params,
    )
    output = outputs[0]

    response = []
    logprobs = []

    for token_info in output.outputs[0].logprobs:
        token_id = list(token_info.keys())[0]
        logprob_info = token_info[token_id]

        response.append(token_id)
        logprobs.append(logprob_info.logprob)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "response": response,
        "logprobs": logprobs,
    }


if __name__ == "__main__":
    unittest.main()
