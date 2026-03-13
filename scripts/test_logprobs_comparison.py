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


MAX_TOKENS = 20
SEED = 42
PACK_LENGTH = 64
DTYPE = "bfloat16"
HYBRID_MODEL = "allenai/Olmo-Hybrid-Instruct-DPO-7B"
TRANSFORMER_MODEL = "allenai/Olmo-3-1025-7B"

logger = logging.getLogger(__name__)


class TestLogprobsComparison(unittest.TestCase):
    """Test logprobs calculation and comparison between HuggingFace and vLLM."""

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([
        ("hybrid_capital", HYBRID_MODEL, "The capital of France is"),
        ("hybrid_weather", HYBRID_MODEL, "The weather today is"),
        ("hybrid_ml", HYBRID_MODEL, "Machine learning is"),
        ("transformer_capital", TRANSFORMER_MODEL, "The capital of France is"),
        ("transformer_weather", TRANSFORMER_MODEL, "The weather today is"),
        ("transformer_ml", TRANSFORMER_MODEL, "Machine learning is"),
    ])
    def test_vllm_hf_logprobs_eager(self, _name, model_name, prompt):
        """Test that vLLM eager mode matches HuggingFace logprobs."""
        self._run_comparison(model_name, prompt, enforce_eager=True)

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([
        ("hybrid_capital", HYBRID_MODEL, "The capital of France is"),
        ("hybrid_weather", HYBRID_MODEL, "The weather today is"),
        ("hybrid_ml", HYBRID_MODEL, "Machine learning is"),
        ("transformer_capital", TRANSFORMER_MODEL, "The capital of France is"),
        ("transformer_weather", TRANSFORMER_MODEL, "The weather today is"),
        ("transformer_ml", TRANSFORMER_MODEL, "Machine learning is"),
    ])
    def test_vllm_hf_logprobs_compiled(self, _name, model_name, prompt):
        """Test logprob divergence when vLLM uses torch.compile."""
        self._run_comparison(model_name, prompt, enforce_eager=False)

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([
        ("hybrid_capital", HYBRID_MODEL, "The capital of France is"),
        ("hybrid_weather", HYBRID_MODEL, "The weather today is"),
        ("hybrid_ml", HYBRID_MODEL, "Machine learning is"),
    ])
    def test_vllm_hf_logprobs_compiled_fp32_state(self, _name, model_name, prompt):
        """Test that fp32 SSM state fixes compile divergence for hybrid models."""
        self._run_comparison(
            model_name, prompt, enforce_eager=False, ssm_cache_dtype="float32",
        )

    def _run_comparison(
        self, model_name, prompt, enforce_eager, ssm_cache_dtype=None,
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        query = tokenizer(prompt)['input_ids']

        vllm_output = _get_vllm_logprobs(
            model_name, query,
            enforce_eager=enforce_eager,
            ssm_cache_dtype=ssm_cache_dtype,
        )
        gc.collect()
        torch.cuda.empty_cache()
        packed_sequences = rl_utils.pack_sequences(
            queries=[query],
            responses=[vllm_output["response"]],
            masks=[[1] * len(vllm_output["response"])],
            pack_length=PACK_LENGTH,
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

        self.assertEqual(len(vllm_logprobs), len(vllm_output["response"]))
        self.assertEqual(
            len(vllm_logprobs), len(hf_logprobs),
            f'{vllm_logprobs=}\n{hf_logprobs=}',
        )
        self.assertEqual(vllm_output['response'], packed_response_tokens)

        vllm_arr = np.array(vllm_logprobs)
        hf_arr = np.array(hf_logprobs)
        abs_diff = np.abs(vllm_arr - hf_arr)
        mode = "eager" if enforce_eager else "compiled"
        if ssm_cache_dtype:
            mode += f"+{ssm_cache_dtype}_state"
        is_hybrid = "Hybrid" in model_name
        logger.info(
            "model=%s hybrid=%s mode=%s mean_diff=%.4f max_diff=%.4f std_diff=%.4f",
            model_name.split("/")[-1], is_hybrid, mode,
            abs_diff.mean(), abs_diff.max(), abs_diff.std(),
        )
        for i, (v, h, d) in enumerate(zip(vllm_logprobs, hf_logprobs, abs_diff)):
            logger.info("  token %2d: vllm=%.4f  hf=%.4f  diff=%.4f", i, v, h, d)

        np.testing.assert_array_almost_equal(vllm_logprobs, hf_logprobs, decimal=1)


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
) -> Dict[str, Union[List[int], List[float]]]:
    """Get logprobs using vLLM."""
    kwargs = {}
    if ssm_cache_dtype is not None:
        kwargs["mamba_ssm_cache_dtype"] = ssm_cache_dtype
    llm = vllm.LLM(
        model=model_name,
        seed=SEED,
        enforce_eager=enforce_eager,
        max_model_len=1024,
        dtype=DTYPE,
        disable_cascade_attn=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        **kwargs,
    )

    sampling_params = vllm.SamplingParams(
        max_tokens=MAX_TOKENS,
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
