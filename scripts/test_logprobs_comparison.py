import gc
import logging
import unittest

import datasets
import numpy as np
import parameterized
import torch
import transformers
import vllm
from typing import Dict, List, Union
from transformers import PreTrainedTokenizer

from open_instruct import model_utils
from open_instruct import rl_utils


SEED = 42
DTYPE = "bfloat16"
HYBRID_MODEL = "allenai/Olmo-Hybrid-Instruct-DPO-7B"
DATASET_NAME = "hamishivi/rlvr_acecoder_filtered_filtered"

logger = logging.getLogger(__name__)


def _load_prompts(tokenizer: PreTrainedTokenizer, n: int = 3) -> List[List[int]]:
    """Load real prompts from the production dataset and tokenize with olmo123 template."""
    ds = datasets.load_dataset(DATASET_NAME, split="train")
    prompts = []
    for i in range(n):
        messages = ds[i]["messages"]
        if len(messages) > 1 and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=False,
        )
        prompts.append(input_ids)
    return prompts


class TestGRPOLogprobsMatch(unittest.TestCase):
    """Test that mirrors production GRPO: vLLM generates, HF scores the result.

    In production GRPO:
    1. vLLM generates a response autoregressively (decode path) and returns logprobs
    2. Local HF model scores the full query+response in one forward pass (prefill path)
    3. The logprobs are compared

    This test does exactly that with real dataset prompts at production-scale lengths.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([
        ("eager_1024", True, None, 1024),
        ("eager_2048", True, None, 2048),
        ("eager_4096", True, None, 4096),
        ("eager_8192", True, None, 8192),
        ("compiled_1024", False, None, 1024),
        ("compiled_2048", False, None, 2048),
        ("compiled_4096", False, None, 4096),
        ("compiled_8192", False, None, 8192),
        ("compiled_fp32_1024", False, "float32", 1024),
        ("compiled_fp32_2048", False, "float32", 2048),
        ("compiled_fp32_4096", False, "float32", 4096),
        ("compiled_fp32_8192", False, "float32", 8192),
    ])
    def test_grpo_logprobs(self, _name, enforce_eager, ssm_cache_dtype, max_tokens):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            HYBRID_MODEL, trust_remote_code=True,
        )
        prompts = _load_prompts(tokenizer, n=1)
        prompt = prompts[0]
        logger.info("Prompt length: %d tokens", len(prompt))

        # Step 1: vLLM generates response and returns logprobs (decode path)
        vllm_result = _vllm_generate(
            HYBRID_MODEL, prompt, max_tokens=max_tokens,
            enforce_eager=enforce_eager, ssm_cache_dtype=ssm_cache_dtype,
        )
        response = vllm_result["response"]
        vllm_logprobs = vllm_result["logprobs"]
        logger.info("Generated %d response tokens", len(response))
        gc.collect()
        torch.cuda.empty_cache()

        # Step 2: Pack sequences (same as GRPO)
        pack_length = len(prompt) + len(response) + 16
        packed = rl_utils.pack_sequences(
            queries=[prompt],
            responses=[response],
            masks=[[1] * len(response)],
            pack_length=pack_length,
            pad_token_id=tokenizer.pad_token_id,
            vllm_logprobs=[vllm_logprobs],
        )

        # Step 3: HF scores the full sequence in one forward pass (prefill path)
        hf_logprobs = _hf_score(
            HYBRID_MODEL,
            packed.query_responses[0],
            packed.attention_masks[0],
            packed.position_ids[0],
            tokenizer.pad_token_id,
            query_len=len(prompt),
            response_len=len(response),
        )
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
            "RESULT resp_len=%d mode=%s mean_diff=%.4f max_diff=%.4f std_diff=%.4f",
            len(response), mode, abs_diff.mean(), abs_diff.max(), abs_diff.std(),
        )

        # Log divergence at different positions through the response
        positions = [0, len(response) // 8, len(response) // 4,
                     len(response) // 2, 3 * len(response) // 4, len(response) - 1]
        for pos in positions:
            if pos < len(abs_diff):
                window = abs_diff[max(0, pos - 10):pos + 10]
                logger.info(
                    "  resp_pos %5d: diff=%.4f window_mean=%.4f",
                    pos, abs_diff[pos], window.mean(),
                )


def _vllm_generate(
    model_name: str, prompt: List[int], max_tokens: int,
    enforce_eager: bool = True, ssm_cache_dtype: str | None = None,
) -> Dict[str, Union[List[int], List[float]]]:
    """Generate with vLLM and return response tokens + logprobs (decode path)."""
    kwargs = {}
    if ssm_cache_dtype is not None:
        kwargs["mamba_ssm_cache_dtype"] = ssm_cache_dtype
    llm = vllm.LLM(
        model=model_name,
        seed=SEED,
        enforce_eager=enforce_eager,
        max_model_len=len(prompt) + max_tokens + 128,
        dtype=DTYPE,
        disable_cascade_attn=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.45,
        **kwargs,
    )
    sampling_params = vllm.SamplingParams(
        max_tokens=max_tokens,
        logprobs=0,
        seed=SEED,
        temperature=1.0,
        ignore_eos=True,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": prompt}], sampling_params=sampling_params,
    )
    output = outputs[0]

    response = []
    logprobs = []
    for token_info in output.outputs[0].logprobs:
        token_id = list(token_info.keys())[0]
        logprobs.append(token_info[token_id].logprob)
        response.append(token_id)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return {"response": response, "logprobs": logprobs}


def _hf_score(
    model_name: str,
    query_response: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    pad_token_id: int,
    query_len: int,
    response_len: int,
) -> List[float]:
    """Score the full query+response with HF in one forward pass (prefill path)."""
    padding_mask = query_response != pad_token_id
    input_ids = torch.masked_fill(query_response, ~padding_mask, 0)

    model = transformers.AutoModelForCausalLM.from_pretrained(
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
        # Extract only response token logprobs (same as GRPO)
        logprobs = logprobs[:, query_len - 1: query_len - 1 + response_len]

    result = logprobs.flatten().tolist()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


class TestPackingStateLeak(unittest.TestCase):
    """Test that cu_seqlens monkey-patch resets recurrent state at sequence boundaries.

    For hybrid (recurrent) models, packing multiple sequences into one forward pass
    can leak recurrent state across sequence boundaries. The monkey-patch in
    grpo_fast._patch_gated_deltanet_cu_seqlens passes cu_seqlens to FLA kernels
    so state resets happen natively inside a single forward pass.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    def test_packed_vs_individual_logprobs(self):
        from open_instruct import grpo_fast
        from open_instruct import grpo_utils

        grpo_fast._patch_gated_deltanet_cu_seqlens()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            HYBRID_MODEL, trust_remote_code=True,
        )
        prompts = _load_prompts(tokenizer, n=2)

        model = transformers.AutoModelForCausalLM.from_pretrained(
            HYBRID_MODEL,
            torch_dtype=getattr(torch, DTYPE),
            device_map="cuda",
            attn_implementation="flash_attention_2",
            use_cache=False,
            trust_remote_code=True,
        )

        max_tokens = 256
        vllm_engine = vllm.LLM(
            model=HYBRID_MODEL,
            seed=SEED,
            enforce_eager=True,
            max_model_len=max(len(p) for p in prompts) + max_tokens + 128,
            dtype=DTYPE,
            disable_cascade_attn=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.30,
        )
        sampling_params = vllm.SamplingParams(
            max_tokens=max_tokens, logprobs=0, seed=SEED,
            temperature=1.0, ignore_eos=True,
        )
        outputs = vllm_engine.generate(
            [{"prompt_token_ids": p} for p in prompts],
            sampling_params=sampling_params,
        )
        responses = []
        vllm_lps = []
        for out in outputs:
            resp, lps = [], []
            for token_info in out.outputs[0].logprobs:
                token_id = list(token_info.keys())[0]
                resp.append(token_id)
                lps.append(token_info[token_id].logprob)
            responses.append(resp)
            vllm_lps.append(lps)
        del vllm_engine
        gc.collect()
        torch.cuda.empty_cache()

        pack_length = sum(len(p) + len(r) for p, r in zip(prompts, responses)) + 32
        packed = rl_utils.pack_sequences(
            queries=prompts,
            responses=responses,
            masks=[[1] * len(r) for r in responses],
            pack_length=pack_length,
            pad_token_id=tokenizer.pad_token_id,
            vllm_logprobs=vllm_lps,
        )
        self.assertEqual(len(packed.query_responses), 1, "Expected both sequences in one pack")
        attn_mask = packed.attention_masks[0]
        seq_ids = attn_mask.unique()
        seq_ids = seq_ids[seq_ids > 0]
        self.assertEqual(len(seq_ids), 2, "Expected 2 sequences packed together")

        input_ids = packed.query_responses[0].unsqueeze(0).to("cuda")
        attn = packed.attention_masks[0].unsqueeze(0).to("cuda")
        pos_ids = packed.position_ids[0].unsqueeze(0).to("cuda")
        pad_id = tokenizer.pad_token_id

        with torch.no_grad():
            packed_logprobs, _ = grpo_utils.forward_for_logprobs(
                model, input_ids, attn, pos_ids, pad_id, temperature=1.0,
            )

        second_id = seq_ids[1]
        mask2 = attn[0] == second_id
        indices2 = mask2.nonzero(as_tuple=True)[0]
        s2_start = indices2[0].item()
        s2_end = indices2[-1].item() + 1

        individual_logprobs = _hf_score_raw(
            model, input_ids[0, s2_start:s2_end], pad_id,
        )
        packed_second = packed_logprobs[0, s2_start:s2_end - 1]

        reset_diff = (packed_second - individual_logprobs).abs().mean().item()
        logger.info("Second sequence: patched_vs_individual=%.6f", reset_diff)
        self.assertLess(reset_diff, 0.5, "cu_seqlens-patched logprobs should nearly match individual")

        del model
        gc.collect()
        torch.cuda.empty_cache()


def _hf_score_raw(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Score a single sequence with fresh state."""
    tokens = tokens.unsqueeze(0).to("cuda")
    seq_len = tokens.size(1)
    attn = torch.ones(1, seq_len, device="cuda", dtype=torch.long)
    pos_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
    with torch.no_grad():
        output = model(input_ids=tokens, attention_mask=attn, position_ids=pos_ids)
        logits = getattr(output, "logits", output)
        logits = logits[:, :-1]
        labels = tokens[:, 1:].clone()
        labels[labels == pad_token_id] = 0
        logprobs = model_utils.log_softmax_and_gather(logits, labels)
    return logprobs[0]


if __name__ == "__main__":
    unittest.main()
