import gc
import unittest

import datasets
import numpy as np
import parameterized
import torch
import transformers
import vllm

from open_instruct import logger_utils
from open_instruct import model_utils
from open_instruct import rl_utils


SEED = 42
DTYPE = "bfloat16"
HYBRID_MODEL = "allenai/Olmo-Hybrid-Instruct-DPO-7B"
TRANSFORMER_MODEL = "allenai/Olmo-3-1025-7B"

# Production config from scripts/train/olmo3/7b_instruct_hybrid_rl.sh
PROD_RESPONSE_LENGTH = 8192
PROD_PACK_LENGTH = 11264
PROD_MAX_PROMPT_LENGTH = 2048

PROD_DATASETS = [
    "hamishivi/rlvr_acecoder_filtered_filtered",
    "hamishivi/omega-combined-no-boxed_filtered",
    "hamishivi/rlvr_orz_math_57k_collected_filtered",
    "hamishivi/polaris_53k",
    "allenai/IF_multi_constraints_upto5_filtered_dpo_0625_filter-keyword-filtered-topic-char-topic-filtered",
    "allenai/rlvr_general_mix-keyword-filtered-topic-chars-char-filt-topic-filtered",
]

logger = logger_utils.setup_logger(__name__)


def _load_prod_prompts(
    tokenizer: transformers.PreTrainedTokenizer, per_dataset: int = 2,
) -> list[list[int]]:
    """Load prompts from the full production dataset mix and tokenize."""
    prompts = []
    for ds_name in PROD_DATASETS:
        ds = datasets.load_dataset(ds_name, split="train")
        for i in range(min(per_dataset, len(ds))):
            messages = ds[i]["messages"]
            if len(messages) > 1 and messages[-1]["role"] == "assistant":
                messages = messages[:-1]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_dict=False,
            )
            prompts.append(input_ids)
    return prompts


def _vllm_generate(
    model_name: str, prompt: list[int], max_tokens: int,
    enforce_eager: bool = True, ssm_cache_dtype: str | None = None,
) -> dict[str, list[int] | list[float]]:
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
    apply_patch: bool = False,
) -> list[float]:
    """Score the full query+response with HF in one forward pass (prefill path)."""
    if apply_patch:
        from open_instruct import grpo_fast
        grpo_fast._patch_gated_deltanet_cu_seqlens()

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
        logprobs = logprobs[:, query_len - 1: query_len - 1 + response_len]

    result = logprobs.flatten().tolist()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


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


def _log_diff_stats(label: str, abs_diff: np.ndarray):
    """Log summary statistics for an absolute-difference array."""
    logger.info(
        "%s: mean=%.6f max=%.6f std=%.6f median=%.6f",
        label, abs_diff.mean(), abs_diff.max(), abs_diff.std(), np.median(abs_diff),
    )


def _log_positional_diffs(abs_diff: np.ndarray, response_len: int):
    """Log diff at key positions through the response."""
    positions = [
        0, response_len // 8, response_len // 4,
        response_len // 2, 3 * response_len // 4, response_len - 1,
    ]
    for pos in positions:
        if pos < len(abs_diff):
            window = abs_diff[max(0, pos - 10):pos + 10]
            logger.info(
                "  resp_pos %5d: diff=%.4f window_mean=%.4f",
                pos, abs_diff[pos], window.mean(),
            )


class TestGRPOLogprobsMatch(unittest.TestCase):
    """Test that mirrors production GRPO: vLLM generates, HF scores the result.

    Uses production sequence length (8192 response tokens) and 1 prompt per
    dataset from the production mix.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([
        ("hybrid_8192", HYBRID_MODEL, PROD_RESPONSE_LENGTH),
        ("transformer_8192", TRANSFORMER_MODEL, PROD_RESPONSE_LENGTH),
    ])
    def test_grpo_logprobs(self, _name, model_name, max_tokens):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        prompts = _load_prod_prompts(tokenizer, per_dataset=1)

        all_diffs = []
        for prompt_idx, prompt in enumerate(prompts):
            logger.info(
                "model=%s max_tokens=%d prompt=%d/%d prompt_len=%d",
                model_name, max_tokens, prompt_idx + 1, len(prompts), len(prompt),
            )

            vllm_result = _vllm_generate(
                model_name, prompt, max_tokens=max_tokens, enforce_eager=True,
            )
            response = vllm_result["response"]
            vllm_logprobs = vllm_result["logprobs"]
            logger.info("Generated %d response tokens", len(response))

            pack_length = len(prompt) + len(response) + 16
            packed = rl_utils.pack_sequences(
                queries=[prompt],
                responses=[response],
                masks=[[1] * len(response)],
                pack_length=pack_length,
                pad_token_id=tokenizer.pad_token_id,
                vllm_logprobs=[vllm_logprobs],
            )

            hf_logprobs = _hf_score(
                model_name,
                packed.query_responses[0],
                packed.attention_masks[0],
                packed.position_ids[0],
                tokenizer.pad_token_id,
                query_len=len(prompt),
                response_len=len(response),
            )

            self.assertEqual(len(vllm_logprobs), len(hf_logprobs))

            vllm_arr = np.array(vllm_logprobs)
            hf_arr = np.array(hf_logprobs)
            abs_diff = np.abs(vllm_arr - hf_arr)
            all_diffs.append(abs_diff)

            _log_diff_stats(
                f"prompt_{prompt_idx} resp_len={len(response)}", abs_diff,
            )
            _log_positional_diffs(abs_diff, len(response))

        combined = np.concatenate(all_diffs)
        _log_diff_stats(
            f"AGGREGATE model={model_name} max_tokens={max_tokens}", combined,
        )


class TestPatchEffect(unittest.TestCase):
    """Test whether the cu_seqlens monkey-patch changes logprobs for single sequences.

    For each model, compares:
    - unpatched HF vs patched HF (patch_vs_unpatched)
    - vLLM vs unpatched HF (vllm_vs_unpatched)
    - vLLM vs patched HF (vllm_vs_patched)

    This answers whether the patch helps, hurts, or is neutral for each model.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([
        ("hybrid_8192", HYBRID_MODEL, PROD_RESPONSE_LENGTH),
        ("transformer_8192", TRANSFORMER_MODEL, PROD_RESPONSE_LENGTH),
    ])
    def test_patch_effect(self, _name, model_name, max_tokens):
        from open_instruct import grpo_fast

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        prompts = _load_prod_prompts(tokenizer, per_dataset=1)
        prompt = prompts[0]
        logger.info(
            "model=%s max_tokens=%d prompt_len=%d",
            model_name, max_tokens, len(prompt),
        )

        vllm_result = _vllm_generate(
            model_name, prompt, max_tokens=max_tokens, enforce_eager=True,
        )
        response = vllm_result["response"]
        vllm_lps = np.array(vllm_result["logprobs"])

        # Load model for unpatched scoring.
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, DTYPE),
            device_map="cuda",
            attn_implementation="flash_attention_2",
            use_cache=False,
            trust_remote_code=True,
        )

        query_response = torch.tensor(
            prompt + response, device="cuda",
        ).unsqueeze(0)
        seq_len = query_response.size(1)
        attn_mask = torch.ones(1, seq_len, device="cuda", dtype=torch.long)
        pos_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        query_len = len(prompt)

        # Score WITHOUT the patch.
        with torch.no_grad():
            output = model(
                input_ids=query_response,
                attention_mask=attn_mask,
                position_ids=pos_ids,
            )
            logits = output.logits.to(torch.float32)
            unpatched_lps = model_utils.log_softmax_and_gather(
                logits[:, :-1], query_response[:, 1:],
            )
            unpatched_resp = unpatched_lps[
                0, query_len - 1: query_len - 1 + len(response)
            ].cpu().numpy()

        # Apply the patch.
        grpo_fast._patch_gated_deltanet_cu_seqlens()

        # Score WITH the patch.
        with torch.no_grad():
            output = model(
                input_ids=query_response,
                attention_mask=attn_mask,
                position_ids=pos_ids,
            )
            logits = output.logits.to(torch.float32)
            patched_lps = model_utils.log_softmax_and_gather(
                logits[:, :-1], query_response[:, 1:],
            )
            patched_resp = patched_lps[
                0, query_len - 1: query_len - 1 + len(response)
            ].cpu().numpy()

        patch_diff = np.abs(patched_resp - unpatched_resp)
        vllm_vs_unpatched = np.abs(vllm_lps - unpatched_resp)
        vllm_vs_patched = np.abs(vllm_lps - patched_resp)

        _log_diff_stats(f"patch_vs_unpatched resp_len={len(response)}", patch_diff)
        _log_diff_stats(f"vllm_vs_unpatched  resp_len={len(response)}", vllm_vs_unpatched)
        _log_diff_stats(f"vllm_vs_patched    resp_len={len(response)}", vllm_vs_patched)

        del model
        gc.collect()
        torch.cuda.empty_cache()


class TestLengthScaling(unittest.TestCase):
    """Test how the vLLM-vs-HF logprob gap scales with sequence length.

    For both models, generates at increasing lengths and reports diff_mean at each.
    Constant gap implies a kernel offset; growing gap implies accumulating error
    through recurrent layers.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([
        ("hybrid", HYBRID_MODEL),
        ("transformer", TRANSFORMER_MODEL),
    ])
    def test_length_scaling(self, _name, model_name):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        prompts = _load_prod_prompts(tokenizer, per_dataset=1)
        prompt = prompts[0]

        lengths = [1024, 4096, 8192]
        results = []

        for max_tokens in lengths:
            logger.info(
                "LENGTH_SCALING model=%s max_tokens=%d prompt_len=%d",
                model_name, max_tokens, len(prompt),
            )

            vllm_result = _vllm_generate(
                model_name, prompt, max_tokens=max_tokens, enforce_eager=True,
            )
            response = vllm_result["response"]
            vllm_logprobs = vllm_result["logprobs"]

            pack_length = len(prompt) + len(response) + 16
            packed = rl_utils.pack_sequences(
                queries=[prompt],
                responses=[response],
                masks=[[1] * len(response)],
                pack_length=pack_length,
                pad_token_id=tokenizer.pad_token_id,
                vllm_logprobs=[vllm_logprobs],
            )

            hf_logprobs = _hf_score(
                model_name,
                packed.query_responses[0],
                packed.attention_masks[0],
                packed.position_ids[0],
                tokenizer.pad_token_id,
                query_len=len(prompt),
                response_len=len(response),
            )

            vllm_arr = np.array(vllm_logprobs)
            hf_arr = np.array(hf_logprobs)
            abs_diff = np.abs(vllm_arr - hf_arr)
            diff_mean = abs_diff.mean()
            results.append((max_tokens, len(response), diff_mean))

            _log_diff_stats(
                f"LENGTH_SCALING resp_len={len(response)}", abs_diff,
            )

        logger.info("LENGTH_SCALING SUMMARY model=%s", model_name)
        for max_tok, resp_len, mean in results:
            logger.info(
                "  max_tokens=%5d resp_len=%5d diff_mean=%.6f", max_tok, resp_len, mean,
            )


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
        prompts = _load_prod_prompts(tokenizer, per_dataset=1)
        prompts = prompts[:2]

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


if __name__ == "__main__":
    unittest.main()
