import gc
import unittest

import datasets
import numpy as np
import parameterized
import torch
import transformers
import vllm

from open_instruct import dataset_transformation, grpo_fast, grpo_utils, logger_utils, model_utils, rl_utils

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


def _load_prod_prompts(tokenizer: transformers.PreTrainedTokenizer, per_dataset: int = 2) -> list[list[int]]:
    """Load prompts from the full production dataset mix and tokenize.

    Matches production chat template setup:
    - Hybrid model uses olmo123 (falls through to tokenizer's built-in template)
    - Transformer model uses olmo (from CHAT_TEMPLATES in dataset_transformation.py)
    """
    if not tokenizer.chat_template or isinstance(tokenizer.chat_template, dict):
        tokenizer.chat_template = dataset_transformation.CHAT_TEMPLATES["olmo"]
    prompts = []
    for ds_name in PROD_DATASETS:
        ds = datasets.load_dataset(ds_name, split="train")
        for i in range(min(per_dataset, len(ds))):
            messages = ds[i]["messages"]
            if len(messages) > 1 and messages[-1]["role"] == "assistant":
                messages = messages[:-1]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
            prompts.append(input_ids)
    return prompts


def _vllm_generate(
    model_name: str,
    prompt: list[int],
    max_tokens: int,
    enforce_eager: bool = True,
    ssm_cache_dtype: str | None = None,
    ignore_eos: bool = True,
    stop_strings: list[str] | None = None,
    gpu_memory_utilization: float = 0.45,
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
        gpu_memory_utilization=gpu_memory_utilization,
        **kwargs,
    )
    sampling_params = vllm.SamplingParams(
        max_tokens=max_tokens, logprobs=0, seed=SEED, temperature=1.0, ignore_eos=ignore_eos, stop=stop_strings or []
    )
    outputs = llm.generate([{"prompt_token_ids": prompt}], sampling_params=sampling_params)
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
        grpo_fast._patch_gated_deltanet_cu_seqlens()

    padding_mask = query_response != pad_token_id
    input_ids = torch.masked_fill(query_response, ~padding_mask, 0)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, DTYPE),
        device_map="cuda",
        attn_implementation="flash_attention_2",
        use_cache=False,
        trust_remote_code=True,
    )

    with torch.no_grad():
        input_ids = input_ids[None, :].to("cuda")
        attention_mask = attention_mask[None, :].to("cuda")
        position_ids = position_ids[None, :].to("cuda")
        output = model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1].clamp(0, 1),
            position_ids=position_ids[:, :-1],
            return_dict=True,
        )
        logits = output.logits.to(torch.float32)
        logprobs = model_utils.log_softmax_and_gather(logits, input_ids[:, 1:])
        logprobs = logprobs[:, query_len - 1 : query_len - 1 + response_len]

    result = logprobs.flatten().tolist()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _log_diff_stats(label: str, abs_diff: np.ndarray):
    """Log summary statistics for an absolute-difference array."""
    logger.info(
        "%s: mean=%.6f max=%.6f std=%.6f median=%.6f",
        label,
        abs_diff.mean(),
        abs_diff.max(),
        abs_diff.std(),
        np.median(abs_diff),
    )


def _log_positional_diffs(abs_diff: np.ndarray, response_len: int):
    """Log diff at key positions through the response."""
    positions = [0, response_len // 8, response_len // 4, response_len // 2, 3 * response_len // 4, response_len - 1]
    for pos in positions:
        if pos < len(abs_diff):
            window = abs_diff[max(0, pos - 10) : pos + 10]
            logger.info("  resp_pos %5d: diff=%.4f window_mean=%.4f", pos, abs_diff[pos], window.mean())


class TestGRPOLogprobsMatch(unittest.TestCase):
    """Test that mirrors production GRPO: vLLM generates, HF scores the result.

    Uses production sequence length (8192 response tokens) and 1 prompt per
    dataset from the production mix.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand(
        [
            ("hybrid_8192", HYBRID_MODEL, PROD_RESPONSE_LENGTH),
            ("transformer_8192", TRANSFORMER_MODEL, PROD_RESPONSE_LENGTH),
        ]
    )
    def test_grpo_logprobs(self, _name, model_name, max_tokens):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        prompts = _load_prod_prompts(tokenizer, per_dataset=1)

        all_diffs = []
        for prompt_idx, prompt in enumerate(prompts):
            logger.info(
                "model=%s max_tokens=%d prompt=%d/%d prompt_len=%d",
                model_name,
                max_tokens,
                prompt_idx + 1,
                len(prompts),
                len(prompt),
            )

            vllm_result = _vllm_generate(model_name, prompt, max_tokens=max_tokens, enforce_eager=True)
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

            _log_diff_stats(f"prompt_{prompt_idx} resp_len={len(response)}", abs_diff)
            _log_positional_diffs(abs_diff, len(response))

        combined = np.concatenate(all_diffs)
        _log_diff_stats(f"AGGREGATE model={model_name} max_tokens={max_tokens}", combined)


class TestVllmVsPackedHF(unittest.TestCase):
    """Reproduce the exact production comparison: vLLM logprobs vs packed-HF logprobs.

    Production computes local logprobs using forward_for_logprobs on packed
    tensors (with sequence-ID attention masks), then compares against vLLM
    logprobs. Our earlier tests compared either vLLM-vs-unpacked-HF or
    packed-HF-vs-individual-HF, but never vLLM-vs-packed-HF.

    Tests two variants:
    - packed: 2 sequences packed into PROD_PACK_LENGTH (the production case)
    - single: 1 sequence in its own tensor (isolates forward_for_logprobs effect)
    """

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([("packed", 2), ("single", 1)])
    def test_vllm_vs_packed_hf(self, _name, num_sequences):
        grpo_fast._patch_gated_deltanet_cu_seqlens()

        tokenizer = transformers.AutoTokenizer.from_pretrained(HYBRID_MODEL, trust_remote_code=True)
        prompts = _load_prod_prompts(tokenizer, per_dataset=1)
        prompts = prompts[:num_sequences]

        max_tokens = 4500
        vllm_engine = vllm.LLM(
            model=HYBRID_MODEL,
            seed=SEED,
            enforce_eager=True,
            max_model_len=max(len(p) for p in prompts) + max_tokens + 128,
            dtype=DTYPE,
            disable_cascade_attn=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.40,
        )
        sampling_params = vllm.SamplingParams(
            max_tokens=max_tokens, logprobs=0, seed=SEED, temperature=1.0, ignore_eos=True
        )
        outputs = vllm_engine.generate([{"prompt_token_ids": p} for p in prompts], sampling_params=sampling_params)
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

        for i, (p, r) in enumerate(zip(prompts, responses)):
            logger.info("VLLM_VS_PACKED seq=%d prompt_len=%d resp_len=%d total=%d", i, len(p), len(r), len(p) + len(r))

        pack_length = len(prompts[0]) + len(responses[0]) + 16 if num_sequences == 1 else PROD_PACK_LENGTH

        packed = rl_utils.pack_sequences(
            queries=prompts,
            responses=responses,
            masks=[[1] * len(r) for r in responses],
            pack_length=pack_length,
            pad_token_id=tokenizer.pad_token_id,
            vllm_logprobs=vllm_lps,
        )
        n_packs = len(packed.query_responses)
        logger.info(
            "VLLM_VS_PACKED num_seq=%d packed into %d tensor(s) of length %d", num_sequences, n_packs, pack_length
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            HYBRID_MODEL,
            torch_dtype=getattr(torch, DTYPE),
            device_map="cuda",
            attn_implementation="flash_attention_2",
            use_cache=False,
            trust_remote_code=True,
        )

        all_diffs = []
        seq_idx = 0
        for pack_idx in range(n_packs):
            input_ids = packed.query_responses[pack_idx].unsqueeze(0).to("cuda")
            attn = packed.attention_masks[pack_idx].unsqueeze(0).to("cuda")
            pos_ids = packed.position_ids[pack_idx].unsqueeze(0).to("cuda")
            resp_mask = packed.response_masks[pack_idx].unsqueeze(0).to("cuda")
            vllm_lp_tensor = packed.vllm_logprobs[pack_idx].unsqueeze(0).to("cuda")
            pad_id = tokenizer.pad_token_id

            with torch.no_grad():
                local_logprobs, _ = grpo_utils.forward_for_logprobs(
                    model, input_ids, attn, pos_ids, pad_id, temperature=1.0
                )

            resp_mask_shifted = resp_mask[:, 1:].bool()
            vllm_lp_shifted = vllm_lp_tensor[:, 1:]
            valid = resp_mask_shifted & ~torch.isnan(vllm_lp_shifted)

            diff = (local_logprobs - vllm_lp_shifted).abs()
            diff_masked = torch.masked_fill(diff, ~valid, 0.0)

            seq_ids = attn[0].unique()
            seq_ids = seq_ids[seq_ids > 0]
            logger.info("VLLM_VS_PACKED pack=%d has %d sequences", pack_idx, len(seq_ids))

            for sid in seq_ids:
                mask = attn[0] == sid
                resp_in_seq = mask[1:] & resp_mask_shifted[0] & valid[0]
                if resp_in_seq.sum() == 0:
                    continue

                seq_diff = diff_masked[0][resp_in_seq].cpu().numpy()
                all_diffs.append(seq_diff)
                _log_diff_stats(f"VLLM_VS_PACKED pack={pack_idx} seq={seq_idx} resp_tokens={len(seq_diff)}", seq_diff)
                seq_idx += 1

        if all_diffs:
            combined = np.concatenate(all_diffs)
            _log_diff_stats(f"VLLM_VS_PACKED AGGREGATE num_seq={num_sequences}", combined)

        del model
        gc.collect()
        torch.cuda.empty_cache()


class TestNaturalResponseLength(unittest.TestCase):
    """Hypothesis 2: production responses are shorter than 8192 tokens.

    Production uses --stop_strings "</answer>", so many responses terminate
    early. The transformer's sliding-window gap scales steeply with length,
    so shorter responses would explain the low production diff (0.06-0.11)
    vs our forced-8192 test diff (2.394).

    Generates with natural stopping (no ignore_eos) and measures actual
    response lengths + vLLM-HF diff at those lengths.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([("hybrid", HYBRID_MODEL), ("transformer", TRANSFORMER_MODEL)])
    def test_natural_length(self, _name, model_name):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        prompts = _load_prod_prompts(tokenizer, per_dataset=1)

        results = []
        for prompt_idx, prompt in enumerate(prompts):
            vllm_result = _vllm_generate(
                model_name,
                prompt,
                max_tokens=PROD_RESPONSE_LENGTH,
                enforce_eager=True,
                ignore_eos=False,
                stop_strings=["</answer>"],
            )
            response = vllm_result["response"]
            vllm_logprobs = vllm_result["logprobs"]
            resp_len = len(response)

            logger.info(
                "NATURAL_LENGTH model=%s prompt=%d/%d prompt_len=%d resp_len=%d",
                model_name,
                prompt_idx + 1,
                len(prompts),
                len(prompt),
                resp_len,
            )

            if resp_len == 0:
                logger.info("NATURAL_LENGTH skipping empty response")
                continue

            pack_length = len(prompt) + resp_len + 16
            packed = rl_utils.pack_sequences(
                queries=[prompt],
                responses=[response],
                masks=[[1] * resp_len],
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
                response_len=resp_len,
            )

            vllm_arr = np.array(vllm_logprobs)
            hf_arr = np.array(hf_logprobs)
            abs_diff = np.abs(vllm_arr - hf_arr)
            diff_mean = abs_diff.mean()
            results.append((prompt_idx, resp_len, diff_mean))

            _log_diff_stats(f"NATURAL_LENGTH prompt={prompt_idx} resp_len={resp_len}", abs_diff)

        logger.info("NATURAL_LENGTH SUMMARY model=%s", model_name)
        resp_lens = [r[1] for r in results]
        diff_means = [r[2] for r in results]
        if resp_lens:
            logger.info(
                "  resp_len: min=%d max=%d mean=%.0f median=%.0f",
                min(resp_lens),
                max(resp_lens),
                np.mean(resp_lens),
                np.median(resp_lens),
            )
            logger.info(
                "  diff_mean: min=%.6f max=%.6f mean=%.6f median=%.6f",
                min(diff_means),
                max(diff_means),
                np.mean(diff_means),
                np.median(diff_means),
            )


if __name__ == "__main__":
    unittest.main()
