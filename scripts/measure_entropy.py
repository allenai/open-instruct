#!/usr/bin/env python
"""
Measure token-level predictive entropy of a language model over a dataset.

Supports:
- HF Hub/local model via transformers AutoModelForCausalLM
- Input as: JSONL (one object per line), TXT (one prompt per line), or HuggingFace datasets
- Evaluation mode: compute entropy on next-token distribution conditioned on provided text
- Optional generate mode: generate tokens and compute entropy on generated steps

Outputs:
- Per-sample JSONL with mean entropy and optional sequence of entropies
- Summary stats (mean, std, percentiles)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union, Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional vLLM import
try:
    from vllm import LLM as VLLM, SamplingParams as VLLMSamplingParams  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    VLLM = None
    VLLMSamplingParams = None

# Prefer library implementation; fallback to local implementation if unavailable
try:
    from open_instruct.model_utils import entropy_from_logits as _entropy_from_logits  # type: ignore[attr-defined]
    entropy_from_logits = _entropy_from_logits  # noqa: N816
except Exception:
    def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:  # noqa: N802
        pd = torch.nn.functional.softmax(logits, dim=-1)
        return torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)


@dataclass
class Args:
    model_name_or_path: str
    dataset: str
    dataset_split: str = "train"
    text_field: Optional[str] = None
    jsonl_field: Optional[str] = None
    messages_field: Optional[str] = None
    jsonl_messages_field: Optional[str] = None
    batch_size: int = 8
    max_samples: int = 0
    max_length: int = 2048
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "auto"  # float16, bfloat16, float32, or auto
    compile: bool = False
    trust_remote_code: bool = True
    revision: Optional[str] = None
    # generation mode
    do_generate: bool = False
    max_new_tokens: int = 0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    # chat template
    use_chat_template: bool = False
    system_prompt: Optional[str] = None
    # output
    output_path: Optional[str] = None
    write_entropies: bool = False
    # vLLM backend for generation
    use_vllm: bool = False
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: Optional[float] = None
    vllm_enforce_eager: bool = False
    vllm_max_model_len: Optional[int] = None


def parse_args() -> Args:
    import argparse

    p = argparse.ArgumentParser(description="Measure LM predictive entropy on a dataset")
    p.add_argument("model_name_or_path", type=str)
    p.add_argument("dataset", type=str, help="JSONL/TXT file or HF datasets path e.g. 'wikitext' or 'json' module")
    p.add_argument("--dataset-split", type=str, default="train")
    p.add_argument("--text-field", type=str, default=None, help="HF datasets text field")
    p.add_argument("--jsonl-field", type=str, default=None, help="JSONL key for text. If None, try 'text' or 'prompt'")
    p.add_argument("--messages-field", type=str, default=None, help="HF datasets field that holds a messages list")
    p.add_argument("--jsonl-messages-field", type=str, default=None, help="JSONL key that holds a messages list")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--do-generate", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--use-chat-template", action="store_true", help="Encode inputs via tokenizer.apply_chat_template")
    p.add_argument("--system-prompt", type=str, default=None, help="Optional system prompt for single-turn text inputs")
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--write-entropies", action="store_true", help="Write full per-token entropies per sample")
    # vLLM specific
    p.add_argument("--use-vllm", action="store_true", help="Use vLLM for generation (HF used for scoring entropies)")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    p.add_argument("--vllm-enforce-eager", action="store_true")
    p.add_argument("--vllm-max-model-len", type=int, default=None, help="Override vLLM max model length; defaults to --max-length")

    a = p.parse_args()
    return Args(
        model_name_or_path=a.model_name_or_path,
        dataset=a.dataset,
        dataset_split=a.dataset_split,
        text_field=a.text_field,
        jsonl_field=a.jsonl_field,
        messages_field=a.messages_field,
        jsonl_messages_field=a.jsonl_messages_field,
        batch_size=a.batch_size,
        max_samples=a.max_samples,
        max_length=a.max_length,
        device=a.device,
        dtype=a.dtype,
        compile=a.compile,
        trust_remote_code=a.trust_remote_code,
        revision=a.revision,
        do_generate=a.do_generate,
        max_new_tokens=a.max_new_tokens,
        temperature=a.temperature,
        top_p=a.top_p,
        top_k=a.top_k,
        use_chat_template=a.use_chat_template,
        system_prompt=a.system_prompt,
        output_path=a.output_path,
        write_entropies=a.write_entropies,
        use_vllm=a.use_vllm,
        vllm_tensor_parallel_size=a.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=a.vllm_gpu_memory_utilization,
        vllm_enforce_eager=a.vllm_enforce_eager,
        vllm_max_model_len=a.vllm_max_model_len,
    )


def _get_dtype(dtype_str: str):
    if dtype_str == "auto":
        return None
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def load_model_and_tokenizer(args: Args):
    torch_dtype = _get_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, revision=args.revision, trust_remote_code=args.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if args.device == "cuda" else None,
    )
    if args.device == "cpu":
        model.to("cpu")
    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[arg-type]
        except Exception:
            pass
    model.eval()
    return model, tok


def read_dataset(args: Args) -> List[Union[str, List[dict]]]:
    path = args.dataset
    if os.path.isfile(path):
        # JSONL or TXT
        texts: List[Union[str, List[dict]]] = []
        is_jsonl = path.endswith(".jsonl") or path.endswith(".json")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if args.max_samples and len(texts) >= args.max_samples:
                    break
                line = line.rstrip("\n")
                if not line:
                    continue
                if is_jsonl:
                    obj = json.loads(line)
                    # Prefer messages if present/configured
                    mkey = args.jsonl_messages_field
                    if mkey is None and isinstance(obj, dict) and "messages" in obj:
                        mkey = "messages"
                    if mkey is not None and mkey in obj and isinstance(obj[mkey], list):
                        texts.append(obj[mkey])
                    else:
                        key = args.jsonl_field
                        if key is None:
                            if "text" in obj:
                                key = "text"
                            elif "prompt" in obj:
                                key = "prompt"
                        if key is None or key not in obj:
                            raise KeyError("jsonl_field not provided and no 'messages'/'text'/'prompt' in JSONL line")
                        texts.append(str(obj[key]))
                else:
                    texts.append(line)
        return texts

    # HF datasets
    ds = load_dataset(path, split=args.dataset_split)
    # messages field takes precedence
    mfield = args.messages_field
    if mfield is None and "messages" in ds.column_names:
        mfield = "messages"
    if mfield is not None and mfield in ds.column_names:
        sel = ds[mfield]
        if args.max_samples:
            sel = sel[: min(args.max_samples, len(sel))]
        return [x for x in sel]

    field = args.text_field
    if field is None:
        # heuristic
        for candidate in ("text", "prompt", "content"):
            if candidate in ds.column_names:
                field = candidate
                break
    if field is None:
        raise ValueError("Please provide --text-field for this dataset (or --messages-field)")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    return [str(x) for x in ds[field]]


def _has_chat_template(tok) -> bool:
    return hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None)


def _to_messages(item: Union[str, List[dict]], system_prompt: Optional[str]) -> List[dict]:
    if isinstance(item, list):
        return item
    msgs: List[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": item})
    return msgs


def encode_batch(tok, batch_items: List[Union[str, List[dict]]], args: Args):
    if args.use_chat_template and _has_chat_template(tok):
        input_id_seqs: List[List[int]] = []
        for itm in batch_items:
            messages = _to_messages(itm, args.system_prompt)
            ids = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=args.do_generate,
                return_tensors=None,
            )
            if isinstance(ids, torch.Tensor):
                ids_list = ids.squeeze(0).tolist()
            else:
                ids_list = ids  # already list[int]
            input_id_seqs.append(ids_list)
        enc = tok.pad({"input_ids": input_id_seqs}, padding=True, max_length=args.max_length, return_tensors="pt")
        return enc
    else:
        # fallback: plain string encoding
        as_texts = [itm if isinstance(itm, str) else "\n\n".join([f"{m.get('role', 'user')}: {m.get('content','')}" for m in itm]) for itm in batch_items]
        return tok(as_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)


def prompts_from_batch(tok, batch_items: List[Union[str, List[dict]]], args: Args) -> List[str]:
    """Render batch items into prompt strings for vLLM generation.

    Mirrors encode_batch logic but returns strings, ensuring the same chat template usage.
    """
    prompts: List[str] = []
    if args.use_chat_template and _has_chat_template(tok):
        for itm in batch_items:
            messages = _to_messages(itm, args.system_prompt)
            rendered = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors=None,
            )
            if not isinstance(rendered, str):
                rendered = tok.decode(rendered, skip_special_tokens=False)  # safeguard
            prompts.append(rendered)
    else:
        for itm in batch_items:
            if isinstance(itm, str):
                prompts.append(itm)
            else:
                prompts.append("\n\n".join([f"{m.get('role', 'user')}: {m.get('content','')}" for m in itm]))
    return prompts


@torch.no_grad()
def compute_entropy_eval(model, tok, texts: List[Union[str, List[dict]]], args: Args):
    device = args.device
    pad_id = tok.pad_token_id
    results = []
    all_means: List[float] = []

    for i in tqdm(range(0, len(texts), args.batch_size), desc="entropy-eval"):
        batch_texts = texts[i : i + args.batch_size]
        enc = encode_batch(tok, batch_texts, args)
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        outputs = model(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], return_dict=True)
        logits = outputs.logits  # [B, T-1, V]
        ent = entropy_from_logits(logits)  # [B, T-1]

        # mask out padding positions in labels side (next-token positions)
        next_tok_mask = attention_mask[:, 1:].to(dtype=ent.dtype)
        ent_masked = ent * next_tok_mask
        lengths = next_tok_mask.sum(dim=1).clamp(min=1)
        mean_ent = (ent_masked.sum(dim=1) / lengths).tolist()
        all_means.extend(mean_ent)

        for j, t in enumerate(batch_texts):
            item = {"index": i + j, "mean_entropy": float(mean_ent[j])}
            if isinstance(t, list):
                item["messages"] = t
            else:
                item["text"] = t
            if args.write_entropies:
                # store only valid range per example
                valid_len = int(lengths[j].item())
                item["entropies"] = ent[j, :valid_len].tolist()
            results.append(item)

    return results, all_means


@torch.no_grad()
def compute_entropy_generate(model, tok, texts: List[Union[str, List[dict]]], args: Args):
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0 for generate mode")
    device = args.device
    results = []
    all_means: List[float] = []

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True if (args.temperature != 1.0 or args.top_p < 1.0 or args.top_k > 0) else False,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else 0,
        return_dict_in_generate=True,
        output_logits=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    for i in tqdm(range(0, len(texts), args.batch_size), desc="entropy-gen"):
        batch_texts = texts[i : i + args.batch_size]
        enc = encode_batch(tok, batch_texts, args)
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        # logits is a list of length max_new_tokens with shape [B, V]
        logits_steps = torch.stack(out.logits, dim=1)  # [B, S, V]
        ent = entropy_from_logits(logits_steps)  # [B, S]

        mean_ent = ent.mean(dim=1).tolist()
        all_means.extend(mean_ent)
        sequences = out.sequences
        dec = tok.batch_decode(sequences, skip_special_tokens=True)

        for j, prompt in enumerate(batch_texts):
            item = {"index": i + j, "generated_text": dec[j], "mean_entropy": float(mean_ent[j])}
            if isinstance(prompt, list):
                item["messages"] = prompt
            else:
                item["prompt"] = prompt
            if args.write_entropies:
                item["entropies"] = ent[j].tolist()
            results.append(item)

    return results, all_means


@torch.no_grad()
def compute_entropy_generate_vllm(model_hf, tok, texts: List[Union[str, List[dict]]], args: Args):
    if VLLM is None or VLLMSamplingParams is None:
        raise ImportError("vLLM is not installed. Please install vllm to use --use-vllm.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0 for generate mode")

    # Initialize vLLM engine
    vllm_kwargs: dict[str, Any] = {
        "model": args.model_name_or_path,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": max(1, int(args.vllm_tensor_parallel_size)),
        "dtype": args.dtype,
        "enforce_eager": args.vllm_enforce_eager,
    }
    if args.vllm_gpu_memory_utilization is not None:
        vllm_kwargs["gpu_memory_utilization"] = float(args.vllm_gpu_memory_utilization)
    if args.vllm_max_model_len is not None:
        vllm_kwargs["max_model_len"] = int(args.vllm_max_model_len)
    else:
        vllm_kwargs["max_model_len"] = int(args.max_length)

    # Initialize vLLM with adaptive gpu_memory_utilization fallback if needed
    util_candidates: List[float] = []
    initial_util = vllm_kwargs.get("gpu_memory_utilization", 0.9)
    try:
        initial_util = float(initial_util)
    except Exception:
        initial_util = 0.9
    backoffs = [initial_util] + [u for u in (0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5) if u < initial_util]
    for u in backoffs:
        if u not in util_candidates:
            util_candidates.append(u)

    last_err: Optional[BaseException] = None
    llm = None
    for u in util_candidates:
        try:
            vllm_kwargs["gpu_memory_utilization"] = u
            llm = VLLM(**vllm_kwargs)
            if initial_util != u:
                print(f"vLLM: falling back gpu_memory_utilization to {u:.2f}")
            break
        except Exception as e:
            last_err = e
            continue
    if llm is None:
        assert last_err is not None
        raise last_err

    sampling_params = VLLMSamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else None,
        skip_special_tokens=True,
        detokenize=True,
    )

    results: List[dict] = []
    all_means: List[float] = []

    # Determine scoring device for HF model
    try:
        param_device = next(model_hf.parameters()).device.type  # type: ignore[attr-defined]
    except Exception:
        param_device = "cpu"

    for i in tqdm(range(0, len(texts), args.batch_size), desc="entropy-gen-vllm"):
        batch_texts = texts[i : i + args.batch_size]
        prompts = prompts_from_batch(tok, batch_texts, args)

        vouts = llm.generate(prompts, sampling_params=sampling_params)

        # HF rescore per-sample (teacher forcing over generated tokens)
        for j, vout in enumerate(vouts):
            # Concatenate prompt and generation
            gen_text = vout.outputs[0].text if vout.outputs else ""

            # Tokenize prompt and full sequence with HF tokenizer
            enc_prompt = tok(prompts[j], return_tensors="pt", add_special_tokens=False)
            enc_full = tok(prompts[j] + gen_text, return_tensors="pt", add_special_tokens=False)

            prompt_len = int(enc_prompt.input_ids.size(-1))
            full_len = int(enc_full.input_ids.size(-1))
            gen_len = max(0, full_len - prompt_len)

            mean_ent_val: float = float("nan")
            ent_list: Optional[List[float]] = None
            if gen_len > 0:
                # Build teacher-forcing prefix: prompt + generated[:-1]
                tf_input_ids = enc_full.input_ids[:, : full_len - 1]
                tf_attn = torch.ones_like(tf_input_ids)
                tf_input_ids = tf_input_ids.to(param_device)
                tf_attn = tf_attn.to(param_device)

                outputs = model_hf(input_ids=tf_input_ids, attention_mask=tf_attn, return_dict=True)
                logits = outputs.logits  # [1, T-1, V]

                # Steps corresponding to generated tokens are indices [prompt_len-1 .. prompt_len+gen_len-2]
                start_idx = max(0, prompt_len - 1)
                end_idx = start_idx + gen_len  # exclusive on slicing
                step_logits = logits[:, start_idx:end_idx, :]
                ent = entropy_from_logits(step_logits).squeeze(0)  # [S]
                mean_ent_val = float(ent.mean().item())
                if args.write_entropies:
                    ent_list = ent.tolist()

            item = {"index": i + j, "generated_text": gen_text, "mean_entropy": float(mean_ent_val)}
            prompt_obj = batch_texts[j]
            if isinstance(prompt_obj, list):
                item["messages"] = prompt_obj
            else:
                item["prompt"] = prompt_obj
            if args.write_entropies and ent_list is not None:
                item["entropies"] = ent_list
            results.append(item)
            if not math.isnan(mean_ent_val):
                all_means.append(mean_ent_val)

    return results, all_means


def save_outputs(results: List[dict], all_means: List[float], args: Args):
    out_path = args.output_path
    if out_path is None:
        mode = "gen" if args.do_generate else "eval"
        out_path = str(Path("output") / f"entropy_{mode}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary
    if len(all_means) == 0:
        return
    mean = float(sum(all_means) / len(all_means))
    var = float(sum((x - mean) ** 2 for x in all_means) / max(1, len(all_means) - 1))
    std = math.sqrt(var)
    sorted_vals = sorted(all_means)
    def pct(p: float) -> float:
        if not sorted_vals:
            return float("nan")
        k = max(0, min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1)))))
        return float(sorted_vals[k])

    summary = {
        "count": len(all_means),
        "mean": mean,
        "std": std,
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
    }
    summary_path = out_path + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote per-sample to {out_path}")
    print(f"Wrote summary to {summary_path}")


def main():
    args = parse_args()
    # Always load HF tokenizer; load HF model for eval and/or rescoring
    model, tok = load_model_and_tokenizer(args)
    texts = read_dataset(args)
    if args.do_generate:
        if args.use_vllm:
            results, all_means = compute_entropy_generate_vllm(model, tok, texts, args)
        else:
            results, all_means = compute_entropy_generate(model, tok, texts, args)
    else:
        results, all_means = compute_entropy_eval(model, tok, texts, args)
    save_outputs(results, all_means, args)


if __name__ == "__main__":
    main()


