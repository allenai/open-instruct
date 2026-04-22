# Copyright 2026 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Offline value-estimation harness.

Two entry points:

- ``make_dataset``: build a parquet of (prompt, ground_truth, rollout, probe_positions, mc_values)
  from DAPO math. 100 prompts each contribute one correct + one incorrect rollout; for each rollout,
  we probe at every 1000-th token and estimate the Monte-Carlo value as ``fraction_correct`` across
  32 continuations.
- ``score_dataset``: load a trained value model (scalar PPO or generative) and score the
  probes using whatever conditioning flags match its training-time conditioning.

A third helper, ``compare_runs``, ingests several ``score_dataset`` parquet outputs and emits a
consolidated comparison table.

All three are CLI-addressable. Shell wrappers live in ``scripts/eval/value_estimation/``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import os
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------
# Common config
# --------------------------------------------------------------------------------------------
@dataclass
class MakeDatasetConfig:
    model_name_or_path: str
    output_path: str
    dataset_name: str = "hamishivi/DAPO-Math-17k-Processed_filtered"
    dataset_split: str = "train"
    num_prompts_to_sample: int = 2000  # Sample this many, keep first 100 with 1 correct + 1 wrong.
    target_num_pairs: int = 100
    rollouts_per_prompt: int = 8
    continuations_per_probe: int = 32
    probe_interval: int = 1000
    min_probe_remaining_tokens: int = 64
    # Probe selection mode: "fixed" (every probe_interval tokens) or "sae" (SAE boundaries
    # from segment_rollout — tokens with prob < sae_threshold, downsampled to max_probes).
    probe_mode: str = "fixed"
    sae_threshold: float = 0.2
    max_probes: int = 16
    max_prompt_length: int = 2048
    max_response_length: int = 8192
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    verifier_name: str = "math"
    keep_continuation_texts: bool = False


@dataclass
class ScoreDatasetConfig:
    input_dataset_path: str
    output_path: str
    value_model_path: str
    value_model_type: str = "scalar"  # one of: scalar, generative
    # Conditioning flags; must match what the value model was trained with.
    value_model_ground_truth_conditioning: bool = False
    gt_conditioning_template: str = "answer_prefix"
    rollout_context_num_siblings: int = 4
    gen_value_conditioning: str = "none"
    gen_value_score_min: float = 0.0
    gen_value_score_max: float = 10.0
    gen_value_max_new_tokens: int = 8
    tokenizer_name_or_path: str | None = None
    run_name: str = "value_estimation_run"
    device: str = "cuda"
    batch_size: int = 4
    # vLLM options used only for generative value models.
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.85


@dataclass
class CompareRunsConfig:
    score_dataset_paths: list[str] = field(default_factory=list)
    output_markdown_path: str | None = None
    output_csv_path: str | None = None


@dataclass
class ConvertCheckpointConfig:
    checkpoint_dir: str  # directory containing value_model.bin
    output_dir: str
    # Path to a full HF model dir for config + tokenizer. Defaults to the parent of checkpoint_dir.
    base_model_path: str | None = None


# --------------------------------------------------------------------------------------------
# make_dataset
# --------------------------------------------------------------------------------------------
def _run_rollouts(
    prompts: list[str],
    *,
    model_name_or_path: str,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    logprobs: bool = False,
) -> list[list[dict[str, Any]]]:
    """Run `n` rollouts per prompt through vLLM. Returns list-of-lists of dicts with keys
    ``token_ids`` (list[int]), ``text`` (str), ``logprobs`` (list[float] | None).
    """
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling = SamplingParams(
        n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, logprobs=1 if logprobs else None
    )
    raw = llm.generate(prompts, sampling)
    result: list[list[dict[str, Any]]] = []
    for out in raw:
        cands: list[dict[str, Any]] = []
        for c in out.outputs:
            lp = None
            if logprobs and getattr(c, "logprobs", None) is not None:
                lp = [next(iter(p.values())).logprob if p else 0.0 for p in c.logprobs]
            cands.append({"token_ids": list(c.token_ids), "text": c.text, "logprobs": lp})
        result.append(cands)
    # Cleanly shut down the LLM engine so the next call can re-init without leaks.
    with contextlib.suppress(Exception):
        del llm
    return result


_VERIFIER_CACHE: dict[str, Any] = {}


def _verify(prediction: str, ground_truth: str, verifier_name: str) -> bool:
    from open_instruct.ground_truth_utils import MathVerifier, StringMatcherVerifier  # noqa: PLC0415

    key = (verifier_name or "math").lower()
    if key not in _VERIFIER_CACHE:
        _VERIFIER_CACHE[key] = MathVerifier() if key in {"math", "strict_math"} else StringMatcherVerifier()
    v = _VERIFIER_CACHE[key]
    return float(v(tokenized_prediction=[], prediction=prediction, label=ground_truth).score) >= 1.0


def make_dataset(cfg: MakeDatasetConfig) -> str:
    """Build the value-estimation dataset described in the plan."""
    import pandas as pd  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger.info(f"Loading dataset {cfg.dataset_name} split={cfg.dataset_split}")
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    num_to_sample = min(cfg.num_prompts_to_sample, len(ds))
    indices = random.sample(range(len(ds)), num_to_sample)
    records = [ds[i] for i in indices]

    def _extract_prompt(row: dict) -> str:
        # DAPO rows typically have `messages` or a `prompt` field; handle both.
        if "messages" in row and isinstance(row["messages"], list) and row["messages"]:
            return row["messages"][-1].get("content", "")
        return row.get("prompt", "")

    def _extract_gt(row: dict) -> str:
        gt = row.get("ground_truth") or row.get("gt") or row.get("answer") or ""
        if isinstance(gt, list):
            gt = gt[0] if gt else ""
        return str(gt)

    prompts = [_extract_prompt(r) for r in records]
    ground_truths = [_extract_gt(r) for r in records]

    logger.info(f"Running {cfg.rollouts_per_prompt} rollouts/prompt for {len(prompts)} prompts")
    rollouts = _run_rollouts(
        prompts,
        model_name_or_path=cfg.model_name_or_path,
        n=cfg.rollouts_per_prompt,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_response_length,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        logprobs=cfg.probe_mode == "sae",
    )

    # Filter to prompts with at least one correct and one incorrect rollout.
    kept = []
    for i, cands in enumerate(rollouts):
        gt = ground_truths[i]
        verdicts = [_verify(c["text"], gt, cfg.verifier_name) for c in cands]
        has_correct = any(verdicts)
        has_incorrect = not all(verdicts)
        if has_correct and has_incorrect:
            kept.append((i, cands, verdicts))
            if len(kept) >= cfg.target_num_pairs:
                break
    logger.info(f"Kept {len(kept)} prompts with at least one correct + incorrect rollout")

    # Build rows: for each kept prompt, pick the first correct + first incorrect rollout.
    # The other rollouts become the sibling_rollouts pool for conditioning variants.
    rows: list[dict[str, Any]] = []
    continuation_prompts: list[str] = []
    continuation_indices: list[tuple[int, int]] = []  # (row_idx, probe_idx)
    for orig_idx, cands, verdicts in kept:
        gt = ground_truths[orig_idx]
        prompt = prompts[orig_idx]
        first_correct = next(i for i, v in enumerate(verdicts) if v)
        first_incorrect = next(i for i, v in enumerate(verdicts) if not v)
        for rollout_idx in (first_correct, first_incorrect):
            main = cands[rollout_idx]
            siblings = [
                {"text": cands[k]["text"], "is_correct": verdicts[k]} for k in range(len(cands)) if k != rollout_idx
            ]
            tokens = main["token_ids"]
            length = len(tokens)
            if cfg.probe_mode == "sae":
                from open_instruct import value_model_utils  # noqa: PLC0415

                raw_boundaries = value_model_utils.segment_rollout(
                    response_tokens=tokens,
                    response_logprobs=main.get("logprobs"),
                    mode="sae",
                    sae_threshold=cfg.sae_threshold,
                    max_segments=cfg.max_probes,
                )
                probe_positions = [
                    t for t in raw_boundaries if (length - t) >= cfg.min_probe_remaining_tokens
                ]
            else:
                probe_positions = [
                    t
                    for t in range(cfg.probe_interval, length, cfg.probe_interval)
                    if (length - t) >= cfg.min_probe_remaining_tokens
                ]
            row = {
                "prompt": prompt,
                "ground_truth": gt,
                "verifier_name": cfg.verifier_name,
                "rollout_text": main["text"],
                "rollout_tokens": tokens,
                "rollout_is_correct": bool(verdicts[rollout_idx]),
                "sibling_rollouts": siblings,
                "probe_positions": probe_positions,
                "mc_values": [],  # filled in below
                "num_continuations": cfg.continuations_per_probe,
            }
            row_idx = len(rows)
            rows.append(row)
            for p_idx, t in enumerate(probe_positions):
                continuation_prompts.append(prompt + main["text"][: _token_to_char_offset(main, t)])
                continuation_indices.append((row_idx, p_idx))

    # Compute MC values per probe by generating 32 continuations in one big vLLM batch.
    if continuation_prompts:
        logger.info(
            f"Running {cfg.continuations_per_probe} continuations for "
            f"{len(continuation_prompts)} probes ({len(rows)} rollouts)"
        )
        conts = _run_rollouts(
            continuation_prompts,
            model_name_or_path=cfg.model_name_or_path,
            n=cfg.continuations_per_probe,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_response_length,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
        )
        for (row_idx, _p_idx), cands in zip(continuation_indices, conts):
            gt = rows[row_idx]["ground_truth"]
            # Reconstruct the partial-response text we spliced (stored in row["rollout_text"] + probe offset).
            # For MC value we need the FULL text after the partial prefix; the continuation's `text` is the
            # model output from the prefix; verify each continuation concatenated with the partial prefix.
            # For simplicity here we treat the continuation.text alone as the "rest of the answer",
            # which is what vLLM returns for a completion-style call.
            verdicts = [_verify(c["text"], gt, rows[row_idx]["verifier_name"]) for c in cands]
            mc = sum(1 for v in verdicts if v) / max(len(verdicts), 1)
            rows[row_idx]["mc_values"].append(float(mc))

    pathlib.Path(os.path.dirname(cfg.output_path) or ".").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(cfg.output_path, index=False)
    logger.info(f"Wrote {len(rows)} rows to {cfg.output_path}")
    return cfg.output_path


def _token_to_char_offset(rollout: dict[str, Any], num_tokens: int) -> int:
    """Approximate char offset for the first ``num_tokens`` of a rollout by proportional split.

    This is a reasonable approximation when the tokenizer is BPE-like. For better fidelity,
    callers can decode via a tokenizer and sum surface forms; we keep this cheap/fast for
    dataset building purposes.
    """
    total_tokens = len(rollout.get("token_ids", []))
    total_chars = len(rollout.get("text", ""))
    if total_tokens == 0 or total_chars == 0:
        return 0
    return min(total_chars, int(total_chars * num_tokens / max(total_tokens, 1)))


# --------------------------------------------------------------------------------------------
# score_dataset
# --------------------------------------------------------------------------------------------
def score_dataset(cfg: ScoreDatasetConfig) -> str:
    import pandas as pd  # noqa: PLC0415
    from scipy.stats import pearsonr, spearmanr  # noqa: PLC0415

    # Accept either a local parquet path or a HuggingFace dataset name (org/repo).
    if os.path.exists(cfg.input_dataset_path):
        df = pd.read_parquet(cfg.input_dataset_path)
    else:
        from datasets import load_dataset as _load_dataset  # noqa: PLC0415

        logger.info(f"Local path not found; loading from HuggingFace: {cfg.input_dataset_path}")
        hf_ds = _load_dataset(cfg.input_dataset_path, split="test")
        df = hf_ds.to_pandas()
    logger.info(f"Loaded {len(df)} rows from {cfg.input_dataset_path}")

    # Conditioning warning: compare training_args.json if present.
    training_args_path = os.path.join(cfg.value_model_path, "..", "training_args.json")
    if os.path.exists(training_args_path):
        with open(training_args_path) as f:
            ta = json.load(f)
        if bool(ta.get("value_model_ground_truth_conditioning", False)) != cfg.value_model_ground_truth_conditioning:
            logger.warning(
                "Conditioning flag mismatch between checkpoint and score_dataset: "
                f"ckpt={ta.get('value_model_ground_truth_conditioning')}, "
                f"score={cfg.value_model_ground_truth_conditioning}."
            )
        if ta.get("gt_conditioning_template") != cfg.gt_conditioning_template:
            logger.warning(
                f"gt_conditioning_template mismatch: ckpt={ta.get('gt_conditioning_template')!r}, "
                f"score={cfg.gt_conditioning_template!r}."
            )

    preds_per_row: list[list[float]] = []
    all_preds: list[float] = []
    all_mc: list[float] = []

    if cfg.value_model_type == "scalar":
        preds_per_row = _score_with_scalar_value(df, cfg)
    elif cfg.value_model_type == "generative":
        preds_per_row = _score_with_generative_value(df, cfg)
    else:
        raise ValueError(f"Unknown value_model_type: {cfg.value_model_type}")

    correct_preds: list[float] = []
    incorrect_preds: list[float] = []
    probe_rows = []
    for i, row in df.iterrows():
        is_correct = bool(row.get("rollout_is_correct"))
        for pos, p, mc in zip(row["probe_positions"], preds_per_row[i], row["mc_values"]):
            all_preds.append(float(p))
            all_mc.append(float(mc))
            if is_correct:
                correct_preds.append(float(p))
            else:
                incorrect_preds.append(float(p))
            probe_rows.append(
                {
                    "run_name": cfg.run_name,
                    "rollout_idx": i,
                    "rollout_is_correct": is_correct,
                    "probe_position": int(pos),
                    "predicted_value": float(p),
                    "mc_value": float(mc),
                }
            )

    # Metrics
    metrics: dict[str, float] = {}
    if all_preds:
        diffs = [a - b for a, b in zip(all_preds, all_mc)]
        metrics["mae"] = float(np.mean([abs(d) for d in diffs]))
        metrics["mse"] = float(np.mean([d**2 for d in diffs]))
        if len(all_preds) > 1:
            try:
                metrics["pearson"] = float(pearsonr(all_preds, all_mc)[0])
                metrics["spearman"] = float(spearmanr(all_preds, all_mc).statistic)
            except Exception:
                pass
        # Calibration bins (deciles of predicted values).
        order = np.argsort(all_preds)
        bin_size = max(1, len(order) // 10)
        for b in range(10):
            chunk = order[b * bin_size : (b + 1) * bin_size] if b < 9 else order[b * bin_size :]
            if len(chunk) == 0:
                continue
            metrics[f"calib_bin_{b}_pred_mean"] = float(np.mean([all_preds[j] for j in chunk]))
            metrics[f"calib_bin_{b}_mc_mean"] = float(np.mean([all_mc[j] for j in chunk]))

    if correct_preds:
        metrics["correct_pred_mean"] = float(np.mean(correct_preds))
    if incorrect_preds:
        metrics["incorrect_pred_mean"] = float(np.mean(incorrect_preds))

    # Write output parquet.
    import pandas as pd  # noqa: PLC0415

    df_out = df.copy()
    df_out["predicted_values"] = preds_per_row
    df_out["run_config"] = [json.dumps(dataclasses.asdict(cfg))] * len(df_out)
    pathlib.Path(os.path.dirname(cfg.output_path) or ".").mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(cfg.output_path, index=False)
    pd.DataFrame(probe_rows).to_csv(cfg.output_path + ".probes.csv", index=False)
    # Write a small JSON summary next to the parquet.
    summary_path = cfg.output_path + ".summary.json"
    summary = {"run_name": cfg.run_name, "value_model_type": cfg.value_model_type, "metrics": metrics}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {len(df_out)} predictions to {cfg.output_path}; metrics: {metrics}")
    return cfg.output_path


def _score_with_scalar_value(df, cfg: ScoreDatasetConfig) -> list[list[float]]:
    """Score probes using a scalar value model loaded via HF."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    import torch.nn as nn  # noqa: PLC0415
    from safetensors.torch import load_file as _load_sf  # noqa: PLC0415

    tok_path = cfg.tokenizer_name_or_path or cfg.value_model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    # lm_head.weight has shape (1, hidden_size) but config.vocab_size is the full vocabulary.
    # Load with ignore_mismatched_sizes so the embedding table loads correctly, then
    # replace lm_head and load its weight manually from the safetensors file.
    value_model = AutoModelForCausalLM.from_pretrained(
        cfg.value_model_path, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True
    )
    hidden_size = value_model.config.hidden_size
    value_model.lm_head = nn.Linear(hidden_size, 1, bias=False, dtype=torch.bfloat16)
    sf_path = os.path.join(cfg.value_model_path, "model.safetensors")
    if os.path.exists(sf_path):
        sd = _load_sf(sf_path)
        if "lm_head.weight" in sd:
            value_model.lm_head.weight.data.copy_(sd["lm_head.weight"].to(torch.bfloat16))
    value_model = value_model.to(cfg.device)
    value_model.eval()
    from open_instruct import value_model_utils  # noqa: PLC0415

    all_preds: list[list[float]] = []
    with torch.no_grad():
        for _, row in df.iterrows():
            prompt = row["prompt"]
            rollout_tokens = list(row["rollout_tokens"])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            cond_ids: list[int] = []
            if cfg.value_model_ground_truth_conditioning:
                sibs = row.get("sibling_rollouts")
                sibs = list(sibs) if sibs is not None else []
                cond_text = value_model_utils.build_conditioning_text(
                    cfg.gt_conditioning_template, row["ground_truth"], siblings=sibs
                )
                cond_ids = tokenizer.encode(cond_text, add_special_tokens=False)
            is_postfix = value_model_utils.is_postfix_template(cfg.gt_conditioning_template)
            preds: list[float] = []
            for t in row["probe_positions"]:
                partial_ids = rollout_tokens[:t]
                all_ids = prompt_ids + cond_ids + partial_ids if is_postfix else cond_ids + prompt_ids + partial_ids
                input_ids = torch.tensor([all_ids[-16384:]], dtype=torch.long).to(cfg.device)
                out = value_model(input_ids=input_ids)
                logits = getattr(out, "logits", out)[:, -1]  # last-token logit
                v = float(logits.squeeze(-1).item())
                preds.append(v)
            all_preds.append(preds)
    return all_preds


def _score_with_generative_value(df, cfg: ScoreDatasetConfig) -> list[list[float]]:
    """Score probes using a generative value model served via vLLM."""
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from open_instruct import value_model_utils  # noqa: PLC0415

    tok_path = cfg.tokenizer_name_or_path or cfg.value_model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    llm = LLM(
        model=cfg.value_model_path,
        tensor_parallel_size=cfg.vllm_tensor_parallel_size,
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=cfg.gen_value_max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    prompts: list[str] = []
    positions: list[tuple[int, int]] = []
    for idx, row in df.iterrows():
        rollout_tokens = list(row["rollout_tokens"])
        for p_idx, t in enumerate(row["probe_positions"]):
            partial = tokenizer.decode(rollout_tokens[:t], skip_special_tokens=False)
            prompt = value_model_utils.build_generative_value_prompt(
                partial,
                conditioning=cfg.gen_value_conditioning,
                ground_truth=row["ground_truth"],
                siblings=row.get("sibling_rollouts") or [],
                score_min=cfg.gen_value_score_min,
                score_max=cfg.gen_value_score_max,
            )
            prompts.append(prompt)
            positions.append((idx, p_idx))

    raw = llm.generate(prompts, sp)
    # Re-collate into per-row lists.
    all_preds: list[list[float]] = [[0.0 for _ in row["probe_positions"]] for _, row in df.iterrows()]
    for out, (row_idx, p_idx) in zip(raw, positions):
        txt = out.outputs[0].text if hasattr(out, "outputs") else ""
        parsed = value_model_utils.parse_generative_value_score(
            txt, score_min=cfg.gen_value_score_min, score_max=cfg.gen_value_score_max
        )
        if parsed is None:
            parsed = 0.5 * (cfg.gen_value_score_min + cfg.gen_value_score_max)
        all_preds[row_idx][p_idx] = value_model_utils.rescale_gen_value_score(
            parsed, cfg.gen_value_score_min, cfg.gen_value_score_max
        )
    return all_preds


# --------------------------------------------------------------------------------------------
# compare_runs
# --------------------------------------------------------------------------------------------
def compare_runs(cfg: CompareRunsConfig) -> str | None:
    import pandas as pd  # noqa: PLC0415

    rows = []
    for p in cfg.score_dataset_paths:
        summary_path = p + ".summary.json"
        if not os.path.exists(summary_path):
            logger.warning(f"Summary missing for {p}, skipping")
            continue
        with open(summary_path) as f:
            s = json.load(f)
        row = {"run_name": s.get("run_name"), "value_model_type": s.get("value_model_type")}
        row.update(s.get("metrics", {}))
        rows.append(row)
    if not rows:
        logger.warning("No runs to compare; skipping")
        return None
    frame = pd.DataFrame(rows)
    md_path: str | None = None
    if cfg.output_csv_path:
        frame.to_csv(cfg.output_csv_path, index=False)
        logger.info(f"Wrote comparison CSV to {cfg.output_csv_path}")
    if cfg.output_markdown_path:
        md = frame.to_markdown(index=False)
        with open(cfg.output_markdown_path, "w") as f:
            f.write(md)
        md_path = cfg.output_markdown_path
        logger.info(f"Wrote comparison markdown to {cfg.output_markdown_path}")
    logger.info("\n" + frame.to_string())
    return md_path


# --------------------------------------------------------------------------------------------
# convert_checkpoint
# --------------------------------------------------------------------------------------------
def convert_checkpoint(cfg: ConvertCheckpointConfig) -> str:
    """Convert a ``value_model.bin`` checkpoint into a HF-loadable directory.

    The value model's ``lm_head`` has shape ``[1, hidden_size]`` rather than
    ``[vocab_size, hidden_size]``.  We set ``vocab_size=1`` and truncate
    ``embed_tokens.weight`` so that ``AutoModelForCausalLM.from_pretrained``
    can load the converted directory directly.
    """
    import shutil  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    ckpt_dir = pathlib.Path(cfg.checkpoint_dir)
    out_dir = pathlib.Path(cfg.output_dir)
    base_dir = pathlib.Path(cfg.base_model_path) if cfg.base_model_path else ckpt_dir.parent

    value_bin = ckpt_dir / "value_model.bin"
    if not value_bin.exists():
        raise FileNotFoundError(f"value_model.bin not found in {ckpt_dir}")

    config_src = base_dir / "config.json"
    if not config_src.exists():
        raise FileNotFoundError(f"config.json not found in {base_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_src) as f:
        cfg_json = json.load(f)
    # Keep vocab_size intact so embed_tokens stays full-size at inference.
    # lm_head.weight has shape (1, hidden) in the checkpoint; the loader handles
    # the mismatch with ignore_mismatched_sizes and loads it manually.
    cfg_json["tie_word_embeddings"] = False
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg_json, f, indent=2)

    sd = torch.load(value_bin, map_location="cpu", weights_only=True)
    sd_mod = {k: v.bfloat16() for k, v in sd.items()}
    save_file(sd_mod, out_dir / "model.safetensors")
    weight_map = {k: "model.safetensors" for k in sd_mod}
    total_size = sum(v.numel() * 2 for v in sd_mod.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
        src = base_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)

    logger.info(f"Converted value checkpoint to {out_dir}")
    return str(out_dir)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _cfg_from_args(cfg_cls, args_ns) -> Any:
    kwargs = {}
    for f in dataclasses.fields(cfg_cls):
        if hasattr(args_ns, f.name):
            v = getattr(args_ns, f.name)
            if v is not None:
                kwargs[f.name] = v
    return cfg_cls(**kwargs)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_make = sub.add_parser("make_dataset", help="Build the value-estimation dataset.")
    for f in dataclasses.fields(MakeDatasetConfig):
        _add_field(p_make, f)

    p_score = sub.add_parser("score_dataset", help="Score the value-estimation dataset with a model.")
    for f in dataclasses.fields(ScoreDatasetConfig):
        _add_field(p_score, f)

    p_cmp = sub.add_parser("compare_runs", help="Aggregate score_dataset outputs into a table.")
    p_cmp.add_argument("--score_dataset_paths", nargs="+", required=True)
    p_cmp.add_argument("--output_markdown_path", default=None)
    p_cmp.add_argument("--output_csv_path", default=None)

    p_conv = sub.add_parser("convert_checkpoint", help="Convert value_model.bin to a HF-loadable directory.")
    for f in dataclasses.fields(ConvertCheckpointConfig):
        _add_field(p_conv, f)

    args = parser.parse_args()
    if args.cmd == "make_dataset":
        cfg = _cfg_from_args(MakeDatasetConfig, args)
        make_dataset(cfg)
    elif args.cmd == "score_dataset":
        cfg = _cfg_from_args(ScoreDatasetConfig, args)
        score_dataset(cfg)
    elif args.cmd == "compare_runs":
        cfg = CompareRunsConfig(
            score_dataset_paths=args.score_dataset_paths,
            output_markdown_path=args.output_markdown_path,
            output_csv_path=args.output_csv_path,
        )
        compare_runs(cfg)
    elif args.cmd == "convert_checkpoint":
        cfg = _cfg_from_args(ConvertCheckpointConfig, args)
        convert_checkpoint(cfg)


def _add_field(parser, f) -> None:
    kwargs: dict[str, Any] = {"dest": f.name}
    if f.default is not dataclasses.MISSING:
        kwargs["default"] = None  # use None so CLI absence means "use default"
    if f.type is bool or f.type == "bool":
        kwargs["action"] = "store_true"
    elif f.type in (int, "int"):
        kwargs["type"] = int
    elif f.type in (float, "float"):
        kwargs["type"] = float
    elif f.type in (str, "str") or getattr(f.type, "__name__", None) == "str":
        kwargs["type"] = str
    else:
        kwargs["type"] = str  # fallback; works for list[str] via comma-separated
    parser.add_argument(f"--{f.name}", **kwargs)


if __name__ == "__main__":
    main()
