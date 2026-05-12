#!/usr/bin/env python
# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

"""Sample teacher logits with vLLM and save compressed top-k logprobs."""

import json
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import tqdm
import vllm
import yaml
from vllm.logprobs import FlatLogprobs

from open_instruct import logger_utils
from open_instruct.dataset_transformation import INPUT_IDS_KEY, TokenizerConfig, get_cached_dataset_tulu
from open_instruct.distillkit.compression import DistributionQuantizationConfig, LogprobCompressor
from open_instruct.distillkit.sample_common import StreamingParquetWriter, compressed_logit_schema
from open_instruct.utils import ArgumentParserPlus

logger = logger_utils.setup_logger(__name__)

LOGIT_SAMPLING_TARGET_COLUMNS = ["input_ids", "attention_mask", "labels", "messages", "text"]


@dataclass
class SampleLogitsArguments:
    model_name_or_path: str
    compression_config_path: str

    model_revision: str | None = None
    model_dtype: str | None = None
    quantization: str | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_expert_parallel: bool = False
    gpu_memory_utilization: float = 0.9

    dataset_mixer_list: list[str] = field(default_factory=lambda: ["allenai/tulu-3-sft-personas-algebra", "1.0"])
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )
    dataset_target_columns: list[str] = field(default_factory=lambda: LOGIT_SAMPLING_TARGET_COLUMNS.copy())
    dataset_cache_mode: Literal["hf", "local"] = "local"
    dataset_local_cache_dir: str = "local_dataset_cache"
    dataset_config_hash: str | None = None
    dataset_skip_cache: bool = False

    max_seq_length: int = 2048
    max_samples: int | None = None
    seed: int = 42
    num_unique_prompts: int = 256
    max_workers: int | None = None

    auto_vocab_size: bool = True

    output_dir: str = "output/logits"
    exp_name: str = "sample_logits_vllm"


def process_prompt_logprobs(prompt_logprobs: FlatLogprobs, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract dense top-k token ids/logprobs from flat vLLM prompt logprob outputs."""
    # some vLLM backends include an empty first slot for prompt position 0
    # detect and skip it so tensors align to real prompt token positions
    start_pos = 0
    if len(prompt_logprobs) > 0:
        first_start = prompt_logprobs.start_indices[0]
        first_end = prompt_logprobs.end_indices[0]
        if first_end - first_start == 0:
            start_pos = 1

    num_prompt_tokens = len(prompt_logprobs) - start_pos
    if num_prompt_tokens <= 0:
        return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.float32)

    top_indices = torch.empty((num_prompt_tokens, k), dtype=torch.long, device="cpu")
    top_values = torch.full((num_prompt_tokens, k), fill_value=float("-inf"), dtype=torch.float32, device="cpu")

    seq_ids = []
    rank_ids = []
    token_ids_to_copy = []
    logprobs_to_copy = []

    # FlatLogprobs has flattened arrays with per-position start/end pointers
    # use this to rebuild token position / rank coordinates
    for pos_id in range(start_pos, len(prompt_logprobs)):
        seq_id = pos_id - start_pos
        start_idx = prompt_logprobs.start_indices[pos_id]
        end_idx = prompt_logprobs.end_indices[pos_id]
        for i in range(start_idx, end_idx):
            rank = prompt_logprobs.ranks[i]
            # vLLM rank is int or None, so we guard against None in the conditional
            # We do not break on rank > k because we are not traversing in rank order
            if rank is None or rank > k:
                continue
            seq_ids.append(seq_id)
            rank_ids.append(rank - 1)
            token_ids_to_copy.append(prompt_logprobs.token_ids[i])
            logprobs_to_copy.append(prompt_logprobs.logprobs[i])

    # This can be empty if all candidate entries had rank=None or rank>k.
    if token_ids_to_copy:
        seq_idx_tensor = torch.tensor(seq_ids, dtype=torch.long)
        rank_idx_tensor = torch.tensor(rank_ids, dtype=torch.long)
        top_indices[seq_idx_tensor, rank_idx_tensor] = torch.tensor(token_ids_to_copy, dtype=top_indices.dtype)
        top_values[seq_idx_tensor, rank_idx_tensor] = torch.tensor(logprobs_to_copy, dtype=top_values.dtype)

    return top_indices, top_values


def load_and_validate_compression_config(
    compression_config_path: str, tokenizer: Any, auto_vocab_size: bool
) -> DistributionQuantizationConfig:
    """Load compression config and align vocab sizing with tokenizer when requested."""
    with Path(compression_config_path).open(encoding="utf-8") as f:
        cfg = DistributionQuantizationConfig.from_dict(yaml.safe_load(f))

    tok_vocab = tokenizer.get_vocab()
    tok_vocab_size = max(len(tok_vocab) + 1, max(tok_vocab.values()))
    if cfg.d == tok_vocab_size:
        return cfg
    if auto_vocab_size:
        cfg.d = tok_vocab_size
        logger.warning(f"Auto-set compressor vocab size to {tok_vocab_size}.")
    elif cfg.d < tok_vocab_size:
        logger.error(f"Compression vocab size too small: {cfg.d} < {tok_vocab_size}.")
        sys.exit(-1)
    elif abs(cfg.d - tok_vocab_size) > 32:
        logger.warning(f"Compression vocab size ({cfg.d}) is larger than tokenizer size ({tok_vocab_size}).")
    return cfg


def build_sampling_params(k: int) -> vllm.SamplingParams:
    """Build vLLM sampling parameters for prompt-logprob extraction only.

    This path is intentionally deterministic and returns one-token generations
    while requesting top-k prompt logprobs in flat format.
    """
    return vllm.SamplingParams(
        top_k=-1,
        prompt_logprobs=k,
        logprobs=k,
        flat_logprobs=True,
        max_tokens=1,
        detokenize=False,
        skip_special_tokens=False,
    )


def prepare_output_dir(output_dir: str, exp_name: str, cfg: DistributionQuantizationConfig) -> Path:
    """Create output directory and persist effective compression config."""
    output_path = Path(output_dir) / exp_name
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "compression_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    return output_path


def process_and_write_sample(
    req_out: vllm.RequestOutput,
    input_ids_sample: list[int],
    messages_sample: list[dict] | None,
    text_sample: str | None,
    writer: StreamingParquetWriter,
    compressor: LogprobCompressor,
    k: int,
) -> None:
    """Convert one vLLM request output to compressed row format and enqueue write."""
    if req_out.prompt_logprobs is None:
        raise ValueError("vLLM output is missing prompt_logprobs.")
    if not isinstance(req_out.prompt_logprobs, FlatLogprobs):
        raise ValueError("Expected FlatLogprobs from vLLM when flat_logprobs=True.")
    top_indices, top_values = process_prompt_logprobs(req_out.prompt_logprobs, k=k)
    top_indices.unsqueeze_(0)
    top_values.unsqueeze_(0)

    row_out = compressor.compress_from_sparse(top_indices, top_values)
    writer.write(
        {
            "input_ids": input_ids_sample if isinstance(input_ids_sample, list) else input_ids_sample.tolist(),
            "compressed_logprobs": row_out["compressed_logprobs"].cpu().squeeze(0).tolist(),
            "bytepacked_indices": row_out["bytepacked_indices"].cpu().squeeze(0).tolist(),
            "messages": json.dumps(messages_sample) if messages_sample else "",
            "text": text_sample or "",
        }
    )


def iter_request_contexts(
    batch_slice: dict[str, Any], outputs: list[vllm.RequestOutput]
) -> Iterator[tuple[vllm.RequestOutput, list[int], list[dict] | None, str | None]]:
    """Yield aligned vLLM outputs and source dataset context for each prompt."""
    batch_input_ids = batch_slice[INPUT_IDS_KEY]
    batch_messages = batch_slice.get("messages", [None] * len(batch_input_ids))
    batch_text = batch_slice.get("text", [None] * len(batch_input_ids))
    for idx, req_out in enumerate(outputs):
        yield req_out, batch_input_ids[idx], batch_messages[idx], batch_text[idx]


def run_sampling_loop(
    llm: vllm.LLM,
    sampling_params: vllm.SamplingParams,
    train_dataset: Any,
    output_dir: Path,
    num_unique_prompts: int,
    max_workers: int | None,
    compressor: LogprobCompressor,
    k: int,
) -> None:
    """Run batched vLLM generation and stream compressed outputs to parquet shards."""
    with (
        StreamingParquetWriter(
            str(output_dir),
            schema=compressed_logit_schema(),
            file_max_rows=num_unique_prompts,
            queue_maxsize=num_unique_prompts * 2,
        ) as writer,
        ThreadPoolExecutor(max_workers=max_workers) as executor,
    ):
        futures = []
        for i0 in tqdm.tqdm(range(0, len(train_dataset), num_unique_prompts), desc="Logit Batches"):
            batch_slice = train_dataset[i0 : i0 + num_unique_prompts]

            # Convert HF dataset rows into vLLM prompt_token_ids payloads.
            prompts = [
                {"prompt_token_ids": x.tolist() if isinstance(x, torch.Tensor) else list(x)}
                for x in batch_slice[INPUT_IDS_KEY]
            ]
            outputs = llm.generate(prompts, sampling_params=sampling_params)
            for req_out, input_ids_sample, messages_sample, text_sample in iter_request_contexts(batch_slice, outputs):
                futures.append(
                    executor.submit(
                        process_and_write_sample,
                        req_out,
                        input_ids_sample,
                        messages_sample,
                        text_sample,
                        writer,
                        compressor,
                        k,
                    )
                )
            # Keep only a bounded number of pending writes
            # Otherwise generation can outpace writes and consume too much memory
            while len(futures) > num_unique_prompts * 2:
                futures.pop(0).result()
        for future in futures:
            future.result()


def main(args: SampleLogitsArguments, tc: TokenizerConfig) -> None:
    """Run vLLM prompt-logprob sampling and write compressed outputs to parquet."""
    tc.tokenizer_revision = args.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    tokenizer = tc.tokenizer
    cfg = load_and_validate_compression_config(args.compression_config_path, tokenizer, args.auto_vocab_size)
    k = cfg.k

    train_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=args.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=args.dataset_transform_fn,
        transform_fn_args=[{"max_seq_length": args.max_seq_length}, {}],
        target_columns=args.dataset_target_columns,
        dataset_cache_mode=args.dataset_cache_mode,
        dataset_config_hash=args.dataset_config_hash,
        hf_entity=None,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
    )
    train_dataset = train_dataset.shuffle(seed=args.seed, keep_in_memory=True)
    if args.max_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))

    train_dataset.set_format(type="pt", columns=[INPUT_IDS_KEY], output_all_columns=True)

    for i, sample in enumerate(train_dataset):
        if len(sample[INPUT_IDS_KEY]) > args.max_seq_length:
            logger.error(f"Sample {i} has {len(sample[INPUT_IDS_KEY])} tokens > max_seq_length={args.max_seq_length}")
            sys.exit(-1)

    llm_kwargs: dict[str, Any] = {
        "model": args.model_name_or_path,
        "tokenizer": tc.tokenizer_name_or_path,
        "trust_remote_code": tc.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "enable_expert_parallel": args.enable_expert_parallel,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_logprobs": k,
        "logprobs_mode": "raw_logprobs",
        "max_model_len": args.max_seq_length,
    }
    if args.model_dtype is not None:
        llm_kwargs["dtype"] = args.model_dtype
    if args.quantization is not None:
        llm_kwargs["quantization"] = args.quantization
    llm = vllm.LLM(**llm_kwargs)
    compressor = LogprobCompressor(config=cfg)
    sampling_params = build_sampling_params(k)
    output_dir = prepare_output_dir(args.output_dir, args.exp_name, cfg)
    run_sampling_loop(
        llm=llm,
        sampling_params=sampling_params,
        train_dataset=train_dataset,
        output_dir=output_dir,
        num_unique_prompts=args.num_unique_prompts,
        max_workers=args.max_workers,
        compressor=compressor,
        k=k,
    )
    logger.info(f"Logits saved to {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParserPlus((SampleLogitsArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
