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
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import tqdm
import vllm
import yaml
from vllm.logprobs import FlatLogprobs, PromptLogprobs

from open_instruct import logger_utils
from open_instruct.dataset_transformation import INPUT_IDS_KEY, TokenizerConfig, get_cached_dataset_tulu
from open_instruct.distillkit.compression import DistributionQuantizationConfig, LogprobCompressor
from open_instruct.distillkit.sample_common import StreamingParquetWriter, compressed_logit_schema
from open_instruct.utils import ArgumentParserPlus

logger = logger_utils.setup_logger(__name__)

LOGIT_SAMPLING_TARGET_COLUMNS = ["input_ids", "attention_mask", "labels", "messages", "text"]


@dataclass
class SampleLogitsArguments:
    model_name_or_path: str = field(metadata={"help": "Path to model to sample logits from."})
    compression_config: str = field(metadata={"help": "Path to compression config YAML file."})

    model_revision: str | None = field(default=None, metadata={"help": "Optional model revision."})
    dtype: str | None = field(default=None, metadata={"help": "Model dtype (e.g., bfloat16)."})
    quantization: str | None = field(default=None, metadata={"help": "Optional vLLM quantization mode."})
    tensor_parallel_size: int = field(default=1, metadata={"help": "Tensor parallel world size."})
    pipeline_parallel_size: int = field(default=1, metadata={"help": "Pipeline parallel world size."})
    enable_expert_parallel: bool = field(default=False, metadata={"help": "Enable expert parallelism for MoE."})
    gpu_memory_utilization: float = field(default=0.9, metadata={"help": "Target GPU memory utilization."})

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

    max_seq_length: int = field(default=2048, metadata={"help": "Max context length for tokenization and vLLM."})
    max_samples: int | None = field(default=None, metadata={"help": "Optional max samples to process."})
    seed: int = field(default=42, metadata={"help": "Dataset shuffle seed."})
    macrobatch_size: int = field(default=256, metadata={"help": "Number of prompts per vLLM call."})
    max_workers: int | None = field(default=None, metadata={"help": "Background CPU writer workers."})

    auto_vocab_size: bool = field(default=True, metadata={"help": "Auto-fix config vocab size to tokenizer vocab."})
    use_flat_logprobs: bool = field(default=True, metadata={"help": "Use FlatLogprobs path when available."})

    output_dir: str = field(default="output/logits", metadata={"help": "Output root directory."})
    exp_name: str = field(default="sample_logits_vllm", metadata={"help": "Output experiment name."})


def process_prompt_logprobs(prompt_logprobs: PromptLogprobs, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract dense top-k token ids/logprobs from vLLM prompt logprob outputs."""
    if isinstance(prompt_logprobs, FlatLogprobs):
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
                if rank is None or rank > k:
                    continue
                seq_ids.append(seq_id)
                rank_ids.append(rank - 1)
                token_ids_to_copy.append(prompt_logprobs.token_ids[i])
                logprobs_to_copy.append(prompt_logprobs.logprobs[i])

        if seq_ids:
            seq_idx_tensor = torch.tensor(seq_ids, dtype=torch.long)
            rank_idx_tensor = torch.tensor(rank_ids, dtype=torch.long)
            top_indices[seq_idx_tensor, rank_idx_tensor] = torch.tensor(token_ids_to_copy, dtype=top_indices.dtype)
            top_values[seq_idx_tensor, rank_idx_tensor] = torch.tensor(logprobs_to_copy, dtype=top_values.dtype)

        return top_indices, top_values

    valid_logprobs = [lp for lp in prompt_logprobs]
    if valid_logprobs and (valid_logprobs[0] is None or len(valid_logprobs[0]) < 1):
        valid_logprobs.pop(0)
    if not valid_logprobs:
        return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.float32)

    top_indices = torch.empty((len(valid_logprobs), k), dtype=torch.long)
    top_values = torch.full((len(valid_logprobs), k), fill_value=float("-inf"), dtype=torch.float32)
    for seq_id, logprobs in enumerate(valid_logprobs):
        if logprobs is None:
            raise ValueError(f"Missing logprobs for token at position {seq_id + 1}.")
        for tok_id, logprob in logprobs.items():
            if logprob.rank is None or logprob.rank > k:
                continue
            top_indices[seq_id, logprob.rank - 1] = tok_id
            top_values[seq_id, logprob.rank - 1] = logprob.logprob
    return top_indices, top_values


def main(args: SampleLogitsArguments, tc: TokenizerConfig) -> None:
    """Run vLLM prompt-logprob sampling and write compressed outputs to parquet."""
    tc.tokenizer_revision = args.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    tokenizer = tc.tokenizer

    with open(args.compression_config, encoding="utf-8") as f:
        cfg = DistributionQuantizationConfig.from_dict(yaml.safe_load(f))
    k = cfg.k

    tok_vocab = tokenizer.get_vocab()
    tok_vocab_size = max(len(tok_vocab) + 1, max(tok_vocab.values()))
    if cfg.d != tok_vocab_size:
        if args.auto_vocab_size:
            cfg.d = tok_vocab_size
            logger.warning(f"Auto-set compressor vocab size to {tok_vocab_size}.")
        elif cfg.d < tok_vocab_size:
            logger.error(f"Compression vocab size too small: {cfg.d} < {tok_vocab_size}.")
            sys.exit(-1)
        elif abs(cfg.d - tok_vocab_size) > 32:
            logger.warning(f"Compression vocab size ({cfg.d}) is larger than tokenizer size ({tok_vocab_size}).")

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
    if args.dtype is not None:
        llm_kwargs["dtype"] = args.dtype
    if args.quantization is not None:
        llm_kwargs["quantization"] = args.quantization
    llm = vllm.LLM(**llm_kwargs)
    compressor = LogprobCompressor(config=cfg)

    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        top_p=1.0,
        min_p=0.0,
        top_k=-1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        prompt_logprobs=k,
        logprobs=k,
        flat_logprobs=args.use_flat_logprobs,
        max_tokens=1,
        detokenize=False,
        skip_special_tokens=False,
    )

    def process_and_write_sample(
        req_out: vllm.RequestOutput,
        input_ids_sample: list[int],
        messages_sample: list[dict] | None,
        text_sample: str | None,
        writer: StreamingParquetWriter,
    ) -> None:
        if req_out.prompt_logprobs is None:
            raise ValueError("vLLM output is missing prompt_logprobs.")
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

    output_dir = os.path.join(args.output_dir, args.exp_name)
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "compression_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

        with (
            StreamingParquetWriter(
                output_dir,
                schema=compressed_logit_schema(),
                file_max_rows=args.macrobatch_size,
                queue_maxsize=args.macrobatch_size * 2,
            ) as writer,
            ThreadPoolExecutor(max_workers=args.max_workers) as executor,
        ):
            futures = []
            for i0 in tqdm.tqdm(range(0, len(train_dataset), args.macrobatch_size), desc="Logit Batches"):
                batch_slice = train_dataset[i0 : i0 + args.macrobatch_size]
                batch_input_ids = batch_slice[INPUT_IDS_KEY]
                batch_messages = batch_slice.get("messages", [None] * len(batch_input_ids))
                batch_text = batch_slice.get("text", [None] * len(batch_input_ids))

                # Convert HF dataset rows into vLLM prompt_token_ids payloads.
                prompts = [
                    {"prompt_token_ids": x.tolist() if hasattr(x, "tolist") else list(x)} for x in batch_input_ids
                ]
                outputs = llm.generate(prompts, sampling_params=sampling_params)
                for idx, req_out in enumerate(outputs):
                    futures.append(
                        executor.submit(
                            process_and_write_sample,
                            req_out,
                            batch_input_ids[idx],
                            batch_messages[idx] if batch_messages else None,
                            batch_text[idx] if batch_text else None,
                            writer,
                        )
                    )
                while len(futures) > args.macrobatch_size * 2:
                    futures.pop(0).result()
            for future in futures:
                future.result()
        logger.info(f"Logits saved to {output_dir}")
    finally:
        del llm


if __name__ == "__main__":
    parser = ArgumentParserPlus((SampleLogitsArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
