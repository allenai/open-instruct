# !/usr/bin/env python
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

"""
SFT script for OLMo models using OLMo-core training infrastructure.

Usage:
    torchrun --nproc_per_node=8 open_instruct/olmo_core_finetune.py \
        --model_name_or_path allenai/OLMo-2-0325-32B-DPO \
        --mixer_list allenai/tulu-3-sft-olmo-2-mixture 1.0 \
        --max_seq_length 4096 \
        --learning_rate 8e-5 \
        --num_epochs 3
"""

import dataclasses
import datetime
import glob
import gzip
import hashlib
import json
import os
import pathlib
import re
import shlex
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from olmo_core import data as oc_data
from olmo_core import optim
from olmo_core.config import DType
from olmo_core.data import utils as oc_data_utils
from olmo_core.distributed import parallel
from olmo_core.distributed.utils import is_distributed
from olmo_core.train import Duration, LoadStrategy, TrainerConfig, callbacks, teardown_training_environment
from olmo_core.train import train_module as train_module_lib
from olmo_core.train.checkpoint import CheckpointerConfig

from open_instruct import dataset_transformation, logger_utils, numpy_dataset_conversion, olmo_core_utils, utils

logger = logger_utils.setup_logger(__name__)


# Must stay strictly below CheckpointConfig.checkpointing_steps (500), which olmo-core
# requires;
_DEFAULT_EPHEMERAL_SAVE_INTERVAL = 250

_NUMPY_SFT_SUBDIR = "numpy_sft"

_PART_INDEX_RE = re.compile(r"_part_(\d+)\.")


def _chunk_indices(numpy_dir: str, pattern: str) -> set[int]:
    indices = set()
    for path in glob.glob(os.path.join(numpy_dir, pattern)):
        match = _PART_INDEX_RE.search(os.path.basename(path))
        if match is not None:
            indices.add(int(match.group(1)))
    return indices


def _numpy_dir_is_populated(numpy_dir: str, require_statistics: bool = False) -> bool:
    """Return True only if every chunk has token_ids, labels_mask, and metadata.

    With `require_statistics`, also require `dataset_statistics.json`, which the conversion writes
    last, so a run interrupted after the parts but before the statistics counts as unpopulated.
    """
    if require_statistics and not os.path.exists(os.path.join(numpy_dir, "dataset_statistics.json")):
        return False
    token_chunks = _chunk_indices(numpy_dir, numpy_dataset_conversion.TOKEN_IDS_NPY_GLOB)
    if not token_chunks:
        return False
    labels_chunks = _chunk_indices(numpy_dir, numpy_dataset_conversion.LABELS_MASK_NPY_GLOB)
    metadata_chunks = _chunk_indices(numpy_dir, numpy_dataset_conversion.TOKEN_IDS_METADATA_GLOB)
    return token_chunks == labels_chunks == metadata_chunks


def _seed_cache_suffix(seed: int, max_seq_length: int) -> str:
    return hashlib.sha256(f"{seed}:{max_seq_length}".encode()).hexdigest()[:8]


# Appended to the numpy cache directory name for caches written with row-aligned parts, so they
# never share a directory with the 1 GiB mid-row-cut caches the default writes under the same hash.
_ROW_ALIGNED_CACHE_SUFFIX = "-rowaligned"


def _numpy_cache_dir(
    local_cache_dir: str, cache_hash: str, seed: int, max_seq_length: int, row_aligned_parts: bool
) -> str:
    """The cache directory. It depends on the file layout, not on how training reads boundaries."""
    name = f"{cache_hash}-{_seed_cache_suffix(seed, max_seq_length)}"
    if row_aligned_parts:
        name += _ROW_ALIGNED_CACHE_SUFFIX
    return os.path.join(local_cache_dir, _NUMPY_SFT_SUBDIR, name)


def _check_row_aligned_cache(numpy_dir: str) -> None:
    """Raise unless `numpy_dir` was written with row-aligned parts: one metadata line per row."""
    stats_path = os.path.join(numpy_dir, "dataset_statistics.json")
    with open(stats_path) as f:
        stats = json.load(f)
    if not stats["configuration"].get(numpy_dataset_conversion.ROW_ALIGNED_PARTS_KEY, False):
        raise ValueError(f"{numpy_dir} was not written with --row_aligned_parts ({stats_path} has no marker).")
    num_lines = 0
    for path in glob.glob(os.path.join(numpy_dir, numpy_dataset_conversion.TOKEN_IDS_METADATA_GLOB)):
        with gzip.open(path, "rt") as f:
            num_lines += sum(1 for _ in f)
    num_rows = stats["overall_statistics"]["total_instances"]
    if num_lines != num_rows:
        raise ValueError(
            f"{numpy_dir} has {num_lines:,} metadata lines for {num_rows:,} rows, so some row spans two parts."
        )


def _olmo_core_supports_metadata_boundaries() -> bool:
    """Whether the installed OLMo-core can take packing boundaries from the metadata files.

    `use_array_if_local` reached `NumpyPackedFSLDatasetConfig` in allenai/OLMo-core#843.
    """
    return "use_array_if_local" in {field.name for field in dataclasses.fields(oc_data.NumpyPackedFSLDatasetConfig)}


def _document_boundary_kwargs(document_boundaries_from_metadata: bool, row_aligned_parts: bool) -> dict[str, Any]:
    """Extra `NumpyPackedFSLDatasetConfig` arguments for the requested boundary source.

    Empty by default, so the config, its fingerprint and OLMo-core's packing caches are unchanged.
    """
    if not document_boundaries_from_metadata:
        return {}
    if not row_aligned_parts:
        # A mid-row-cut cache has two metadata lines for each cut row, so with metadata boundaries
        # that row would train as two documents, the second without its prompt.
        raise ValueError("--document_boundaries_from_metadata requires --row_aligned_parts.")
    if not _olmo_core_supports_metadata_boundaries():
        raise ValueError(
            "--document_boundaries_from_metadata needs an OLMo-core whose NumpyPackedFSLDatasetConfig "
            "has use_array_if_local (allenai/OLMo-core#843); the installed one does not."
        )
    return {"use_array_if_local": False}


@dataclasses.dataclass
class _NumpyDirStats:
    num_parts: int
    num_tokens: int
    num_loss_tokens: int
    num_docs: int = 0
    num_docs_over_length: int = 0
    num_tokens_truncated: int = 0


# Element-wise checks of extra dirs read this many tokens at a time, to bound memory.
_VALIDATION_BLOCK_TOKENS = 1 << 26


def _part_files(numpy_dir: str, pattern: str) -> dict[str, str]:
    """Map each part's index string, as written (e.g. "0003"), to its path."""
    out = {}
    for path in glob.glob(os.path.join(numpy_dir, pattern)):
        match = _PART_INDEX_RE.search(os.path.basename(path))
        if match is not None:
            out[match.group(1)] = path
    return out


def _count_mask_true(path: str, num_elements: int) -> int:
    """Count True entries of a one-byte label mask, raising if any entry is not 0 or 1."""
    mask = np.memmap(path, mode="r", dtype=np.uint8, shape=(num_elements,))
    total = 0
    for start in range(0, num_elements, _VALIDATION_BLOCK_TOKENS):
        block = np.asarray(mask[start : start + _VALIDATION_BLOCK_TOKENS])
        if block.max() > 1:
            raise ValueError(f"{path} is not a bool/uint8 label mask (found value {int(block.max())}).")
        total += int(np.count_nonzero(block))
    return total


def _check_extra_numpy_dir(
    numpy_dir: str,
    tokenizer: oc_data.TokenizerConfig,
    token_dtype: type[np.unsignedinteger],
    sequence_length: int,
    require_metadata: bool,
) -> _NumpyDirStats:
    """Validate one `--extra_numpy_dirs` entry and count its tokens.

    The dir must hold the layout `numpy_dataset_conversion` writes: raw `token_ids_part_*.npy` in the
    dtype OLMo-core reads (`token_dtype`), and a one-byte `labels_mask_part_*.npy` with the same part
    index and element count. OLMo-core sorts each glob's matches and pairs token and mask files by
    position, so equal sets of index strings keep the two lists aligned.
    """
    if not os.path.isdir(numpy_dir):
        raise FileNotFoundError(f"--extra_numpy_dirs entry {numpy_dir} is not a local directory.")
    token_parts = _part_files(numpy_dir, numpy_dataset_conversion.TOKEN_IDS_NPY_GLOB)
    mask_parts = _part_files(numpy_dir, numpy_dataset_conversion.LABELS_MASK_NPY_GLOB)
    if not token_parts:
        raise FileNotFoundError(f"{numpy_dir} has no {numpy_dataset_conversion.TOKEN_IDS_NPY_GLOB} parts.")
    if set(token_parts) != set(mask_parts):
        raise ValueError(
            f"{numpy_dir}: token parts without a label mask: {sorted(set(token_parts) - set(mask_parts))}; "
            f"label masks without a token part: {sorted(set(mask_parts) - set(token_parts))}."
        )
    if require_metadata and set(_part_files(numpy_dir, numpy_dataset_conversion.TOKEN_IDS_METADATA_GLOB)) != set(
        token_parts
    ):
        raise ValueError(
            f"{numpy_dir}: --document_boundaries_from_metadata needs a "
            f"{numpy_dataset_conversion.TOKEN_IDS_METADATA_GLOB} file for every token part."
        )
    dtype_name = np.dtype(token_dtype).name
    itemsize = np.dtype(token_dtype).itemsize
    stats = _NumpyDirStats(num_parts=len(token_parts), num_tokens=0, num_loss_tokens=0)
    for index in sorted(token_parts):
        token_path, mask_path = token_parts[index], mask_parts[index]
        token_bytes = os.path.getsize(token_path)
        if token_bytes == 0 or token_bytes % itemsize:
            raise ValueError(
                f"{token_path} has {token_bytes:,} bytes, not a positive multiple of {itemsize} "
                f"({dtype_name}, the dtype OLMo-core reads for vocab size {tokenizer.vocab_size:,})."
            )
        num_tokens = token_bytes // itemsize
        mask_bytes = os.path.getsize(mask_path)
        if mask_bytes != num_tokens:
            raise ValueError(
                f"{mask_path} has {mask_bytes:,} one-byte elements but {token_path} has {num_tokens:,} "
                f"{dtype_name} tokens; tokens must be {dtype_name} and the mask bool/uint8, one entry per token."
            )
        tokens = np.memmap(token_path, mode="r", dtype=token_dtype, shape=(num_tokens,))
        for start in range(0, num_tokens, _VALIDATION_BLOCK_TOKENS):
            max_id = int(np.asarray(tokens[start : start + _VALIDATION_BLOCK_TOKENS]).max())
            if max_id >= tokenizer.vocab_size:
                raise ValueError(
                    f"{token_path} has token id {max_id:,} >= vocab size {tokenizer.vocab_size:,}; "
                    f"was it written as {dtype_name}?"
                )
        if int(tokens[-1]) != tokenizer.eos_token_id:
            raise ValueError(
                f"{token_path} does not end with EOS ({tokenizer.eos_token_id}); OLMo-core's EOS scan "
                "drops the tokens after a file's last EOS."
            )
        # The boundaries OLMo-core's packer finds in a local file.
        doc_lengths = np.array(
            [
                end - start
                for start, end in oc_data_utils.iter_document_indices(
                    token_path,
                    use_array_if_local=True,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    dtype=token_dtype,
                )
            ],
            dtype=np.int64,
        )
        over = doc_lengths[doc_lengths > sequence_length]
        stats.num_docs += len(doc_lengths)
        stats.num_docs_over_length += len(over)
        stats.num_tokens_truncated += int((over - sequence_length).sum())
        stats.num_tokens += num_tokens
        stats.num_loss_tokens += _count_mask_true(mask_path, num_tokens)
    return stats


def _cache_dir_stats(numpy_dir: str, token_dtype: type[np.unsignedinteger]) -> _NumpyDirStats:
    """Token and loss-token counts of the chat cache, the latter from its statistics file if present."""
    token_parts = _part_files(numpy_dir, numpy_dataset_conversion.TOKEN_IDS_NPY_GLOB)
    num_tokens = sum(os.path.getsize(path) for path in token_parts.values()) // np.dtype(token_dtype).itemsize
    stats_path = os.path.join(numpy_dir, "dataset_statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            num_loss_tokens = int(json.load(f)["overall_statistics"]["trainable_tokens"])
    else:
        mask_parts = _part_files(numpy_dir, numpy_dataset_conversion.LABELS_MASK_NPY_GLOB)
        num_loss_tokens = sum(_count_mask_true(path, os.path.getsize(path)) for path in mask_parts.values())
    return _NumpyDirStats(num_parts=len(token_parts), num_tokens=num_tokens, num_loss_tokens=num_loss_tokens)


def _share(part: int, whole: int) -> str:
    return f"{100 * part / whole:.2f}%" if whole else "n/a"


def _check_extra_numpy_dirs(
    extra_numpy_dirs: Sequence[str],
    cache_dir: str,
    tokenizer: oc_data.TokenizerConfig,
    sequence_length: int,
    require_metadata: bool,
    log: bool = True,
) -> dict[str, _NumpyDirStats]:
    """Validate every extra dir, then log token and loss-token totals for the cache and the extras."""
    if not extra_numpy_dirs:
        return {}
    if len(set(extra_numpy_dirs)) != len(extra_numpy_dirs):
        raise ValueError(f"--extra_numpy_dirs lists a directory more than once: {list(extra_numpy_dirs)}")
    token_dtype = oc_data.NumpyPackedFSLDatasetConfig(tokenizer=tokenizer, paths=[], sequence_length=1).get_dtype()
    stats = {
        numpy_dir: _check_extra_numpy_dir(numpy_dir, tokenizer, token_dtype, sequence_length, require_metadata)
        for numpy_dir in extra_numpy_dirs
    }
    if log:
        for numpy_dir, s in stats.items():
            logger.info(
                f"Extra numpy dir {numpy_dir}: {s.num_parts} parts, {s.num_docs:,} docs, {s.num_tokens:,} tokens, "
                f"{s.num_loss_tokens:,} loss tokens ({_share(s.num_loss_tokens, s.num_tokens)}); "
                f"{s.num_docs_over_length:,} docs longer than {sequence_length:,} lose "
                f"{s.num_tokens_truncated:,} tokens to truncation."
            )
        cache = _cache_dir_stats(cache_dir, token_dtype)
        extra_tokens = sum(s.num_tokens for s in stats.values())
        extra_loss = sum(s.num_loss_tokens for s in stats.values())
        logger.info(
            f"Chat cache {cache_dir}: {cache.num_tokens:,} tokens, {cache.num_loss_tokens:,} loss tokens. "
            f"Extras ({len(stats)} dirs): {extra_tokens:,} tokens, {extra_loss:,} loss tokens; "
            f"{_share(extra_tokens, cache.num_tokens + extra_tokens)} of all tokens and "
            f"{_share(extra_loss, cache.num_loss_tokens + extra_loss)} of all loss tokens, before packing."
        )
    return stats


def _numpy_dataset_config(
    numpy_dir: str,
    tokenizer: oc_data.TokenizerConfig,
    work_dir: str,
    sequence_length: int,
    document_boundary_kwargs: dict[str, Any],
    extra_numpy_dirs: Sequence[str] = (),
) -> oc_data.NumpyPackedFSLDatasetConfig:
    # One glob per dir, cache first. OLMo-core expands each glob to a sorted list and concatenates
    # the lists in order, pairing token and mask files by position; `_check_extra_numpy_dir`
    # guarantees each extra dir has the same part indices for both.
    dirs = [numpy_dir, *extra_numpy_dirs]
    return oc_data.NumpyPackedFSLDatasetConfig(
        tokenizer=tokenizer,
        work_dir=work_dir,
        paths=[os.path.join(d, numpy_dataset_conversion.TOKEN_IDS_NPY_GLOB) for d in dirs],
        expand_glob=True,
        label_mask_paths=[os.path.join(d, numpy_dataset_conversion.LABELS_MASK_NPY_GLOB) for d in dirs],
        generate_doc_lengths=True,
        long_doc_strategy=oc_data.LongDocStrategy.truncate,
        sequence_length=sequence_length,
        **document_boundary_kwargs,
    )


def _tokenize_to_numpy_dir(
    numpy_dir: str,
    args: "SFTArguments",
    tc: dataset_transformation.TokenizerConfig,
    transform_fn_args: list[dict],
    visualize: bool,
) -> None:
    logger.info(f"Tokenizing dataset into numpy format at {numpy_dir}")
    numpy_dataset_conversion.convert_hf_to_numpy_sft(
        output_dir=pathlib.Path(numpy_dir),
        dataset_mixer_list=args.dataset.mixer_list,
        dataset_mixer_list_splits=args.dataset.mixer_list_splits,
        tc=tc,
        dataset_transform_fn=args.dataset.transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_target_columns=list(dataset_transformation.TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE),
        max_seq_length=args.training.max_seq_length,
        dataset_cache_mode=args.dataset.cache_mode,
        dataset_local_cache_dir=args.dataset.local_cache_dir,
        dataset_skip_cache=args.dataset.skip_cache,
        dataset_config_hash=args.dataset.config_hash,
        shuffle_seed=args.tracking.seed,
        resume=True,
        visualize=visualize,
        row_aligned_parts=args.sft.row_aligned_parts,
    )


@dataclasses.dataclass
class SFTConfig:
    """Settings read only by this script.

    They stay out of the shared configs because DPO and GRPO inherit those and
    would advertise flags neither trainer reads.
    """

    dist_timeout_hours: float = 24
    """Timeout for distributed collectives, in hours."""
    save_async: bool = True
    """Whether olmo-core saves checkpoints asynchronously."""
    tracking_url: str | None = None
    """Optional URL (GitHub issue, ticket, experiment log) recorded in the run
    directory's provenance README so any copy of a checkpoint traces back to it."""
    row_aligned_parts: bool = False
    """Cut the numpy cache's 1 GiB part files only between rows, so no row spans two files.

    The default cuts every 1 GiB, mid-row, and OLMo-core's EOS scan never yields the piece of a row
    before a file's end. Row-aligned caches live in their own directory (suffix `-rowaligned`), so
    turning this on tokenizes once more. It does not change how training finds document boundaries;
    that is `document_boundaries_from_metadata`."""
    document_boundaries_from_metadata: bool = False
    """Take document boundaries from the per-row metadata files instead of scanning for EOS.

    The EOS scan splits a multi-turn row at each non-final assistant turn's EOS, and merges a row
    with no trailing EOS (a final tool call, or a row truncated at max_seq_length) into the next.
    With this set, packing, the attention mask and EMO document pools (if the model routes with
    EMO) follow the rows exactly. Requires
    `row_aligned_parts`, and an OLMo-core with `use_array_if_local` (allenai/OLMo-core#843)."""
    extra_numpy_dirs: list[str] = dataclasses.field(default_factory=list)
    """Local pretokenized numpy dirs to train on alongside the chat cache, in the converter's layout:
    `token_ids_part_XXXX.npy` (raw tokens in the dtype OLMo-core picks for the vocab, uint32 for Olmo)
    and `labels_mask_part_XXXX.npy` (one bool per token). Each part must end with EOS. Their parts
    follow the cache's and are packed the same way, one source file at a time, so an instance never
    mixes extra docs with chat rows; docs longer than max_seq_length are truncated. Checked at
    startup, which logs token and loss-token totals for the cache and the extras."""


@dataclasses.dataclass
class SFTArguments:
    tracking: olmo_core_utils.ExperimentConfig
    model: olmo_core_utils.ModelConfig
    training: olmo_core_utils.TrainingConfig
    dataset: olmo_core_utils.DatasetConfig
    logging: olmo_core_utils.LoggingConfig
    checkpoint: olmo_core_utils.CheckpointConfig
    sft: SFTConfig


def main(args: SFTArguments, tc: dataset_transformation.TokenizerConfig) -> None:
    # Fail before tokenizing, not after, if the installed OLMo-core cannot honour the flag.
    document_boundary_kwargs = _document_boundary_kwargs(
        args.sft.document_boundaries_from_metadata, args.sft.row_aligned_parts
    )
    use_hf_ckpt = olmo_core_utils.is_hf_checkpoint(args.model.model_name_or_path)

    olmo_core_utils.setup_tokenizer_and_cache(args.model, args.dataset, tc)
    transform_fn_args = [
        dataset_transformation.sft_tokenize_fn_args(args.training.max_seq_length, args.training.over_length_strategy),
        {},
    ]

    dcs = dataset_transformation.load_dataset_configs(
        dataset_mixer_list=args.dataset.mixer_list,
        dataset_mixer_list_splits=args.dataset.mixer_list_splits,
        dataset_transform_fn=args.dataset.transform_fn,
        transform_fn_args=transform_fn_args,
        target_columns=list(dataset_transformation.TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE),
    )
    cache_hash = dataset_transformation.compute_config_hash(dcs, tc)
    numpy_dir = _numpy_cache_dir(
        args.dataset.local_cache_dir,
        cache_hash,
        args.tracking.seed,
        args.training.max_seq_length,
        args.sft.row_aligned_parts,
    )

    if args.dataset.cache_dataset_only:
        pre_init_rank = int(os.environ.get("RANK", 0))
        if pre_init_rank == 0:
            if _numpy_dir_is_populated(numpy_dir, require_statistics=args.sft.row_aligned_parts):
                logger.info(f"Numpy SFT files already present at {numpy_dir}; nothing to do.")
            else:
                _tokenize_to_numpy_dir(numpy_dir, args, tc, transform_fn_args, visualize=True)
            logger.info("Dataset cached successfully. Exiting because --cache_dataset_only was set.")
        return

    if not _numpy_dir_is_populated(numpy_dir, require_statistics=args.sft.row_aligned_parts):
        cache_args = [
            f"--model_name_or_path {args.model.model_name_or_path}",
            f"--tokenizer_name_or_path {tc.tokenizer_name_or_path}",
            f"--max_seq_length {args.training.max_seq_length}",
            f"--mixer_list {' '.join(args.dataset.mixer_list)}",
            f"--mixer_list_splits {' '.join(args.dataset.mixer_list_splits)}",
            f"--seed {args.tracking.seed}",
        ]
        if tc.chat_template_name is not None:
            cache_args.append(f"--chat_template_name {tc.chat_template_name}")
        if tc.add_bos:
            cache_args.append("--add_bos")
        if args.dataset.transform_fn:
            cache_args.append(f"--transform_fn {' '.join(args.dataset.transform_fn)}")
        # Part of the cache hash.
        if args.training.over_length_strategy != dataset_transformation.DEFAULT_OVER_LENGTH_STRATEGY:
            cache_args.append(f"--over_length_strategy {args.training.over_length_strategy}")
        # Also part of the cache hash, and the values contain shell metacharacters: an
        # unquoted `<think>` would be a redirection, and a command that dropped the flag
        # would tokenize to a different hash than the one this job is looking for.
        if tc.reserved_slot_tokens:
            quoted = " ".join(shlex.quote(token) for token in tc.reserved_slot_tokens)
            cache_args.append(f"--reserved_slot_tokens {quoted}")
        # Part of the cache directory name.
        if args.sft.row_aligned_parts:
            cache_args.append("--row_aligned_parts")
        cache_args += [f"--local_cache_dir {args.dataset.local_cache_dir}", "--cache_dataset_only"]
        cache_cmd = " \\\n      ".join(cache_args)
        raise FileNotFoundError(
            "Pre-tokenized numpy SFT dataset not found.\n"
            f"  expected: {numpy_dir}\n"
            f"  hash:     {cache_hash}\n\n"
            "Launch a CPU-only tokenization job on a Weka-mounted cluster first, e.g.:\n\n"
            "  uv run python mason.py \\\n"
            "      --cluster ai2/jupiter --workspace ai2/open-instruct-dev \\\n"
            '      --priority urgent --image "$BEAKER_IMAGE" --budget ai2/oe-adapt \\\n'
            "      --gpus 0 --num_nodes 1 --no_auto_dataset_cache \\\n"
            "      -- uv run python open_instruct/olmo_core_finetune.py \\\n"
            f"      {cache_cmd}\n\n"
            "Re-launch training once the tokenization job has completed."
        )

    if args.sft.row_aligned_parts:
        _check_row_aligned_cache(numpy_dir)

    oc_tokenizer_config = olmo_core_utils.to_oc_tokenizer_config(tc)
    _check_extra_numpy_dirs(
        args.sft.extra_numpy_dirs,
        numpy_dir,
        oc_tokenizer_config,
        sequence_length=args.training.max_seq_length,
        require_metadata=args.sft.document_boundaries_from_metadata,
        log=int(os.environ.get("RANK", 0)) == 0,
    )

    global_rank, world_size, is_main_process = olmo_core_utils.setup_distributed_env(
        seed=args.tracking.seed, timeout=datetime.timedelta(hours=args.sft.dist_timeout_hours)
    )

    if is_main_process:
        os.makedirs(args.checkpoint.output_dir, exist_ok=True)
    if is_distributed():
        dist.barrier()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_config = olmo_core_utils.setup_model(
        args.model, tc, init_device="meta", emo_segments_from_doc_lens=args.sft.document_boundaries_from_metadata
    )

    cp_config = olmo_core_utils.build_cp_config(args.training)
    cp_degree = args.training.cp_degree or 1
    gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    dp_shard_degree = gpus_per_node // cp_degree

    dp_config = train_module_lib.TransformerDataParallelConfig(
        name=parallel.DataParallelType.hsdp if world_size > dp_shard_degree else parallel.DataParallelType.fsdp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
        shard_degree=dp_shard_degree,
    )

    ac_config = olmo_core_utils.build_ac_config(
        args.training.activation_memory_budget,
        args.training.compile_model,
        args.training.activation_checkpointing_mode,
        args.training.activation_checkpointing_modules,
    )

    rank_batch_size_seqs = args.training.per_device_train_batch_size * args.training.gradient_accumulation_steps
    rank_microbatch_size = args.training.per_device_train_batch_size * args.training.max_seq_length
    dp_world_size = world_size // cp_degree

    np_dataset_config = _numpy_dataset_config(
        numpy_dir,
        oc_tokenizer_config,
        work_dir=args.checkpoint.output_dir,
        sequence_length=args.training.max_seq_length,
        document_boundary_kwargs=document_boundary_kwargs,
        extra_numpy_dirs=args.sft.extra_numpy_dirs,
    )
    np_dataset = np_dataset_config.build()
    np_dataset.prepare()

    global_batch_size_seqs = rank_batch_size_seqs * dp_world_size

    num_training_steps = len(np_dataset) // global_batch_size_seqs * args.training.num_epochs
    effective_steps = (
        args.training.max_train_steps if args.training.max_train_steps is not None else num_training_steps
    )
    logger.info(f"Total training steps: {effective_steps} (epochs={args.training.num_epochs})")
    if effective_steps < 1:
        raise ValueError(
            f"Computed {effective_steps} training steps from {len(np_dataset)} packed instances "
            f"// global batch {global_batch_size_seqs} * {args.training.num_epochs} epochs. "
            "Use more data, or lower --gradient_accumulation_steps / --per_device_train_batch_size."
        )
    scheduler = olmo_core_utils.build_scheduler(
        args.training.lr_scheduler_type, args.training.warmup_ratio, effective_steps
    )

    train_module_config = train_module_lib.TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=args.training.max_seq_length,
        z_loss_multiplier=None,
        compile_model=args.training.compile_model,
        optim=optim.SkipStepAdamWConfig(
            lr=args.training.learning_rate, weight_decay=args.training.weight_decay, betas=(0.9, 0.95), compile=False
        ),
        dp_config=dp_config,
        cp_config=cp_config,
        ac_config=ac_config,
        scheduler=scheduler,
        max_grad_norm=args.training.max_grad_norm
        if args.training.max_grad_norm and args.training.max_grad_norm > 0
        else None,
    )

    train_module = train_module_config.build(model)

    data_loader_seed = (
        args.tracking.data_loader_seed if args.tracking.data_loader_seed is not None else args.tracking.seed
    )
    global_batch_size_tokens = global_batch_size_seqs * args.training.max_seq_length
    data_loader_config = oc_data.NumpyDataLoaderConfig(
        global_batch_size=global_batch_size_tokens,
        seed=data_loader_seed,
        work_dir=args.checkpoint.output_dir,
        num_workers=4,
        target_device_type=device.type,
    )
    data_loader = data_loader_config.build(
        np_dataset,
        collator=oc_data.DataCollator(
            pad_token_id=oc_tokenizer_config.pad_token_id, vocab_size=oc_tokenizer_config.padded_vocab_size()
        ),
        dp_process_group=train_module.dp_process_group,
    )
    data_loader.reshuffle(epoch=1)

    if use_hf_ckpt:
        olmo_core_utils.reload_hf_checkpoint_after_parallelization(
            train_module, args.model.model_name_or_path, args.checkpoint.output_dir
        )

    if args.training.max_train_steps is not None:
        max_duration = Duration.steps(args.training.max_train_steps)
    else:
        max_duration = Duration.epochs(args.training.num_epochs)

    run_name = args.tracking.run_name or f"sft-{os.path.basename(args.model.model_name_or_path)}"
    config_dict = dataclasses.asdict(args)

    if is_main_process:
        olmo_core_utils.write_provenance_readme(
            output_dir=args.checkpoint.output_dir,
            run_name=run_name,
            model_name_or_path=args.model.model_name_or_path,
            tracking_url=args.sft.tracking_url,
            wandb_project=args.logging.wandb_project if args.logging.with_tracking else None,
            wandb_entity=args.logging.wandb_entity,
        )

    trainer_callbacks: dict[str, Any] = olmo_core_utils.build_base_callbacks(
        config_dict=config_dict,
        run_name=run_name,
        checkpointing_steps=args.checkpoint.checkpointing_steps,
        ephemeral_save_interval=args.checkpoint.ephemeral_save_interval,
        with_tracking=args.logging.with_tracking,
        wandb_project=args.logging.wandb_project,
        wandb_entity=args.logging.wandb_entity or "ai2-llm",
        max_checkpoints=args.checkpoint.keep_last_n_checkpoints,
        save_async=args.sft.save_async,
    )
    trainer_callbacks["config_saver"] = callbacks.ConfigSaverCallback(_config=config_dict)
    trainer_callbacks["garbage_collector"] = callbacks.GarbageCollectorCallback()

    trainer = TrainerConfig(
        save_folder=args.checkpoint.output_dir,
        load_strategy=LoadStrategy.never,
        max_duration=max_duration,
        metrics_collect_interval=args.logging.logging_steps,
        callbacks=trainer_callbacks,
        save_overwrite=True,
        checkpointer=CheckpointerConfig(save_thread_count=1, load_thread_count=32, throttle_uploads=True),
    ).build(train_module, data_loader)

    # Loaded here rather than by fit(), which skips its own load once anything has
    # been loaded: putting the base weights in first would suppress it. Precedence
    # is fit()'s own -- an interrupted run in output_dir, then an explicit resume,
    # then the base weights.
    resumed = trainer.maybe_load_checkpoint(args.checkpoint.output_dir, load_trainer_state=True, load_optim_state=True)
    if args.checkpoint.resume_from_checkpoint is not None:
        if resumed:
            logger.warning(
                f"Ignoring --resume_from_checkpoint ({args.checkpoint.resume_from_checkpoint}) "
                f"since a checkpoint was found in {args.checkpoint.output_dir}"
            )
        else:
            logger.info(f"Resuming from {args.checkpoint.resume_from_checkpoint}...")
            # Raises when the path holds no checkpoint: a typo here must not
            # silently fall back to an expensive restart from the base weights.
            trainer.load_checkpoint(args.checkpoint.resume_from_checkpoint)
            resumed = True
    if not resumed and not use_hf_ckpt:
        logger.info(f"Loading olmo-core checkpoint from {args.model.model_name_or_path}...")
        trainer.load_checkpoint(args.model.model_name_or_path, load_trainer_state=False)

    # Last point at which the base weights are settled: the HF path loaded them before the
    # trainer was built, the olmo-core path just above, and the trainer is configured with
    # LoadStrategy.never so fit() loads nothing of its own. Skipped when resuming, since those
    # rows have been trained since they were seeded.
    if not resumed:
        olmo_core_utils.initialize_promoted_token_embeddings(train_module, tc.tokenizer)

    logger.info("Starting training...")
    trainer.fit()
    logger.info("Training complete.")

    teardown_training_environment()


if __name__ == "__main__":
    parser = utils.ArgumentParserPlus(
        (  # ty: ignore[invalid-argument-type]
            olmo_core_utils.ExperimentConfig,
            olmo_core_utils.ModelConfig,
            olmo_core_utils.TrainingConfig,
            olmo_core_utils.DatasetConfig,
            olmo_core_utils.LoggingConfig,
            olmo_core_utils.CheckpointConfig,
            SFTConfig,
            dataset_transformation.TokenizerConfig,
        )
    )
    parser.set_defaults(
        exp_name="sft",
        ephemeral_save_interval=_DEFAULT_EPHEMERAL_SAVE_INTERVAL,
        num_epochs=3,
        learning_rate=8e-5,
        warmup_ratio=0.03,
        mixer_list=["allenai/tulu-3-sft-olmo-2-mixture", "1.0"],
        transform_fn=["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"],
        target_columns=list(dataset_transformation.TOKENIZED_SFT_DATASET_KEYS),
    )
    tracking, model, training, dataset, logging_cfg, checkpoint, sft, tc = parser.parse()  # ty: ignore[invalid-assignment, not-iterable]
    args = SFTArguments(
        tracking=tracking,
        model=model,
        training=training,
        dataset=dataset,
        logging=logging_cfg,
        checkpoint=checkpoint,
        sft=sft,
    )
    main(args, tc)
