# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "beaker-py>=1.32.2,<2.0",
#     "datasets>=4.0.0",
#     "numpy<2",
#     "ray[default]>=2.44.1",
#     "rich>=13.7.0",
#     "tqdm",
#     "transformers>=4.52.4",
#     "torch>=2.7.0,<2.8",
# ]
# ///

"""
This script converts SFT datasets to OLMoCore format. OLMoCore has a more efficient
implementation of the OLMo models (espeically for MoE), and so it can be preferable
to use it for training on next-token prediction tasks (e.g. SFT).

OLMoCore accepts data in numpy mmap format. One file is for the input tokens and one for the labels mask.

## Usage:
    ```bash
    python scripts/data/convert_sft_data_for_olmocore.py \
        --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
        --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0 \
        --output_dir ./data/tulu-3-sft-olmo-2-mixture-0225-olmocore \
        --chat_template_name olmo
    ```

## Ai2 Internal Usage (requires `gantry>=3`):
    ```bash
    gantry run \
        --workspace ai2/jacobm \
        --budget ai2/oe-base \
        --priority normal \
        --cluster ai2/neptune --gpus 1 \
        --weka=oe-training-default:/weka/oe-training-default \
        --task-name convert-sft-data-for-olmocore --yes \
        --env-secret HF_TOKEN=HF_TOKEN \
        --install "echo 'do nothing'" \
        -- uv run --script scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture 1.0 \
            --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
            --output_dir /weka/oe-training-default/ai2-llm/tylerr/data/sft/tulu-3-sft-olmo-2-mixture-0225-olmocore-test1 \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 16384
    ```

NOTE: allenai/OLMo-2-1124-7B tokenizer is the same as allenai/dolma2-tokenizer, but allenai/OLMo-2-1124-7B
has additional metadata required for this script.

Recommendations:
  * Set max_seq_length, and use the same length you use during SFT
"""

import os
from dataclasses import dataclass, field
from typing import Literal

from open_instruct import dataset_transformation, numpy_dataset_conversion, utils


@dataclass
class ConvertSFTDataArguments:
    """Arguments for converting SFT data to OLMoCore format."""

    output_dir: str = field()
    """Output directory"""

    dataset_name: str | None = field(default=None)
    """The name of the dataset to use (via the datasets library)."""

    dataset_mixer: dict | None = field(default=None)
    """A dictionary of datasets (local or HF) to sample from."""

    dataset_mixer_list: list[str] = field(default_factory=lambda: ["allenai/tulu-3-sft-olmo-2-mixture-0225", "1.0"])
    """A list of datasets (local or HF) to sample from."""

    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""

    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )
    """The list of transform functions to apply to the dataset."""

    dataset_target_columns: list[str] = field(
        default_factory=lambda: list(dataset_transformation.TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE)
    )
    """The columns to use for the dataset."""

    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""

    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""

    dataset_config_hash: str | None = None
    """The hash of the dataset configuration."""

    dataset_skip_cache: bool = False
    """Whether to skip the cache."""

    max_seq_length: int | None = field(default=None)
    """Maximum sequence length. If not provided, no truncation will be performed."""

    num_examples: int = field(default=0)
    """Number of examples to process for debugging. 0 means process all examples."""

    visualize: bool = field(default=False)
    """Visualize first token sequence"""

    tokenizer_config_only: bool = field(default=False)
    """Only write the tokenizer config to the output directory"""

    resume: bool = field(default=False)
    """Resume from checkpoint if available"""

    checkpoint_interval: int = field(default=100_000)
    """Checkpoint save interval (number of samples)"""

    shuffle_seed: int = field(default=42)
    """Shuffle seed for reproducible dataset ordering"""


def main(args: ConvertSFTDataArguments, tc: dataset_transformation.TokenizerConfig) -> None:
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if utils.is_beaker_job():
        beaker_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
        if os.path.exists(beaker_cache_dir):
            args.dataset_local_cache_dir = beaker_cache_dir

    transform_fn_args = []
    for fn_name in args.dataset_transform_fn:
        if fn_name == "sft_tulu_tokenize_and_truncate_v1":
            transform_fn_args.append({"max_seq_length": args.max_seq_length})
        else:
            transform_fn_args.append({})

    numpy_dataset_conversion.convert_hf_to_numpy_sft(
        output_dir=args.output_dir,
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=args.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=args.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_target_columns=args.dataset_target_columns,
        max_seq_length=args.max_seq_length,
        dataset_cache_mode=args.dataset_cache_mode,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
        dataset_config_hash=args.dataset_config_hash,
        shuffle_seed=args.shuffle_seed,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        visualize=args.visualize,
        tokenizer_config_only=args.tokenizer_config_only,
        num_examples=args.num_examples,
    )


if __name__ == "__main__":
    parser = utils.ArgumentParserPlus((ConvertSFTDataArguments, dataset_transformation.TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
