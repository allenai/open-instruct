"""open-instruct text SFT data as a ``MixtureSource`` (docs/design/multimodal_sft.md §5).

The first source adapter for the generic mixture layer: the ``nlp`` group of the
multimodal stage-2 mixture is produced by open-instruct's own dataset tooling —
``dataset_transformation``'s mixer, chat templates, tools-aware assistant-span label
masking, and content-addressed caching — rather than the vision branch's weka dump.

The adapter consumes open-instruct's pre-tokenized rows as-is (§5.2): re-encoding
through the vision branch's layout modules would silently reintroduce mm_olmo's text
dialect and drop tools support. It maps them to the vision-branch example schema
(§5.3): open-instruct rows are unshifted (``labels[i]`` is the target *at* position
``i``, −100 = masked); the vision schema is already next-token-shifted with float
``loss_masks``, and a segment's end predicts EOS. Text examples are zero-crop —
``images``/``pooled_patches_idx`` keep the real trailing dims on empty arrays, which
the collator and 2D-knapsack packer read off zero-length tensors.

Importing this module registers the ``open_instruct_sft`` source type in
``sft_mixture.SOURCE_REGISTRY``.
"""

import dataclasses
import math
from typing import Any

import numpy as np
import torch.distributed as dist
from datasets import Dataset
from olmo_core.distributed import utils as dist_utils
from olmo_core.nn.vision.molmo2_tokens import N_PATCHES_SQ, PATCH_DIM, POOL_H, POOL_W

from open_instruct import dataset_transformation, logger_utils, sft_mixture

logger = logger_utils.setup_logger(__name__)

LOSS_TOKEN_WEIGHTINGS = ("root_tokens", "none")


@dataclasses.dataclass
class OpenInstructTextDatasetConfig:
    """How to build the text source from an open-instruct dataset mix."""

    mixer_list: list[str]
    """Alternating [dataset_name, amount, ...] — the open-instruct mixer format."""

    max_seq_length: int

    mixer_list_splits: list[str] = dataclasses.field(default_factory=lambda: ["train"])

    transform_fn: list[str] = dataclasses.field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )

    chat_template_name: str | None = None
    """None (or an unregistered name) falls through to the tokenizer's own built-in
    template — the ``olmo123`` convention from the validated Olmo 3 text pipeline."""

    loss_token_weighting: str = "root_tokens"
    """``root_tokens`` scales each example's mask by ``2/sqrt(n_loss_tokens)`` (parity
    with the vision branch's tulu source); ``none`` keeps binary masks."""

    message_weight: float | None = None
    """Optional flat per-example scalar on top of the weighting."""

    base_vocab_size: int | None = None
    """Guard: every input id must be below the LM base vocab (the
    ``SplitVocabEmbedding`` extra block is inputs-only, so ids at or above this are
    invalid LM-head targets). None skips the guard."""

    local_cache_dir: str = "local_dataset_cache"
    dataset_config_seed: int = 42
    skip_cache: bool = False

    def __post_init__(self):
        if self.loss_token_weighting not in LOSS_TOKEN_WEIGHTINGS:
            raise ValueError(
                f"loss_token_weighting must be one of {LOSS_TOKEN_WEIGHTINGS}, got {self.loss_token_weighting!r}"
            )
        if not self.mixer_list:
            raise ValueError("mixer_list is required for the open_instruct_sft source (e.g. --mixer_list <name> 1.0)")

    def build(self, tokenizer: Any) -> "OpenInstructTextDataset":
        """Tokenize (or load from cache) the mix with the run's tokenizer.

        Rank 0 transforms and writes the cache first; other ranks wait on the barrier
        and then read the cache (memory-mapped — ``dataset_keep_in_memory=False`` —
        so N local ranks share the page cache instead of holding N copies in RAM).
        """
        if tokenizer.eos_token_id is None:
            raise ValueError(f"Tokenizer {tokenizer.name_or_path!r} has no eos token id")
        tc = dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=tokenizer.name_or_path,
            trust_remote_code=True,
            chat_template_name=self.chat_template_name,
            add_bos=False,
        )

        def _load() -> Dataset:
            dataset, _ = dataset_transformation.get_cached_dataset_tulu_with_statistics(
                dataset_mixer_list=self.mixer_list,
                dataset_mixer_list_splits=self.mixer_list_splits,
                tc=tc,
                dataset_transform_fn=list(self.transform_fn),
                transform_fn_args=[{"max_seq_length": self.max_seq_length}, {}],
                target_columns=list(dataset_transformation.TOKENIZED_SFT_DATASET_KEYS),
                dataset_local_cache_dir=self.local_cache_dir,
                dataset_config_seed=self.dataset_config_seed,
                dataset_skip_cache=self.skip_cache,
                dataset_keep_in_memory=False,
            )
            return dataset

        if dist_utils.is_distributed():
            if dist_utils.get_rank() == 0:
                dataset = _load()
            dist.barrier()
            if dist_utils.get_rank() != 0:
                dataset = _load()
        else:
            dataset = _load()

        dataset = dataset.with_format("numpy", columns=["input_ids", "labels"])
        logger.info("open_instruct_sft source: %d examples from %s", len(dataset), self.mixer_list)
        return OpenInstructTextDataset(
            dataset,
            eos_token_id=tokenizer.eos_token_id,
            base_vocab_size=self.base_vocab_size,
            loss_token_weighting=self.loss_token_weighting,
            message_weight=self.message_weight,
        )


class OpenInstructTextDataset:
    """Map-style, index-deterministic ``MixtureSource`` over pre-tokenized text rows."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        eos_token_id: int,
        base_vocab_size: int | None = None,
        loss_token_weighting: str = "root_tokens",
        message_weight: float | None = None,
    ):
        self._dataset = dataset
        self.eos_token_id = eos_token_id
        self.base_vocab_size = base_vocab_size
        self.loss_token_weighting = loss_token_weighting
        self.message_weight = message_weight

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        row = self._dataset[int(index)]
        input_ids = np.asarray(row["input_ids"], dtype=np.int64)
        unshifted_labels = np.asarray(row["labels"], dtype=np.int64)
        n = input_ids.shape[0]

        if self.base_vocab_size is not None and int(input_ids.max(initial=0)) >= self.base_vocab_size:
            raise ValueError(
                f"Example {index} contains token id {int(input_ids.max())} >= base vocab "
                f"{self.base_vocab_size}; ids in the inputs-only extra-vocab block are not valid LM targets"
            )

        # open-instruct labels are unshifted; the vision schema wants labels[i] = input_ids[i+1]
        # with the loss mask shifted to match, and a segment's end predicting EOS.
        labels = np.empty(n, dtype=np.int64)
        labels[:-1] = input_ids[1:]
        labels[-1] = self.eos_token_id  # don't-care: its mask is always 0
        loss_masks = np.zeros(n, dtype=np.float32)
        loss_masks[:-1] = unshifted_labels[1:] != dataset_transformation.MASKED_TOKEN_VALUE

        n_loss_tokens = float(loss_masks.sum())
        if self.loss_token_weighting == "root_tokens" and n_loss_tokens > 0:
            loss_masks *= 2.0 / math.sqrt(n_loss_tokens)
        if self.message_weight is not None:
            loss_masks *= self.message_weight

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_masks": loss_masks,
            "position_ids": np.arange(n, dtype=np.int64),
            "token_type_ids": np.zeros(n, dtype=np.int64),
            # Zero-crop: the trailing dims must be real even on empty arrays — the
            # collator and packer read shape[1]/shape[2] off them.
            "images": np.zeros((0, N_PATCHES_SQ, PATCH_DIM), dtype=np.float32),
            "pooled_patches_idx": np.full((0, POOL_H * POOL_W), -1, dtype=np.int64),
        }


def _build_open_instruct_source(
    spec: sft_mixture.SourceSpec, tokenizer: Any, *, seed: int, max_sequence_length: int
) -> OpenInstructTextDataset:
    config = OpenInstructTextDatasetConfig(max_seq_length=max_sequence_length, dataset_config_seed=seed, **spec.args)
    return config.build(tokenizer)


sft_mixture.register_source_type(sft_mixture.OPEN_INSTRUCT_SFT_TYPE, _build_open_instruct_source)
