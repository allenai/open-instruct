"""Generic SFT mixture layer over OLMo-core's ``MixtureDataLoader``.

Multimodal (Molmo stage 2) is one *source type* here, not the shape of the system
(docs/design/multimodal_sft.md §4). Every source in a mixture satisfies the
``MixtureSource`` protocol — a map-style dataset whose ``__getitem__(i)`` returns the
vision-branch example schema (``input_ids``, next-token-shifted ``labels``, float
``loss_masks``, ``position_ids``, ``token_type_ids``, ``images``,
``pooled_patches_idx``) and is index-deterministic (epoch shuffling and resume replay
belong to ``MixtureDataLoader``). Adding a new kind of SFT data — another text corpus,
a new modality, tool trajectories — is one factory registration in
``SOURCE_REGISTRY``; the entry point and loader wiring do not change.

``MOLMO_DATA_DIR`` must be set in the process environment before Python imports
``olmo_core.data.multimodal`` (its ``paths.py`` freezes the value at import time), so
it is a launch-environment variable (``mason.py --env MOLMO_DATA_DIR=...``), not a
CLI argument.
"""

import copy
import dataclasses
import json
from typing import Any, Protocol, runtime_checkable

from olmo_core.data.multimodal import mixture_weights
from olmo_core.data.multimodal.mixtures import image_only_v9

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

# The mixture group that carries text-only SFT data in image_only_v9, and the name of
# the weka-dump source it uses by default. `MixtureConfig.nlp_source` swaps this
# source for the open-instruct text adapter.
NLP_GROUP = "nlp"
TULU4_SOURCE_NAME = "tulu4"
OPEN_INSTRUCT_TEXT_SOURCE_NAME = "open_instruct_text"
OPEN_INSTRUCT_SFT_TYPE = "open_instruct_sft"
MOLMO_TYPE = "molmo"


@runtime_checkable
class MixtureSource(Protocol):
    """What ``MixtureDataLoader`` requires of a mixture source."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> dict[str, Any]: ...


@dataclasses.dataclass
class SourceSpec:
    """A declarative mixture-source entry.

    ``type`` selects a factory from ``SOURCE_REGISTRY``; ``args`` are passed to it.
    A spec replaces the sources of ``group`` (specs for the same group accumulate),
    and ``rate``, when set, overrides the group's mixture rate.
    """

    name: str
    type: str
    group: str
    rate: float | None = None
    sampling_rate: float | None = None
    root_size_factor: float | None = None
    args: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: str) -> "SourceSpec":
        data = json.loads(raw)
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown SourceSpec fields {sorted(unknown)} in {raw!r}")
        return cls(**data)


class SourceFactory(Protocol):
    def __call__(self, spec: SourceSpec, tokenizer: Any, *, seed: int, max_sequence_length: int) -> MixtureSource: ...


def _build_molmo_source(spec: SourceSpec, tokenizer: Any, *, seed: int, max_sequence_length: int) -> MixtureSource:
    return image_only_v9.build_image_only_v9_dataset(
        spec.name, tokenizer, seed, max_sequence_length=max_sequence_length
    )


SOURCE_REGISTRY: dict[str, SourceFactory] = {MOLMO_TYPE: _build_molmo_source}
"""``SourceSpec.type`` -> factory. The ``open_instruct_sft`` adapter registers itself
here when it lands (docs/design/multimodal_sft.md §5); external code can register
additional types via ``register_source_type``."""


def register_source_type(name: str, factory: SourceFactory, *, overwrite: bool = False) -> None:
    if name in SOURCE_REGISTRY and not overwrite:
        raise ValueError(f"Source type {name!r} is already registered")
    SOURCE_REGISTRY[name] = factory


@dataclasses.dataclass
class MixtureConfig:
    """Mixture selection for the multimodal SFT entry point."""

    mixture: str = "debug"
    """A named preset: any key of ``image_only_v9.VALIDATION_MIXTURES`` (``debug``,
    ``demo``, ``pointing``, ``academic``, ``multi-image``, ..., ``image-only-v9``)."""

    sources: list[str] = dataclasses.field(default_factory=list)
    """JSON ``SourceSpec`` objects that replace or extend preset groups — the general
    mechanism, e.g. '{"group": "nlp", "rate": 0.166, "type": "open_instruct_sft",
    "name": "open_instruct_text", "args": {...}}'."""

    nlp_source: str = TULU4_SOURCE_NAME
    """Shorthand for the common override: ``tulu4`` keeps the vision branch's weka
    dump; ``open_instruct`` swaps the nlp group for the open-instruct text adapter."""

    nlp_rate: float | None = None
    """Override the nlp group's mixture rate (preset default: 0.166)."""

    mixer_list: list[str] = dataclasses.field(default_factory=list)
    """open-instruct dataset mixer for the ``open_instruct`` nlp source."""

    mixer_list_splits: list[str] = dataclasses.field(default_factory=lambda: ["train"])

    text_chat_template_name: str | None = None
    """Chat template for the ``open_instruct`` nlp source. None (or an unregistered
    name) falls through to the run tokenizer's own built-in template, which keeps the
    text half template-consistent with the image data (design doc §5.2)."""

    text_local_cache_dir: str = "local_dataset_cache"
    """dataset_transformation cache directory for the ``open_instruct`` nlp source."""

    text_base_vocab_size: int | None = None
    """LM base vocab size for the ``open_instruct`` nlp source's target-id guard
    (``SplitVocabEmbedding``'s extra block is inputs-only). The entry point fills this
    from the model config; None skips the guard."""

    pack_sequences: bool = True
    pack_max_crops: int = 125
    """Per-pack crop capacity for the 2D-knapsack packer (Stage2: 5 * (1 + 24))."""

    est_tokens_per_example: int = 1500
    prefetch_workers: int = 4
    max_crops: int = 8
    p_high_res: float | None = None

    def __post_init__(self):
        if self.mixture not in image_only_v9.VALIDATION_MIXTURES:
            known = ", ".join(sorted(image_only_v9.VALIDATION_MIXTURES))
            raise ValueError(f"Unknown mixture {self.mixture!r}; use one of: {known}")
        if self.nlp_source not in (TULU4_SOURCE_NAME, "open_instruct"):
            raise ValueError(f"nlp_source must be 'tulu4' or 'open_instruct', got {self.nlp_source!r}")

    def source_specs(self) -> list[SourceSpec]:
        """The ``sources`` field parsed, with the nlp shorthands desugared."""
        specs = [SourceSpec.from_json(raw) for raw in self.sources]
        if self.nlp_source == "open_instruct" and not any(s.group == NLP_GROUP for s in specs):
            specs.append(
                SourceSpec(
                    name=OPEN_INSTRUCT_TEXT_SOURCE_NAME,
                    type=OPEN_INSTRUCT_SFT_TYPE,
                    group=NLP_GROUP,
                    rate=self.nlp_rate,
                    args={
                        "mixer_list": self.mixer_list,
                        "mixer_list_splits": self.mixer_list_splits,
                        "chat_template_name": self.text_chat_template_name,
                        "local_cache_dir": self.text_local_cache_dir,
                        "base_vocab_size": self.text_base_vocab_size,
                    },
                )
            )
        elif self.nlp_rate is not None and not any(s.group == NLP_GROUP for s in specs):
            specs.append(SourceSpec(name=TULU4_SOURCE_NAME, type=MOLMO_TYPE, group=NLP_GROUP, rate=self.nlp_rate))
        return specs


def resolve_groups(config: MixtureConfig) -> tuple[list[Any], dict[str, SourceSpec]]:
    """Resolve the preset + overrides into pruned SubMixture groups.

    Returns the groups (deep-copied ``mixture_weights.SubMixture``s) and a map from
    source name to the ``SourceSpec`` that should build it (sources absent from the
    map are built by the ``molmo`` factory). Pruning to the named mixture's sources
    happens here, BEFORE any dataset is built, so lazy weka datasets outside the
    subset are never touched.
    """
    groups = copy.deepcopy(image_only_v9.IMAGE_ONLY_V9_SUBMIXTURES)
    specs_by_group: dict[str, list[SourceSpec]] = {}
    for spec in config.source_specs():
        specs_by_group.setdefault(spec.group, []).append(spec)

    custom_specs: dict[str, SourceSpec] = {}
    replaced_names: dict[str, set[str]] = {}
    known_groups = {group.name for group in groups}
    unknown = set(specs_by_group) - known_groups
    if unknown:
        raise ValueError(f"SourceSpec groups {sorted(unknown)} not in mixture groups {sorted(known_groups)}")

    for group in groups:
        specs = specs_by_group.get(group.name)
        if not specs:
            continue
        replaced_names[group.name] = {source.name for source in group.datasets}
        rates = [spec.rate for spec in specs if spec.rate is not None]
        if rates:
            group.rate = rates[-1]
        group.datasets = [
            mixture_weights.DatasetSource(
                spec.name, sampling_rate=spec.sampling_rate, root_size_factor=spec.root_size_factor
            )
            for spec in specs
        ]
        for spec in specs:
            if spec.type not in SOURCE_REGISTRY:
                known = ", ".join(sorted(SOURCE_REGISTRY))
                raise ValueError(f"Unknown source type {spec.type!r} for {spec.name!r}; registered: {known}")
            custom_specs[spec.name] = spec

    names_filter = image_only_v9.VALIDATION_MIXTURES[config.mixture]
    if names_filter is not None:
        # A named mixture lists preset source names; a replaced group's new sources
        # stand in for any of that group's original names in the filter.
        allowed = set(names_filter)
        for group in groups:
            if group.name in replaced_names and allowed & replaced_names[group.name]:
                continue  # group was overridden and the filter selected (some of) its old sources
            group.datasets = [source for source in group.datasets if source.name in allowed]

    groups = [group for group in groups if group.datasets and group.rate > 0]
    if not groups:
        raise ValueError(f"Mixture {config.mixture!r} with the given overrides selects no sources")
    return groups, custom_specs


def build_mixture(
    tokenizer: Any, config: MixtureConfig, *, max_sequence_length: int, seed: int
) -> tuple[list[MixtureSource], list[float], list[str]]:
    """Build (datasets, weights, names) for ``MixtureDataLoader``.

    Weight math is ``mixture_weights.compute_flat_mixture_weights`` (mm_olmo SubMixture
    rate semantics), computed only over the sources that survive pruning.
    """
    groups, custom_specs = resolve_groups(config)

    datasets: dict[str, MixtureSource] = {}
    lengths: dict[str, int] = {}
    for group in groups:
        for source in group.datasets:
            spec = custom_specs.get(source.name) or SourceSpec(name=source.name, type=MOLMO_TYPE, group=group.name)
            factory = SOURCE_REGISTRY[spec.type]
            dataset = factory(spec, tokenizer, seed=seed, max_sequence_length=max_sequence_length)
            datasets[source.name] = dataset
            lengths[source.name] = len(dataset)

    flat = mixture_weights.compute_flat_mixture_weights(groups, lengths)
    names = [name for name, _ in flat]
    weights = [weight for _, weight in flat]
    logger.info("Mixture %s sources / weights: %s", config.mixture, list(zip(names, [round(w, 4) for w in weights])))
    return [datasets[name] for name in names], weights, names
