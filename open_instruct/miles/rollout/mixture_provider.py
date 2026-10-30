"""Fixed source allocation with an independent shuffled cursor per source."""

import bisect
import copy
import hashlib
import itertools
import json
import random
from pathlib import Path

from miles.utils.data import Dataset

from open_instruct.miles.datasets import mixture
from open_instruct.miles.rollout import prompt_provider


class SourceDatasets:
    """Expose total pool size for MILES run accounting without concatenating rows."""

    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(map(len, self.datasets))


class MixturePromptProvider:
    requires_feedback = False

    def __init__(self, sources, *, seed, shuffle, identity):
        mixture.validate_sources(sources)
        self.sources = sources
        self.identity = {"manifest_sha256": identity, "seed": seed, "shuffle": shuffle}
        self.rng = random.Random(seed)
        self.cumulative = list(itertools.accumulate(source["fraction"] for source in sources))
        self.cumulative[-1] = 1.0  # Only roundoff within the validated 1e-9 tolerance.
        self.providers = [
            prompt_provider.ShuffledPromptProvider(lambda source=source: source["dataset"], shuffle=shuffle)
            for source in sources
        ]

    def next_samples(self, n):
        if type(n) is not int or n < 0:
            raise ValueError("n must be a nonnegative integer")
        samples = []
        for _ in range(n):
            index = 0 if len(self.sources) == 1 else bisect.bisect_right(self.cumulative, self.rng.random())
            provider = self.providers[index]
            sample = copy.copy(provider.next_samples(1)[0])
            sample.metadata = (sample.metadata or {}) | {
                "mixture_source": self.sources[index]["name"],
                "mixture_source_epoch": provider.epoch_id,
            }
            samples.append(sample)
        return samples

    def feedback(self, records):
        """Fixed allocation ignores outcomes; the shared interface remains available."""

    def state_dict(self):
        return {
            "kind": "mixture",
            **self.identity,
            "sources": [{"name": s["name"], "fraction": s["fraction"]} for s in self.sources],
            "rng": self.rng.getstate(),
            "cursors": [provider.state_dict() for provider in self.providers],
        }

    def load_state_dict(self, state):
        previous = self.state_dict()
        for field in ("kind", *self.identity, "sources"):
            if state.get(field) != previous[field]:
                raise RuntimeError(
                    f"Mixture checkpoint mismatch for {field}: checkpoint={state.get(field)!r}, current={previous[field]!r}"
                )
        try:
            for provider, cursor in zip(self.providers, state["cursors"], strict=True):
                provider.load_state_dict(cursor)
            self.rng.setstate(state["rng"])
        except Exception:
            for provider, cursor in zip(self.providers, previous["cursors"], strict=True):
                provider.load_state_dict(cursor)
            self.rng.setstate(previous["rng"])
            raise


def load(args, first_dataset):
    """Read prepared source files independently and pin checkpoint identity to the manifest."""
    if args.apply_chat_template or args.multimodal_keys is not None:
        raise ValueError(
            "Mixture sources require pre-rendered text: apply_chat_template must be false and multimodal_keys must be None"
        )
    path = Path(args.olmo_core.prompt_mixture)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported mixture manifest schema")
    for key, value in (("seed", args.rollout_seed), ("shuffle", args.rollout_shuffle)):
        if manifest.get(key) != value:
            raise ValueError(f"Mixture manifest {key} differs from runtime: {manifest.get(key)!r} != {value!r}")
    mixture.validate_sources(manifest["sources"])
    sources = []
    for index, source in enumerate(manifest["sources"]):
        filename = source["path"]
        if Path(filename).name != filename:
            raise ValueError("Mixture source paths must be filenames within the prepared directory")
        source_path = path.parent / filename
        with source_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != source["sha256"]:
            raise ValueError(f"Mixture source {source['name']}: prepared artifact hash changed")
        if index == 0:
            if source_path.resolve() != Path(args.prompt_data).resolve():
                raise ValueError("prompt_data must name the first prepared mixture source")
            dataset = first_dataset
        else:
            dataset = Dataset(
                str(source_path),
                tokenizer=None,
                processor=None,
                max_length=None,
                prompt_key="input",
                label_key="label",
                metadata_key="metadata",
                seed=args.rollout_seed,
            )
            if args.rollout_shuffle:
                dataset.shuffle(0)
        if not dataset or len(dataset) != source["rows"]:
            raise ValueError(f"Mixture source {source['name']}: row count changed (possibly prompt-length filtering)")
        for sample in dataset.origin_samples:
            length = (sample.metadata or {}).get("run_prompt_tokens")
            if (
                type(length) is not int
                or length < 1
                or (args.rollout_max_prompt_len and length > args.rollout_max_prompt_len)
            ):
                raise ValueError(f"Mixture source {source['name']}: missing or excessive prepared prompt length")
        sources.append(source | {"dataset": dataset})
    provider = MixturePromptProvider(
        sources, seed=args.rollout_seed, shuffle=args.rollout_shuffle, identity=hashlib.sha256(raw).hexdigest()
    )
    return provider, SourceDatasets([source["dataset"] for source in sources])
