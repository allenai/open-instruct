"""Turning your items into rows open-instruct will train on.

An RLVR row is four columns:

    messages     the chat the policy is prompted with
    ground_truth a string; anything structured travels as JSON
    dataset      which verifier scores it - NOT the corpus name
    env_config   which environment runs, and its per-row arguments

The one that trips everyone up is ``dataset``. It looks like provenance and it
is not: ``ground_truth_utils.apply_verifiable_reward`` uses it to look up the
verifier. Provenance lives in ``dataset_source``. If you are scoring with a
group scorer and want no per-sample verifier at all, use ``"passthrough"``,
which is upstream's built-in no-op returning 0.0.

``ground_truth`` is a string column, so ``build_rows`` JSON-encodes the whole
item into it. ``Sample.item`` decodes it on the other side. This is how a
scorer gets at the question, the reference, the rubric, the grade band, or
anything else it needs - without adding columns that the tokenizer would have
to be taught about.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from typing import Any

PASSTHROUGH = "passthrough"


def build_rows(
    items: Sequence[dict],
    system: str | Callable[[dict], str],
    user: Callable[[dict], str],
    verifier: str = PASSTHROUGH,
    env_name: str | None = None,
    env_kwargs: Callable[[dict], dict] | None = None,
    max_steps: int | None = None,
    opening: Callable[[dict], str] | None = None,
    opening_role: str = "assistant",
    keep_keys: Iterable[str] | None = None,
) -> list[dict]:
    """Build RLVR rows from arbitrary items.

    ``opening`` is for a multi-turn task whose environment should speak first.
    open-instruct discards the observation returned by ``env.reset``, so the
    policy always generates first; the partner's opening line therefore has to
    be part of the prompt. Generating it here, offline, is better than at
    runtime anyway - it is then identical across the G completions of a group,
    which makes the group a comparison between policy samples rather than
    between environments, and it costs no GPU during training.

    ``keep_keys`` narrows what goes into ``ground_truth``. Everything is kept by
    default, which is usually right; narrow it if your items carry something
    large that no scorer reads.
    """
    rows = []
    for item in items:
        payload = dict(item) if keep_keys is None else {k: item[k] for k in keep_keys if k in item}
        first = opening(item) if opening else None
        if first:
            payload.setdefault("opening", first)

        messages = []
        rendered_system = system(item) if callable(system) else system
        if rendered_system:
            messages.append({"role": "system", "content": rendered_system})
        messages.append({"role": "user", "content": user(item)})
        if first:
            # the environment's first turn, shown to the policy as context. It is
            # part of the PROMPT, so it is never trained on.
            messages[-1]["content"] += f"\n\n{first}"

        row: dict[str, Any] = {
            "messages": messages,
            "ground_truth": json.dumps(payload, ensure_ascii=False),
            "dataset": verifier,
        }
        if env_name:
            entry = {"env_name": env_name, "item": json.dumps(payload, ensure_ascii=False)}
            entry.update(env_kwargs(item) if env_kwargs else {})
            row["env_config"] = {"env_configs": [entry]}
            if max_steps is not None:
                row["env_config"]["max_steps"] = max_steps
        rows.append(row)
    return rows


def write_jsonl(rows: Sequence[dict], path: str) -> str:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_hf_dataset(rows: Sequence[dict]):
    from datasets import Dataset  # noqa: PLC0415

    return Dataset.from_list(list(rows))


def push(rows: Sequence[dict], repo_id: str, split: str = "train", private: bool = True):
    """Upload to the Hub, which is how ``--dataset_mixer_list`` wants to find it.

    A local jsonl also works via ``--dataset_mixer_list path/to/file.jsonl 1.0``,
    but the Hub path is what the launch scripts and the cluster expect.
    """
    dataset = to_hf_dataset(rows)
    dataset.push_to_hub(repo_id, split=split, private=private)
    return repo_id


def split_by(items: Sequence[dict], key: str, held_out: Iterable[Any]) -> tuple[list[dict], list[dict]]:
    """Split on a categorical field rather than at random.

    Prefer this whenever the field exists. A random slice of one corpus shares
    its authors, its year and its house style with the training set, so
    "held out" measures memorisation of items and nothing else. Holding out a
    whole source measures whether the thing transfers.
    """
    held = set(held_out)
    train = [i for i in items if i.get(key) not in held]
    test = [i for i in items if i.get(key) in held]
    return train, test
