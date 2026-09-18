"""Immutable HF model staging and existing Open Instruct prompt preparation."""

import fcntl
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from open_instruct.miles import opd_config, run_data, workflow
from open_instruct.miles.errors import InputError

_VL_TEXT_PREFIX = "model.language_model."
_INDEX_NAME = "model.safetensors.index.json"


def _weight_names(checkpoint):
    index = checkpoint / _INDEX_NAME
    if index.is_file():
        return sorted(json.loads(index.read_text())["weight_map"])
    names = []
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            names.extend(handle.keys())
    return sorted(names)


def needs_text_key_rewrite(checkpoint):
    """True for text-only Qwen3.5 checkpoints saved with vision-language weight names.

    transformers >= 5 saves a ``Qwen3_5ForCausalLM`` loaded from a Qwen3.5 VL
    repository with its original ``model.language_model.*`` tensor names. SGLang's
    text-only loader strips ``model.`` and then skips every ``language_model.*``
    tensor, so the teacher serves random weights and NaN probabilities. Renaming
    the tensors to ``model.*`` gives SGLang the layout the architecture promises.
    """
    config = json.loads((checkpoint / "config.json").read_text())
    architectures = config.get("architectures") or []
    if not architectures or not all(name.endswith("ForCausalLM") for name in architectures):
        return False
    return any(name.startswith(_VL_TEXT_PREFIX) for name in _weight_names(checkpoint))


def _text_key(name):
    return "model." + name[len(_VL_TEXT_PREFIX) :] if name.startswith(_VL_TEXT_PREFIX) else name


def rewrite_text_keys(original, target):
    """Copy safetensors shards from original to target with text-only tensor names."""
    dropped = ("model.visual.", "mtp.")
    weight_map = {}
    for shard in sorted(original.glob("*.safetensors")):
        tensors = {}
        with safe_open(shard, framework="pt") as handle:
            metadata = handle.metadata() or {"format": "pt"}
            for name in list(handle.keys()):
                if name.startswith(dropped):
                    continue
                tensors[_text_key(name)] = handle.get_tensor(name)
        save_file(tensors, target / shard.name, metadata=metadata)
        weight_map.update({name: shard.name for name in tensors})
    index = original / _INDEX_NAME
    if index.is_file() or len(weight_map) > 1:
        previous = json.loads(index.read_text()) if index.is_file() else {}
        workflow.write_json(target / _INDEX_NAME, {**previous, "weight_map": dict(sorted(weight_map.items()))})
    return weight_map


def eos_token_ids(generation_config, eos_token_id):
    """``eos_token_id`` list for a generation config after adding the teacher's eos: the new id first,
    then the learner's original id(s), so the engine still stops on the learner's own eos."""
    previous = generation_config.get("eos_token_id", [])
    previous = [previous] if isinstance(previous, int) else list(previous)
    return [eos_token_id] + [i for i in previous if i != eos_token_id]


def align_eos(target, eos_token):
    """Make the prepared learner at ``target`` stop on ``eos_token`` (the teacher's eos) as well."""
    tokenizer = AutoTokenizer.from_pretrained(target)
    if eos_token not in tokenizer.get_vocab():
        raise InputError(f"model.align_eos_with_teacher: {eos_token!r} is not in the learner vocabulary")
    tokenizer.eos_token = eos_token
    tokenizer.save_pretrained(target)
    path = Path(target) / "generation_config.json"
    generation_config = json.loads(path.read_text()) if path.is_file() else {}
    generation_config["eos_token_id"] = eos_token_ids(generation_config, tokenizer.convert_tokens_to_ids(eos_token))
    workflow.write_json(path, generation_config)


def prepare(spec):
    assets = Path(spec.output["assets"])
    assets.mkdir(parents=True, exist_ok=True)
    with (assets / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identities = {}
        paths = {}
        # The teacher goes first so the learner can adopt its eos token.
        for role in ("teacher", "model"):
            item = spec.document[role]
            original = None
            align_eos_token = None
            if role == "model" and spec.document["model"]["align_eos_with_teacher"]:
                align_eos_token = AutoTokenizer.from_pretrained(paths["teacher"]).eos_token
            if opd_config.is_local(item["source"]):
                original = Path(item["source"])
                if not (original / "config.json").is_file():
                    raise InputError(f"{role}.source {original} is not a Hugging Face checkpoint directory")
                rewrite = needs_text_key_rewrite(original)
                stamp = workflow.fingerprint(
                    {
                        "path": str(original),
                        "config": (original / "config.json").read_text(),
                        **({"weight_layout": "text"} if rewrite else {}),
                    }
                )[:12]
                identity = {"source": str(original), "revision": "", "thinking": False, "local_fingerprint": stamp}
                target = assets / f"{original.name}-local-{stamp}"
            else:
                rewrite = False
                identity = {"source": item["source"], "revision": item["revision"], "thinking": False}
                target = assets / f"{item['source'].split('/')[-1]}-{item['revision'][:12]}"
            if align_eos_token is not None:
                identity["eos_token"] = align_eos_token
                target = target.with_name(target.name + "-eos")
            marker = target / "opd-source.json"
            if marker.exists():
                if json.loads(marker.read_text()) != identity:
                    raise InputError("Prepared model identity differs")
            else:
                if original is None:
                    original = Path(
                        snapshot_download(
                            item["source"],
                            revision=item["revision"],
                            local_dir=assets
                            / "downloads"
                            / f"{item['source'].split('/')[-1]}-{item['revision'][:12]}",
                            allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt", "*.model"],
                        )
                    )
                target.mkdir(exist_ok=True)
                for source in original.iterdir():
                    if not source.is_file():
                        continue
                    dest = target / source.name
                    if dest.exists() or (rewrite and source.name == _INDEX_NAME):
                        continue
                    if source.suffix == ".safetensors":
                        if not rewrite:
                            dest.symlink_to(source.resolve())
                    else:
                        shutil.copy2(source, dest)
                if rewrite:
                    rewrite_text_keys(original, target)
                tokenizer = AutoTokenizer.from_pretrained(target)
                template = tokenizer.get_chat_template()
                tokenizer.chat_template = "{% set enable_thinking = false %}\n" + template
                tokenizer.save_pretrained(target)
                if align_eos_token is not None:
                    align_eos(target, align_eos_token)
                workflow.write_json(marker, identity)
            tokenizer = AutoTokenizer.from_pretrained(target)
            identities[role] = {
                **identity,
                "vocab_sha256": hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
                "special_tokens": tokenizer.special_tokens_map,
                "template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
            }
            paths[role] = str(target)
        if identities["model"]["vocab_sha256"] != identities["teacher"]["vocab_sha256"]:
            raise InputError("Teacher and learner token IDs differ")
        if identities["model"]["special_tokens"] != identities["teacher"]["special_tokens"]:
            raise InputError("Teacher and learner special tokens differ")
        inf = spec.document["inference"]
        data_key = workflow.fingerprint(
            {
                "data": spec.document["data"],
                "model": identities["model"],
                "prompt_limit": inf["max_context_length"] - inf["max_response_length"],
            }
        )[:16]
        data = run_data.prepare_data(
            spec.document["data"],
            Path(paths["model"]),
            assets / f"data-{data_key}",
            max_prompt_length=inf["max_context_length"] - inf["max_response_length"],
            seed=spec.document["data"]["seed"],
        )
        result = {**paths, "identities": identities, "data": data}
        workflow.write_json(assets / "prepared.json", result)
        return result
