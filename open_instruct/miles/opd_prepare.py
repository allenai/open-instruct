"""Immutable HF model staging and existing Open Instruct prompt preparation."""

import fcntl
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from open_instruct.miles import opd_config, run_data, workflow
from open_instruct.miles.errors import InputError


def prepare(spec):
    assets = Path(spec.output["assets"])
    assets.mkdir(parents=True, exist_ok=True)
    with (assets / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identities = {}
        paths = {}
        for role in ("model", "teacher"):
            item = spec.document[role]
            original = None
            if opd_config.is_local(item["source"]):
                original = Path(item["source"])
                if not (original / "config.json").is_file():
                    raise InputError(f"{role}.source {original} is not a Hugging Face checkpoint directory")
                stamp = workflow.fingerprint(
                    {"path": str(original), "config": (original / "config.json").read_text()}
                )[:12]
                identity = {"source": str(original), "revision": "", "thinking": False, "local_fingerprint": stamp}
                target = assets / f"{original.name}-local-{stamp}"
            else:
                identity = {"source": item["source"], "revision": item["revision"], "thinking": False}
                target = assets / f"{item['source'].split('/')[-1]}-{item['revision'][:12]}"
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
                    if dest.exists():
                        continue
                    if source.suffix == ".safetensors":
                        dest.symlink_to(source.resolve())
                    else:
                        shutil.copy2(source, dest)
                tokenizer = AutoTokenizer.from_pretrained(target)
                template = tokenizer.get_chat_template()
                tokenizer.chat_template = "{% set enable_thinking = false %}\n" + template
                tokenizer.save_pretrained(target)
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
