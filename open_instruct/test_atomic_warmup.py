"""Exact native master injection and checkpoint identity checks."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import torch as safe_torch
from torch import distributed as dist
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

from open_instruct import atomic_think, atomic_warmup


@pytest.mark.parametrize("scope", ["full", "rows"])
def test_handoff_keeps_fp32_values_other_masters_and_fresh_moments(tmp_path, scope):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        source = torch.arange(40, dtype=torch.float32).reshape(10, 4) / 13 + 0.00013
        params = {name: torch.nn.Parameter(source.bfloat16().clone()) for name in atomic_warmup.MATRICES}
        states = {}
        for name in [*params, "backbone"]:
            states[f"{name}.main"] = distribute_tensor(source.flatten().clone(), mesh, [Shard(0)])
            for suffix in ("exp_avg", "exp_avg_sq"):
                states[f"{name}.{suffix}"] = distribute_tensor(torch.zeros(40), mesh, [Shard(0)])
            states[f"{name}.step"] = torch.tensor(0.0)
        optimizer = SimpleNamespace(states=states, param_groups=[{"named_params": params}])
        model = SimpleNamespace(
            tie_word_embeddings=False,
            embeddings=SimpleNamespace(weight=params["embeddings.weight"]),
            lm_head=SimpleNamespace(w_out=SimpleNamespace(weight=params["lm_head.w_out.weight"])),
        )
        values = {name: (source + 0.000031).clone() for name in params}
        entries = {
            name: {"scope": scope, "rows": [2, 7], "base_master_sha256": atomic_think.tensor_hash(source)}
            for name in params
        }
        if scope == "rows":
            values = {name: value[[2, 7]].clone() for name, value in values.items()}
        atomic_warmup.inject(
            SimpleNamespace(model=model, optim=optimizer), {"matrices": entries}, values, tmp_path, "a" * 64
        )
        for name, param in params.items():
            expected = source.clone()
            if scope == "rows":
                expected[[2, 7]] = values[name]
            else:
                expected = values[name]
            assert torch.equal(states[f"{name}.main"].full_tensor().reshape_as(source), expected)
            assert torch.equal(param, expected.bfloat16())
            assert not torch.equal(expected, expected.bfloat16().float())
            assert torch.count_nonzero(states[f"{name}.exp_avg"].full_tensor()) == 0
        assert torch.equal(states["backbone.main"].full_tensor(), source.flatten())
        assert json.loads((tmp_path / "warmup-handoff-rank0.json").read_text())["optimizer_policy"] == "fresh-sft"
        states["backbone.step"].fill_(1)
        with pytest.raises(ValueError, match="fresh SFT"):
            atomic_warmup.inject(
                SimpleNamespace(model=model, optim=optimizer), {"matrices": entries}, values, tmp_path, "a" * 64
            )
    finally:
        dist.destroy_process_group()


def test_resume_pins_warmup_identity():
    callback = atomic_warmup.WarmupIdentity("a" * 64)
    callback.trainer = SimpleNamespace(global_step=512)
    with pytest.raises(ValueError, match="missing"):
        callback.pre_train()
    with pytest.raises(ValueError, match="identity"):
        callback.load_state_dict({"manifest_sha256": "b" * 64, "optimizer_policy": "fresh-sft"})
    callback.load_state_dict(callback.state_dict())
    callback.pre_train()


def test_manifest_checks_identity_rows_precision_and_file_integrity(tmp_path):
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"<think>": 100256, "</think>": 100270},
        promoted_reserved_slot_tokens=[SimpleNamespace(token_id=i, source_ids=()) for i in (100256, 100270)],
    )
    values = {name: torch.randn(2, 1024) for name in atomic_warmup.MATRICES}
    payload = tmp_path / "matrices.safetensors"
    safe_torch.save_file(values, str(payload))
    manifest = {
        "version": 1,
        "optimizer_policy": "fresh-sft",
        "base": "/base",
        "encoding": "atomic",
        "tokenizer_vocab_sha256": atomic_warmup.vocab_digest(tokenizer),
        "file": payload.name,
        "file_sha256": atomic_warmup.file_digest(payload),
        "matrices": {
            name: {"scope": "rows", "rows": [100256, 100270], "tensor_sha256": atomic_think.tensor_hash(value)}
            for name, value in values.items()
        },
    }
    path = tmp_path / "manifest.json"

    def read():
        path.write_text(json.dumps(manifest))
        return atomic_warmup.load_manifest(path, atomic_warmup.file_digest(path), "/base", tokenizer, "atomic-safe")

    _, loaded = read()
    assert all(torch.equal(loaded[name], value) for name, value in values.items())
    manifest["base"] = "/wrong"
    with pytest.raises(ValueError, match="base"):
        read()
    manifest["base"] = "/base"
    manifest["matrices"]["embeddings.weight"]["rows"].reverse()
    with pytest.raises(ValueError, match="promoted"):
        read()
    manifest["matrices"]["embeddings.weight"]["rows"].reverse()
    values["embeddings.weight"] = values["embeddings.weight"].bfloat16()
    safe_torch.save_file(values, str(payload))
    with pytest.raises(ValueError, match="file hash"):
        read()
    manifest["file_sha256"] = atomic_warmup.file_digest(payload)
    with pytest.raises(ValueError, match="finite FP32"):
        read()
