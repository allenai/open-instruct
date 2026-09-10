"""Independent checks reject stale policies and incorrect verifier results."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import torch as tensor_file
from scripts.miles import sft_gsm8k


def fixture_run(root):
    (root / "rollouts").mkdir()
    (root / "metrics").mkdir()
    (root / "arguments.json").write_text(
        json.dumps(["--rollout-max-response-len", "4096", "--eval-max-response-len", "4096"])
    )
    for kind, count in (("train", 8), ("eval", 16)):
        rows = [dict(input=f"{kind} question {i}", label="42") for i in range(count)]
        (root / f"{kind}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for name, version in (("0", 0), ("1", 1), ("eval_0", 0), ("eval_1", 2)):
        samples = []
        for i in range(16):
            prompt = f"eval question {i}" if name.startswith("eval") else f"train question {int(name) * 4 + i // 4}"
            samples.append(
                dict(
                    prompt=prompt,
                    label="42",
                    response="#### 42",
                    reward=1.0,
                    weight_versions=[str(version)],
                    response_length=1,
                    rollout_log_probs=[-0.5],
                )
            )
        torch.save({"samples": samples}, root / f"rollouts/{name}.pt")
    (root / "metrics/publication.jsonl").write_text("".join(json.dumps({"version": i}) + "\n" for i in range(3)))


def test_sft_audit_accepts_complete_run(tmp_path):
    fixture_run(tmp_path)
    sft_gsm8k.audit(tmp_path)
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["passed"] and report["results"]["eval_1"]["accuracy"] == 1
    assert report["results"]["0"]["mixed_reward_groups"] == 0


@pytest.mark.parametrize("change", ["stale_policy", "wrong_reward", "missing_eval_question", "nonfinite_logprob"])
def test_sft_audit_rejects_bad_evidence(tmp_path, change):
    fixture_run(tmp_path)
    path = tmp_path / "rollouts/eval_1.pt"
    data = torch.load(path, weights_only=False)
    sample = data["samples"][0]
    if change == "stale_policy":
        sample["weight_versions"] = ["1"]
    elif change == "wrong_reward":
        sample["reward"] = 0.0
    elif change == "missing_eval_question":
        sample["prompt"] = data["samples"][1]["prompt"]
    else:
        sample["rollout_log_probs"] = [float("nan")]
    torch.save(data, path)
    with pytest.raises(AssertionError):
        sft_gsm8k.audit(tmp_path)


def test_prepare_strips_solutions_and_preserves_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    tensor_file.save_file({"model.weight": torch.zeros(2, dtype=torch.bfloat16)}, source / "model.safetensors")
    template = tmp_path / "template.jinja"
    template.write_text("pinned template\n")
    monkeypatch.setattr(sft_gsm8k, "HF_SOURCE", source)
    monkeypatch.setattr(sft_gsm8k, "TEMPLATE_SOURCE", template)
    monkeypatch.setattr(sft_gsm8k, "TEMPLATE_SHA256", hashlib.sha256(b"pinned template").hexdigest())
    rows = [
        dict(
            messages=[
                dict(role="user", content=f"Question {i}"),
                dict(role="assistant", content="SECRET REFERENCE SOLUTION"),
            ],
            ground_truth=["42"],
        )
        for i in range(24)
    ]
    monkeypatch.setattr(sft_gsm8k, "load_dataset", lambda *args, **kwargs: rows)
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: messages[0]["content"] + " Assistant:",
        encode=lambda text, **kwargs: list(range(len(text))),
    )
    monkeypatch.setattr(sft_gsm8k.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    root = tmp_path / "run"
    sft_gsm8k.prepare(root)
    assert "SECRET" not in (root / "train.jsonl").read_text()
    assert len((root / "train.jsonl").read_text().splitlines()) == 8
    assert len((root / "eval.jsonl").read_text().splitlines()) == 16
    assert (root / "hf/model.safetensors").resolve() == source / "model.safetensors"
    assert not (source / "chat_template.jinja").exists()
    assert tokenizer.chat_template == "pinned template"
    report = json.loads((root / "preparation.json").read_text())
    assert report["tensor_bytes_by_dtype"] == {"BF16": 4}
