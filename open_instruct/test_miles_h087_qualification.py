"""Build synthetic H086-shaped fixtures from its request/hash construction."""

import asyncio
import copy
import json

import pytest

from open_instruct.miles.rewards import general_judge, h080, if_gate_yes
from open_instruct.miles.rewards import h087_qualification as qualify


@pytest.fixture
def bundle(tmp_path):
    reqs, labels = [], []
    for i, control in enumerate(["b2"] * 200 + ["a"] * 300 + ["e"] * 40):
        # H080 build_requests.request, as called by H086 fresh_requests and
        # identity_lines. Strings in conversation are already role-labelled.
        row = {
            "cell": "controls",
            "beaker": "",
            "role": "control",
            "seed": 0,
            "task": "control",
            "native_id": str(i),
            "draw": 0,
            "conversation": f"user:\nQuestion {i} {{literal}}\n",
            "response": f"Final {i}\n",
            "constraint_pass": None,
            "control": control,
        }
        row["prompt"] = h080.PROMPT_PATH.read_text().format(conversation=row["conversation"], response=row["response"])
        row["request_id"] = qualify.sha(
            qualify.encoded([row["cell"], row["task"], row["native_id"], 0, control, None])
        )
        row["sha256"] = qualify.sha(qualify.encoded(row))
        reqs.append(row)
        labels.append(
            {
                "request_id": row["request_id"],
                "request_sha256": row["sha256"],
                "label": ("yes", "partial", "no")[i % 3],
                "parse_failure": False,
                "attempts": [{"attempt": 1, "reply": "synthetic fixture"}],
            }
        )
    request_path, label_path = tmp_path / "requests.jsonl", tmp_path / "labels.jsonl"
    request_path.write_bytes(b"".join(map(qualify.encoded, reqs)))
    label_path.write_bytes(b"".join(map(qualify.encoded, labels)))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        qualify.encoded(
            {
                "artifacts": {"requests.jsonl": qualify.sha(request_path.read_bytes())},
                "configuration": {"service": h080.SERVICE},
            }
        )
    )
    pins = {
        "manifest_sha": qualify.sha(manifest_path.read_bytes()),
        "labels_sha": qualify.sha(label_path.read_bytes()),
    }
    return reqs, {r["request_id"]: r for r in labels}, request_path, label_path, pins


def test_exact_540_fixture_replay_uses_adapter_renderer_and_client(bundle, monkeypatch):
    reqs, labels, rp, lp, pins = bundle
    loaded, loaded_labels, _ = qualify.load_inputs(rp, lp, **pins)
    assert loaded == reqs and loaded_labels == labels
    seen = []
    prompts = {r["prompt"]: labels[r["request_id"]]["label"] for r in reqs}

    async def request(config, prompt):
        seen.append(prompt)
        await asyncio.sleep(0)
        return "Verdict: " + prompts[prompt]

    monkeypatch.setattr(general_judge, "async_request", request)
    report = asyncio.run(
        qualify.replay(
            loaded,
            loaded_labels,
            if_gate_yes.VerdictClient(if_gate_yes.client_config(h080.SERVICE, "http://judge/v1")),
        )
    )
    assert report["PASS"] and report["identical"] == 540
    assert seen == [r["prompt"] for r in reqs]
    assert [r["request_id"] for r in report["results"]] == [r["request_id"] for r in reqs]
    assert [r["request_sha256"] for r in report["results"]] == [r["sha256"] for r in reqs]


@pytest.mark.parametrize("bad", [None, "no"])
def test_one_mismatch_or_null_blocks_qualification(bundle, bad):
    reqs, labels, *_ = bundle

    class Client:
        async def grade(self, conversation, response):
            index = int(response.split()[1])
            return {"label": bad if index == 0 else labels[reqs[index]["request_id"]]["label"]}

    report = asyncio.run(qualify.replay(reqs, labels, Client()))
    assert not report["PASS"] and report["identical"] == 539


@pytest.mark.parametrize("which", ["manifest", "labels", "request", "order", "id", "prompt"])
def test_pin_corruption_order_and_prompt_drift_block_before_grading(bundle, which):
    reqs, _, rp, lp, pins = bundle
    if which == "manifest":
        pins["manifest_sha"] = "0" * 64
    elif which == "labels":
        lp.write_bytes(lp.read_bytes() + b"\n")
    else:
        changed = copy.deepcopy(reqs)
        if which == "request":
            changed[0]["response"] += "tampered"
        elif which == "order":
            changed[0], changed[200] = changed[200], changed[0]
        elif which == "id":
            changed[1] = changed[0]
        else:
            # Even a consistently re-hashed row must render from fields, not
            # submit its stored prompt blindly. Update its label's hash too.
            changed[0]["prompt"] += "tampered"
            changed[0]["sha256"] = qualify.sha(qualify.encoded({k: v for k, v in changed[0].items() if k != "sha256"}))
            records = [json.loads(line) for line in lp.read_text().splitlines()]
            records[0]["request_sha256"] = changed[0]["sha256"]
            lp.write_bytes(b"".join(map(qualify.encoded, records)))
            pins["labels_sha"] = qualify.sha(lp.read_bytes())
        rp.write_bytes(b"".join(map(qualify.encoded, changed)))
        manifest = json.loads(rp.with_name("manifest.json").read_text())
        manifest["artifacts"]["requests.jsonl"] = qualify.sha(rp.read_bytes())
        rp.with_name("manifest.json").write_bytes(qualify.encoded(manifest))
        pins["manifest_sha"] = qualify.sha(rp.with_name("manifest.json").read_bytes())
    with pytest.raises(ValueError):
        qualify.load_inputs(rp, lp, **pins)


def test_input_failure_writes_a_blocking_json_report(tmp_path):
    path = tmp_path / "qualification.json"
    with pytest.raises(RuntimeError, match="qualification failed"):
        asyncio.run(
            qualify.qualify(
                h080.SERVICE,
                "http://judge/v1",
                path,
                requests_path=tmp_path / "missing.jsonl",
                labels_path=tmp_path / "missing-labels.jsonl",
            )
        )
    assert json.loads(path.read_text())["PASS"] is False
    with pytest.raises(FileExistsError):
        asyncio.run(qualify.qualify(h080.SERVICE, "http://judge/v1", path))
