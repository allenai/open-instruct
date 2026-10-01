"""Application fixtures for record/selection integration tests."""

import enum
import json
from types import SimpleNamespace

from open_instruct.miles.configuration.config import CoreConfig


class Status(enum.Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"


def sample(
    index, reward, *, prompt="<system>Be brief.</system><user>What is 6 * 7?</user>", versions=("0",), **kwargs
):
    return SimpleNamespace(
        group_index=11,
        index=index,
        rollout_id=3,
        prompt=prompt,
        response=f"answer {index}",
        response_length=100 + index,
        reward=reward,
        status=kwargs.get("status", Status.COMPLETED),
        weight_versions=list(versions),
        metadata={
            "query": "What is 6 * 7?",
            "verifiers": [{"name": "math", "target": kwargs.get("target", "42"), "weight": 1.0}],
            "prepared_sample_id": "math:train:7",
            "source_dataset": "gsm8k",
            "source_row": 7,
            "run_prompt_token_ids_sha256": "tokens",
            "reward_components": [{"name": "math", "score": reward, "weight": 1.0, "cost": 0.0}],
            "verifier_diagnostics": {"math": {"status": kwargs.get("verifier", "ok")}},
        },
    )


def checkpoint(tmp_path, name="policy"):
    path = tmp_path / name
    path.mkdir(exist_ok=True)
    source = {"path": "/weka/source", "files": [{"path": "config.json", "size": 3, "mtime_ns": 1, "sha256": "x"}]}
    (path / "workflow-model.json").write_text(json.dumps({"identity": {"source": source}, "prepared_files": []}))
    return path


def make_args(tmp_path, *, responses="off", rate=None, run="run-a", policy=None, start=0, temperature=1.0):
    return SimpleNamespace(
        olmo_core=CoreConfig(
            records_root=str(tmp_path / "records"), records_responses=responses, records_response_sample_rate=rate
        ),
        hf_checkpoint=str(policy or checkpoint(tmp_path)),
        wandb_run_name=run,
        rollout_temperature=temperature,
        n_samples_per_prompt=4,
        start_rollout_id=start,
        rollout_seed=17,
    )


def drain(records):
    records.close()
    return records


def rows(directory, kind="group"):
    return [
        row
        for path in sorted(directory.glob("records-*.jsonl"))
        for row in map(json.loads, path.read_text().splitlines())
        if row["kind"] == kind
    ]
