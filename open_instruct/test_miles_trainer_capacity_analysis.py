"""Distributed rates use total tokens and the slower rank, not summed rank time."""

import json

import pytest
from scripts.miles import analyze_trainer_capacity


def reports(root):
    for rank in range(2):
        (root / f"trainer-capacity-rank{rank}.json").write_text(
            json.dumps(
                dict(
                    rank=rank,
                    passed=True,
                    variant="fixture",
                    initialization_seconds=10,
                    batches=[
                        dict(
                            update=0,
                            source_sha256="same",
                            seconds=2 + rank,
                            local_tokens=30,
                            local_response_tokens=15,
                            phases={"optimizer": 0.1},
                            other_seconds=1,
                            compilation={"jit_miss_count": 2, "cache_artifact_writes_by_extension": {".cubin": 1}},
                            dynamo_stats={"unique_graphs": 3},
                            memory_allocated_peak=10,
                        )
                    ],
                )
            )
        )


def test_distributed_rate_and_compilation_evidence(tmp_path):
    reports(tmp_path)
    result = analyze_trainer_capacity.analyze(tmp_path, 0)
    assert result["warm_seconds"] == 3
    assert result["warm_model_tokens_per_second_per_gpu"] == 10
    assert result["warm_response_tokens_per_second_per_gpu"] == 5
    assert result["per_update"][0]["jit_misses_per_rank"] == [2, 2]


@pytest.mark.parametrize(
    "field,value,match", [("passed", False, "complete"), ("source_sha256", "different", "identities")]
)
def test_incomplete_or_mismatched_ranks_rejected(tmp_path, field, value, match):
    reports(tmp_path)
    path = tmp_path / "trainer-capacity-rank1.json"
    report = json.loads(path.read_text())
    (report if field == "passed" else report["batches"][0])[field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        analyze_trainer_capacity.analyze(tmp_path, 0)
