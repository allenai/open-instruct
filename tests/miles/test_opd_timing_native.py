"""Exercise the actual student's timing boundary without changing native input/output."""

import asyncio
import json
from types import SimpleNamespace

from miles.utils.types import Sample

from open_instruct.miles.distillation import opd_timing, opd_timing_student


def test_student_wrapper_passes_seed_and_output_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv(opd_timing.ENV, str(tmp_path))
    sample = Sample(index=4, group_index=2, metadata={})
    opd_timing.start_group([sample])
    input = SimpleNamespace(sample=sample, evaluation=False, sampling_params={"sampling_seed": 43, "temperature": 1.0})
    output = SimpleNamespace(samples=sample)

    async def native(received):
        assert received is input
        assert received.sampling_params == {"sampling_seed": 43, "temperature": 1.0}
        sample.response_length = 5
        sample.status = Sample.Status.COMPLETED
        return output

    monkeypatch.setattr(opd_timing_student.single_turn, "generate", native)
    assert asyncio.run(opd_timing_student.generate(input)) is output
    rows = [json.loads(line) for p in tmp_path.glob("events-*.jsonl") for line in p.read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["student_admission_wait", "student_request"]
    assert rows[1]["response_tokens"] == 5
    assert "teacher_request" not in str(rows)
