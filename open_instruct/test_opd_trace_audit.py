import json
import math

import pytest
import torch

from open_instruct import opd_trace_audit

KL_COEF = 0.5


def _record(sample_idx: int = 0, shift_advantages: bool = False, nan_in_mask: bool = False, kl_coef=KL_COEF):
    torch.manual_seed(sample_idx)
    batch, length, prompt_len = 2, 8, 3
    mask = torch.zeros(batch, length, dtype=torch.bool)
    mask[:, prompt_len:] = True
    rollout = torch.full((batch, length), math.nan)
    rollout[mask] = -torch.rand(int(mask.sum())) * 3
    teacher = -torch.rand(batch, length) * 3
    if nan_in_mask:
        rollout[0, prompt_len] = math.nan
    advantages = torch.where(mask, kl_coef * (teacher - torch.nan_to_num(rollout)), torch.zeros(batch, length))
    if shift_advantages:
        advantages = torch.roll(advantages, shifts=1, dims=1)

    def flat(t):
        return [None if isinstance(v, float) and math.isnan(v) else v for v in t.flatten().tolist()]

    return {
        "step": 1,
        "sample_idx": sample_idx,
        "response_mask_shape": [batch, length],
        "response_mask": flat(mask.float()),
        "vllm_logprobs_shape": [batch, length],
        "vllm_logprobs": flat(rollout),
        "teacher_logprobs_shape": [batch, length],
        "teacher_logprobs": flat(teacher),
        "advantages_shape": [batch, length],
        "advantages": flat(advantages),
        "trainer_logprobs_shape": [batch, length],
        "trainer_logprobs": flat(torch.nan_to_num(rollout) + 1e-3),
    }


def _write(tmp_path, records, run_name="run", step=1):
    path = tmp_path / f"{run_name}_trainer_logprobs_step{step:06d}_rank00000.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


class TestAuditRecord:
    def test_clean_record_passes(self):
        report = opd_trace_audit.audit_record(_record(), KL_COEF, None, 1e-5)
        assert report["errors"] == []
        assert report["response_tokens"] == 10
        assert report["max_abs_opd_signal"] > 0
        assert report["mean_abs_trainer_rollout_gap"] == pytest.approx(1e-3, rel=1e-3)

    def test_shifted_advantages_fail(self):
        report = opd_trace_audit.audit_record(_record(shift_advantages=True), KL_COEF, None, 1e-5)
        assert any("differs" in e for e in report["errors"])

    def test_nan_inside_mask_fails(self):
        report = opd_trace_audit.audit_record(_record(nan_in_mask=True), KL_COEF, None, 1e-5)
        assert any("non-finite rollout" in e for e in report["errors"])

    def test_wrong_kl_coef_fails(self):
        report = opd_trace_audit.audit_record(_record(), 2 * KL_COEF, None, 1e-5)
        assert any("differs" in e for e in report["errors"])

    def test_missing_teacher_dump_is_reported(self):
        record = _record()
        del record["teacher_logprobs"]
        report = opd_trace_audit.audit_record(record, KL_COEF, None, 1e-5)
        assert "lacks" in report["errors"][0]


class TestMain:
    def test_passing_trace_exits_zero(self, tmp_path, capsys):
        _write(tmp_path, [_record(0), _record(1)])
        code = opd_trace_audit.main(
            ["--trace_dir", str(tmp_path), "--run_name", "run", "--step", "1", "--kl_coef", str(KL_COEF)]
        )
        assert code == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["steps"]["1"]["records"] == 2
        assert summary["failures"] == 0

    def test_failing_trace_exits_one(self, tmp_path, capsys):
        _write(tmp_path, [_record(0), _record(1, shift_advantages=True)])
        code = opd_trace_audit.main(
            ["--trace_dir", str(tmp_path), "--run_name", "run", "--step", "1", "--kl_coef", str(KL_COEF)]
        )
        assert code == 1
        summary = json.loads(capsys.readouterr().out)
        assert summary["steps"]["1"]["failed_records"][0]["sample_idx"] == 1

    def test_empty_run_name_matches_any_run(self, tmp_path):
        _write(tmp_path, [_record(0)], run_name="whatever__42__123")
        code = opd_trace_audit.main(["--trace_dir", str(tmp_path), "--step", "1", "--kl_coef", str(KL_COEF)])
        assert code == 0

    def test_missing_trace_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            opd_trace_audit.main(["--trace_dir", str(tmp_path), "--run_name", "run", "--step", "1"])
