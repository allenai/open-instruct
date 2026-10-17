import json
from types import SimpleNamespace

import pytest
from scripts.train.atomic_think import run_warmed

from open_instruct import atomic_resume, atomic_think


def test_warmed_sft_keeps_baseline_order_and_handoff_on_resume(tmp_path, monkeypatch):
    calls = []
    staged = set()

    def prepare(*args, **kwargs):
        # Match stage_cache's refusal to reuse a destination. The qualification
        # must support both phases within one job without overwriting staging.
        local = args[4]
        if local in staged:
            raise FileExistsError(local)
        staged.add(local)
        return {"sha": kwargs["manifest_sha256"]}

    monkeypatch.setenv("BEAKER_JOB_ID", "job")
    monkeypatch.setenv("BEAKER_REPLICA_RANK", "0")
    monkeypatch.setattr(run_warmed.run_cached, "prepare", prepare)
    monkeypatch.setattr(run_warmed.run, "train", lambda *args, **kwargs: calls.append((args, kwargs)))
    manifest = tmp_path / "manifest.json"
    run_warmed.launch(tmp_path / "fresh", "atomic-safe", manifest, "a" * 64, 512)
    args, kw = calls[-1]
    assert args[3] == 512 and kw["state_steps"] == [512]
    assert kw["warmup_manifest"] == manifest and kw["warmup_manifest_sha256"] == "a" * 64
    assert kw["offline"] and not kw["verify_resume"]
    source = tmp_path / "fresh/atomic-safe/step2"
    run_warmed.launch(tmp_path / "resumed", "atomic-safe", manifest, "a" * 64, 4, source)
    assert calls[-1][1]["verify_resume"] and calls[-1][1]["resume"] == source
    assert len(staged) == 2


def test_qualification_consumes_native_verified_receipt(tmp_path, monkeypatch):
    # Exercise the actual callback's two distinct receipt files. The short
    # restore marker deliberately does not contain exact-restoration checks.
    sha = "a" * 64
    output = tmp_path / "resumed/control/atomic-audit"
    checks = {"trainer_state_including_rng_exact": True, "optimizer_auxiliary_state_exact": True}
    monkeypatch.setattr(atomic_resume, "verify", lambda *args: dict(checks))
    for rank in range(8):
        monkeypatch.setattr(atomic_think, "get_rank", lambda rank=rank: rank)
        callback = atomic_think.AtomicAuditCallback(
            mode="control", horizon=11768, output_dir=str(output), verify_resume=True
        )
        callback.trainer = SimpleNamespace(global_step=2)
        callback.post_checkpoint_loaded(tmp_path / "fresh/control/step2")
        atomic_think.write_json(
            tmp_path / f"fresh/control/atomic-audit/warmup-handoff-rank{rank}.json", {"manifest_sha256": sha}
        )
    marker = json.loads((output / "restore-step2-rank0.json").read_text())
    assert "trainer_state_including_rng_exact" not in marker
    run_warmed.qualify_node(tmp_path, "control", sha, 0)
    assert (tmp_path / "qualification-node0.json").exists()
    atomic_think.write_json(
        output / "resume-verified-step2-rank7.json", dict(checks, optimizer_auxiliary_state_exact=False)
    )
    with pytest.raises(ValueError, match="Resume qualification failed"):
        run_warmed.qualify_node(tmp_path, "control", sha, 0)
