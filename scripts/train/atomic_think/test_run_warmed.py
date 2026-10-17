from scripts.train.atomic_think import run_warmed


def test_warmed_sft_keeps_baseline_order_and_handoff_on_resume(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("BEAKER_JOB_ID", "job")
    monkeypatch.setenv("BEAKER_REPLICA_RANK", "0")
    monkeypatch.setattr(run_warmed.run_cached, "prepare", lambda *args, **kwargs: {"sha": kwargs["manifest_sha256"]})
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
