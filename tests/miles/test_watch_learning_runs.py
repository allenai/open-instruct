"""The local watcher never retries an ambiguous external submission."""

import json
import subprocess
from pathlib import Path

import pytest
from scripts.miles import watch_learning_runs as watch


def configuration(tmp_path):
    return {
        "checkout": str(tmp_path / "checkout"),
        "source": "committed-source",
        "analysis_image": "sha256:immutable",
        "base_image": "qualified-base",
        "experiments": dict(watch.EXPERIMENTS),
    }


class Commands:
    def __init__(self, *, core_exit=0, audit_exit=None, submission_error=None):
        self.exits = {**dict.fromkeys(watch.EXPERIMENTS.values(), 0), "audit-experiment": audit_exit}
        self.exits[watch.EXPERIMENTS["core"]] = core_exit
        self.submission_error = submission_error
        self.submissions, self.calls = 0, []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if argv[:3] == ["git", "rev-parse", "HEAD"]:
            return "committed-source\n"
        if argv[0] == "git":
            return ""
        if argv[0].endswith("build_image_and_launch.sh"):
            self.submissions += 1
            assert argv[-2:] == ["--stage", "audit"] and "--miles" in argv
            if self.submission_error:
                raise self.submission_error
            return 'Build output\n[{"id":"audit-experiment","workspaceRef":{"name":"open-instruct-dev"}}]\n'
        if argv[:3] == ["beaker", "experiment", "get"]:
            experiment = argv[3]
            status = {"scheduled": "2026-09-11T00:00:00Z"}
            code = self.exits[experiment]
            if code is not None:
                status.update(exited="2026-09-11T02:00:00Z", finalized="2026-09-11T02:00:01Z", exitCode=code)
            return json.dumps(
                [
                    {
                        "id": experiment,
                        "jobs": [
                            {
                                "id": "job-" + experiment,
                                "status": status,
                                "result": {"beaker": "data-" + experiment},
                                "execution": {"envVars": [{"value": "do-not-persist"}]},
                            }
                        ],
                    }
                ]
            )
        if argv[:3] == ["beaker", "job", "logs"]:
            return "retained final log\n"
        if argv[:3] == ["beaker", "dataset", "fetch"]:
            target = Path(argv[-1])
            target.mkdir()
            if "audit" in argv[3]:
                for name in ["core-audit.json", "megatron-audit.json", "comparison.json"]:
                    (target / name).write_text('{"valid":true}')
            return "fetched"
        if argv[0] == "docker":
            mount = next(x for x in argv if x.startswith("type=bind,src=") and x.endswith(",dst=/watch"))
            root = Path(mount.removeprefix("type=bind,src=").removesuffix(",dst=/watch"))
            report = {
                "valid": True,
                "learning_curves": [
                    {"completed_steps": 0, "core": {"accuracy": 99 / 128}, "megatron": {"accuracy": 98 / 128}},
                    {"completed_steps": 500, "core": {"accuracy": 100 / 128}, "megatron": {"accuracy": 102 / 128}},
                ],
                "allocated_runtime_seconds": {"core": 7200, "megatron": 7200},
            }
            (root / "analysis/comparison.json").write_text(json.dumps(report))
            (root / "analysis/comparison.png").write_bytes(b"plot fixture")
            return "analyzed"
        raise AssertionError(argv)


def test_failed_arm_never_submits_audit_and_retains_terminal_evidence(tmp_path):
    runner = Commands(core_exit=1)
    w = watch.Watcher(tmp_path / "out", configuration(tmp_path), runner)
    assert w.poll()
    assert runner.submissions == 0
    assert w.state["attention"] and w.state["jobs"]["core"]["captured"]
    assert (w.output / "core/capture/final.log").read_text() == "retained final log\n"
    assert "do-not-persist" not in w.path.read_text()


def test_success_submits_once_including_watcher_restart(tmp_path):
    runner = Commands()
    config = configuration(tmp_path)
    w = watch.Watcher(tmp_path / "out", config, runner)
    assert not w.poll()
    original_deadline = w.state["created"]
    restarted = watch.Watcher(w.output, config, runner)
    assert not restarted.poll()
    assert runner.submissions == 1 and restarted.state["created"] == original_deadline


@pytest.mark.parametrize("failure", [RuntimeError("network after submission"), ValueError("invalid output")])
def test_ambiguous_submission_is_never_retried(tmp_path, failure):
    runner = Commands(submission_error=failure)
    w = watch.Watcher(tmp_path / "out", configuration(tmp_path), runner)
    assert w.poll()
    assert w.state["audit"]["status"] == "ambiguous"
    restarted = watch.Watcher(w.output, w.config, runner)
    assert restarted.poll()
    assert runner.submissions == 1 and restarted.state["attention"]


def test_crash_after_durable_submission_intent_requires_manual_recovery(tmp_path):
    runner = Commands(submission_error=SystemExit("crash before result was saved"))
    w = watch.Watcher(tmp_path / "out", configuration(tmp_path), runner)
    with pytest.raises(SystemExit):
        w.poll()
    assert json.loads(w.path.read_text())["audit"]["status"] == "submitting"
    restarted = watch.Watcher(w.output, w.config, runner)
    assert restarted.state["audit"]["status"] == "ambiguous"
    restarted.poll()
    assert runner.submissions == 1


def test_passing_audit_runs_pinned_analysis_once_with_actual_allocation(tmp_path):
    runner = Commands(audit_exit=0)
    w = watch.Watcher(tmp_path / "out", configuration(tmp_path), runner)
    assert w.poll() and w.state["analysis"] == "complete"
    restart = watch.Watcher(w.output, w.config, runner)
    assert restart.poll()
    docker = [argv for argv, _ in runner.calls if argv[0] == "docker"]
    assert len(docker) == 1 and "sha256:immutable" in docker[0]
    for backend in ["core", "megatron"]:
        index = docker[0].index(f"--{backend}-allocated-seconds")
        assert float(docker[0][index + 1]) == 7200
    assert "fresh 500-update" in (w.output / "analysis/comparison.md").read_text()
    assert "Configuration differences" in (w.output / "analysis/comparison.md").read_text()


def test_failed_audit_is_retained_without_analysis_or_resubmission(tmp_path):
    runner = Commands(audit_exit=1)
    w = watch.Watcher(tmp_path / "out", configuration(tmp_path), runner)
    assert w.poll()
    w.poll()
    assert runner.submissions == 1 and w.state["jobs"]["audit"]["captured"]
    assert not any(argv[0] == "docker" for argv, _ in runner.calls)


def test_changed_config_and_changed_checkout_are_not_silently_accepted(tmp_path):
    config = configuration(tmp_path)
    w = watch.Watcher(tmp_path / "out", config, Commands())
    with pytest.raises(ValueError, match="persisted identity"):
        watch.Watcher(w.output, {**config, "source": "other"}, Commands())
    runner = Commands()

    def changed(argv, **kwargs):
        if argv == ["git", "rev-parse", "HEAD"]:
            return "other"
        return runner(argv, **kwargs)

    w.runner = changed
    w.poll()
    assert runner.submissions == 0 and w.state["audit"]["status"] == "ambiguous"


def test_parser_rejects_nonterminal_or_unrecognized_submission_output():
    with pytest.raises(ValueError):
        watch.submitted_id('[{"id":"looks-like-an-image"}]')
    with pytest.raises(ValueError):
        watch.submitted_id('[{"id":"audit","workspaceRef":{}}]\nUnexpected trailing output')


def test_explicit_light_identity_is_used_and_persisted(tmp_path):
    config = configuration(tmp_path)
    config["experiments"]["light"] = "replacement-approved-by-user"
    runner = Commands(core_exit=1)
    runner.exits["replacement-approved-by-user"] = 0
    w = watch.Watcher(tmp_path / "out", config, runner)
    assert w.poll()
    assert w.state["jobs"]["light"]["experiment"] == "replacement-approved-by-user"


def test_capture_fetch_retry_uses_fresh_staging_and_does_not_repeat_audit(tmp_path):
    runner = Commands(audit_exit=0)
    failed = False

    def first_fetch_fails(argv, **kwargs):
        nonlocal failed
        if argv[:3] == ["beaker", "dataset", "fetch"] and watch.EXPERIMENTS["core"] in argv[3] and not failed:
            failed = True
            Path(argv[-1]).mkdir()
            raise RuntimeError("transient partial download")
        return runner(argv, **kwargs)

    w = watch.Watcher(tmp_path / "out", configuration(tmp_path), first_fetch_fails)
    assert not w.poll()
    assert w.state["jobs"]["core"]["capture_attempts"] == 1
    assert not w.state["attention"]
    assert (w.output / "core/capture-attempt-1/capture-error.json").exists()
    restarted = watch.Watcher(w.output, w.config, first_fetch_fails)
    assert restarted.poll()
    assert restarted.state["jobs"]["core"]["capture_attempts"] == 2
    assert restarted.state["analysis"] == "complete" and runner.submissions == 1
    assert (w.output / "core/capture/complete.json").is_file()


def test_three_capture_failures_record_attention_and_stop_retrying(tmp_path):
    runner = Commands(core_exit=1)

    def broken_fetch(argv, **kwargs):
        if argv[:3] == ["beaker", "dataset", "fetch"]:
            raise RuntimeError("network down")
        return runner(argv, **kwargs)

    w = watch.Watcher(tmp_path / "out", configuration(tmp_path), broken_fetch)
    assert not w.poll()
    assert not w.poll()
    assert w.poll()
    assert all(j["capture_attempts"] == 3 for j in w.state["jobs"].values())
    assert any("exhausted three" in x for x in w.state["attention"])
    w.poll()
    assert all(j["capture_attempts"] == 3 for j in w.state["jobs"].values())


def test_expired_watch_does_not_call_external_commands(tmp_path):
    runner = Commands()
    w = watch.Watcher(tmp_path / "out", configuration(tmp_path), runner)
    w.state["created"] = "2000-01-01T00:00:00+00:00"
    with pytest.raises(TimeoutError, match="deadline"):
        w.call(["beaker", "experiment", "get", "anything"])
    assert runner.calls == []


def test_timeout_bytes_are_recorded_as_ambiguous_without_retry(tmp_path):
    failure = subprocess.TimeoutExpired(["wrapper"], 1, output=b"possibly submitted", stderr=b"transport timeout")
    runner = Commands(submission_error=failure)
    w = watch.Watcher(tmp_path / "out", configuration(tmp_path), runner)
    assert w.poll() and w.state["audit"]["status"] == "ambiguous"
    assert "possibly submitted" in (w.output / "submission.log").read_text()
    w.poll()
    assert runner.submissions == 1


def test_submission_discards_inherited_existing_image_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MILES_EXISTING_IMAGE", "unrelated-profiler-image")
    runner = Commands()
    w = watch.Watcher(tmp_path / "out", configuration(tmp_path), runner)
    w.poll()
    _, kwargs = next((argv, kw) for argv, kw in runner.calls if argv[0].endswith("build_image_and_launch.sh"))
    assert "MILES_EXISTING_IMAGE" not in kwargs["env"]
