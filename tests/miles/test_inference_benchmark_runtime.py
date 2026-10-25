"""Worker lifecycle regressions using the pinned runtime's specification contract."""

from types import SimpleNamespace

from open_instruct.miles.execution import benchmark


def test_cache_hooks_see_trainers_but_no_trainer_is_launched(monkeypatch):
    names = ["trainer-actor", "inference-router-0", "inference-session", "inference-engine-0"]
    specs = [SimpleNamespace(name=name) for name in names]
    seen = []

    def configure(args, workers):
        # The pinned helper validates both prefixes before attaching hooks.
        assert any(worker.name.startswith("trainer-") for worker in workers)
        assert any(worker.name.startswith("inference-engine-") for worker in workers)
        seen.extend(workers)
        return workers

    monkeypatch.setattr(benchmark.startup_cache, "configure_specs", configure)
    selected = benchmark.inference_specs(SimpleNamespace(), specs)
    assert seen == specs
    assert [worker.name for worker in selected] == names[1:]
