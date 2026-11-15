"""Native key dispatch is deterministic and rejects unsupported applications."""

import hashlib
import inspect
import json
from types import SimpleNamespace

import pytest

from open_instruct.miles.execution import kda_dispatch as policy


class Tuner:
    arg_names = ["x", "N"]
    keys = ["N"]

    def __init__(self):
        self.configs = [SimpleNamespace(kwargs={"BH": n}, num_warps=8) for n in (1, 2)]
        self.cache = {}
        self.best_config = None
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        key = policy.runtime_key(self, args, kwargs)
        self.best_config = self.cache.setdefault(key, self.configs[0])
        return args[0]


def entry():
    return {
        "target": policy.TARGETS[0],
        "run_sha256": hashlib.sha256(inspect.getsource(Tuner.run).encode()).hexdigest(),
        "key_names": ["N"],
        "configs": {"(128, 'torch.bfloat16')": {"kwargs": {"BH": 2}, "num_warps": 8}},
    }


def test_repeated_refresh_and_fresh_use_same_existing_native_config():
    x = SimpleNamespace(dtype="torch.bfloat16")
    for tuner in (Tuner(), Tuner()):
        tuner.run = policy.prepare(tuner, entry(), "manifest-hash")
        for _ in range(5):
            tuner.cache.clear()  # A cache reset cannot change serving numerics.
            assert tuner.run(x, N=128) is x
            assert tuner.best_config is tuner.configs[1]
        assert tuner.run._oi_kda_policy == "manifest-hash"


def test_unlisted_shape_and_precision_fail_before_native_launch():
    tuner = Tuner()
    tuner.run = policy.prepare(tuner, entry(), "manifest-hash")
    for x, n in ((SimpleNamespace(dtype="torch.bfloat16"), 64), (SimpleNamespace(dtype="torch.float16"), 128)):
        with pytest.raises(ValueError, match="Unsupported KDA policy key"):
            tuner.run(x, n)
    assert tuner.calls == 0 and tuner.cache == {}


@pytest.mark.parametrize("problem", ["source", "keys", "config", "duplicate", "empty"])
def test_policy_rejects_invalid_runtime_before_mutation(problem):
    tuner, manifest = Tuner(), entry()
    if problem == "source":
        manifest["run_sha256"] = "bad"
    elif problem == "keys":
        manifest["key_names"] = ["different"]
    elif problem == "config":
        manifest["configs"]["(128, 'torch.bfloat16')"]["kwargs"] = {"BH": 99}
    elif problem == "duplicate":
        tuner.configs.append(tuner.configs[1])
    else:
        manifest["configs"] = {}
    original = tuner.run
    with pytest.raises(ValueError):
        policy.prepare(tuner, manifest, "hash")
    assert tuner.run == original and tuner.cache == {}


def test_no_policy_leaves_native_dispatch_untouched(monkeypatch):
    monkeypatch.delenv("OI_MILES_KDA_POLICY", raising=False)
    assert policy.install() is None


def test_install_validates_all_targets_before_any_mutation(monkeypatch, tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("immutable native source")
    (tmp_path / "config.json").write_bytes(source.read_bytes())
    monkeypatch.setattr(policy, "RUNTIME_LOCK", source)
    tuners = [Tuner(), Tuner()]
    modules = {}
    entries = []
    for target, tuner in zip(policy.TARGETS, tuners, strict=True):
        module_name, attr, wrapper = target.rsplit(".", 2)
        modules[module_name] = SimpleNamespace(__file__=str(source), **{attr: SimpleNamespace(**{wrapper: tuner})})
        entries.append({**entry(), "target": target, "module_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    modules["torch"] = SimpleNamespace(
        cuda=SimpleNamespace(get_device_capability=lambda device: (10, 0), set_device=lambda device: None)
    )
    monkeypatch.setattr(policy.importlib, "import_module", modules.__getitem__)
    manifest = {
        "version": 1,
        "device_capability": [10, 0],
        "serving_scope": {"quantization": "nvfp4_online", "kv_cache_dtype": "bf16"},
        "entries": entries,
        "runtime_lock_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    path = tmp_path / "policy.json"
    args = SimpleNamespace(quantization="nvfp4_online", kv_cache_dtype="bf16", model_path=str(tmp_path))
    originals = [t.run for t in tuners]
    entries[1]["module_sha256"] = "bad"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="kernel source"):
        policy.install(path, server_args=args, gpu_id=1)
    assert [t.run for t in tuners] == originals
    entries[1]["module_sha256"] = entries[0]["module_sha256"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="serving scope"):
        policy.install(path, server_args=SimpleNamespace(quantization=None, model_path=str(tmp_path)), gpu_id=1)
    assert [t.run for t in tuners] == originals
    digest = policy.install(path, server_args=args, gpu_id=1)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(t.run._oi_kda_policy == digest for t in tuners)
    assert policy.install(path, server_args=args, gpu_id=1) == digest
    manifest["provenance"] = "different manifest"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Cannot replace"):
        policy.install(path, server_args=args, gpu_id=1)


def test_evaluator_registration_uses_application_scheduler(monkeypatch):
    calls = []
    engine = type("Engine", (), {})
    modules = {
        "olmo_sglang": SimpleNamespace(register=lambda: calls.append("register")),
        "sglang.srt.entrypoints.engine": SimpleNamespace(Engine=engine),
    }
    monkeypatch.setattr(policy.importlib, "import_module", modules.__getitem__)
    monkeypatch.setenv("OI_MILES_KDA_POLICY", "immutable.json")
    policy.register()
    assert calls == ["register"]
    assert engine.run_scheduler_process_func is policy.run_scheduler_process


def test_scheduler_installs_before_native_scheduler_starts(monkeypatch):
    calls = []
    args = SimpleNamespace(quantization="nvfp4_online")
    monkeypatch.setattr(policy, "install", lambda **kw: calls.append(("install", kw["server_args"])))
    monkeypatch.setattr(
        policy.quantized_publication, "install_nvfp4_publication_context", lambda: calls.append(("loader", args))
    )
    scheduler = SimpleNamespace(run_scheduler_process=lambda *a, **kw: calls.append(("native", a[0])))
    monkeypatch.setattr(policy.importlib, "import_module", lambda _: scheduler)
    policy.run_scheduler_process(args, "ports", gpu_id=1)
    assert calls == [("install", args), ("loader", args), ("native", args)]
