"""CPU regressions for TPU tooling; cloud commands and checkpoint I/O are replaced locally."""

import copy
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import types
from importlib import util
from pathlib import Path

import datasets
import numpy as np
import pytest
import torch
import yaml
from pyarrow import parquet as pq
from safetensors import torch as safetensors_torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/tpu/maxtext"
BUILDERS = [
    ("data/build_full_dolci.py", "allenai/Dolci-Instruct-SFT"),
    ("olmo35/build_full_dolci_think.py", "allenai/Dolci-Think-SFT"),
]


def load_script(name):
    spec = util.spec_from_file_location("tpu_test_script", SCRIPTS / name)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script,dataset_id", BUILDERS)
def test_builder_resolves_one_revision_and_shards_each_row_once(monkeypatch, tmp_path, script, dataset_id):
    module = load_script(script)
    rows = datasets.Dataset.from_list(
        [
            {"id": str(i), "messages": [{"role": "assistant", "content": str(i)}], "source_dataset": "test"}
            for i in range(6)
        ]
    )
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return rows

    monkeypatch.setattr(module.datasets, "load_dataset", load)
    out = tmp_path / "shards"
    monkeypatch.setattr(sys, "argv", [script, str(out), "2", "7", "--revision", "one-commit"])
    module.main()
    assert calls == [((dataset_id,), {"revision": "one-commit", "split": "train", "cache_dir": None})]
    actual = [row for shard in out.glob("*.parquet") for row in pq.read_table(shard).to_pylist()]
    assert sorted(row["id"] for row in actual) == [str(i) for i in range(6)]
    assert all(row["messages"][0]["content"] == row["id"] for row in actual)


@pytest.mark.parametrize("script,dataset_id", BUILDERS)
def test_builder_refuses_old_output_before_loading(monkeypatch, tmp_path, script, dataset_id):
    module = load_script(script)
    stale = tmp_path / "train-00000-of-00003.parquet"
    stale.write_bytes(b"old dataset")
    monkeypatch.setattr(module.datasets, "load_dataset", lambda *a, **kw: pytest.fail("must reject before loading"))
    monkeypatch.setattr(sys, "argv", [script, str(tmp_path), "2", "7"])
    with pytest.raises(SystemExit, match="2"):
        module.main()
    assert stale.read_bytes() == b"old dataset"


@pytest.mark.parametrize("script,dataset_id", BUILDERS)
def test_builder_accepts_explicit_local_files(monkeypatch, tmp_path, script, dataset_id):
    module = load_script(script)
    source = tmp_path / "source.parquet"
    datasets.Dataset.from_list(
        [{"id": "a", "messages": [{"role": "assistant", "content": "hello"}], "source_dataset": "test"}]
    ).to_parquet(source)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", [script, str(out), "1", "0", "--data-files", str(source), "--cache-dir", str(tmp_path / "cache")]
    )
    module.main()
    assert pq.read_table(next(out.glob("*.parquet"))).column("id").to_pylist() == ["a"]


def executable(path, source):
    path.write_text(source)
    path.chmod(0o755)


@pytest.mark.parametrize(
    "state,api_error,log_error,expected",
    [
        ("Completed", "0", "0", 0),
        ("Completed", "once", "0", 0),
        ("Failed", "0", "0", 1),
        ("", "0", "0", 124),
        ("", "42", "0", 42),
        ("Completed", "0", "1", 1),
    ],
)
def test_watcher_reports_outcome(tmp_path, state, api_error, log_error, expected):
    executable(
        tmp_path / "kubectl",
        """#!/bin/bash
case "$*" in
  *"get jobset"*)
    if [[ "$API_ERROR" == once ]]; then
      if [[ ! -e "$FAIL_ONCE" ]]; then touch "$FAIL_ONCE"; exit 42; fi
      printf '%s' "$FAKE_STATE"; exit 0
    fi
    printf '%s' "$FAKE_STATE"; exit "$API_ERROR" ;;
  *"get pods"*"-o name"*) echo pod/test ;;
  *"get pods"*) echo 'test 1/1 Running 0' ;;
  *"logs"*)
    [[ "$*" != *--request-timeout* ]] || exit 99
    echo 'step log'; exit "$LOG_ERROR" ;;
  *) exit 99 ;;
esac
""",
    )
    executable(tmp_path / "sleep", "#!/bin/sh\nexit 0\n")
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "FAKE_STATE": state,
        "API_ERROR": api_error,
        "FAIL_ONCE": str(tmp_path / "failed-once"),
        "LOG_ERROR": log_error,
        "RUNS_DIR": str(tmp_path / "logs"),
    }
    result = subprocess.run(
        ["bash", str(SCRIPTS / "k8s/watch_jobset.sh"), "job", "run", "1"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    if log_error == "0":
        assert (tmp_path / "logs/run/test.log").read_text() == "step log\n"
    if api_error == "42":
        assert "state=running" not in result.stdout


def test_watcher_collects_other_pods_after_one_log_failure(tmp_path):
    executable(
        tmp_path / "kubectl",
        """#!/bin/bash
case "$*" in
  *"get jobset"*) echo Completed ;;
  *"get pods"*"-o name"*) printf 'pod/bad\npod/good\n' ;;
  *"get pods"*) echo 'good 1/1 Running 0' ;;
  *"logs pod/bad"*) exit 1 ;;
  *"logs pod/good"*) echo 'good log' ;;
  *) exit 99 ;;
esac
""",
    )
    logs = tmp_path / "logs/run"
    logs.mkdir(parents=True)
    (logs / "bad.log").write_text("previous snapshot")
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "RUNS_DIR": str(tmp_path / "logs")}
    result = subprocess.run(
        ["bash", str(SCRIPTS / "k8s/watch_jobset.sh"), "job", "run", "1"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1
    assert (logs / "good.log").read_text() == "good log\n"
    assert (logs / "bad.log").read_text() == "previous snapshot"


@pytest.mark.parametrize("image", ["image", "image-olmo35-rf", "image-olmo35-fs"])
@pytest.mark.parametrize("service_account", ["", "projects/test/serviceAccounts/builder@example.com"])
def test_image_build_config_is_portable_and_account_is_optional(tmp_path, image, service_account):
    root = tmp_path / "maxtext"
    shutil.copytree(SCRIPTS / image, root / image)
    executable(
        root / "fetch_patches.sh",
        """#!/bin/bash
set -eu
read -r _ _ _ name < maxtext_source
mkdir -p patches
printf 'test patch\n' > "patches/$name"
""",
    )
    executable(
        tmp_path / "gcloud",
        """#!/bin/bash
for arg in "$@"; do
  case "$arg" in --config=*) cp "${arg#--config=}" "$CAPTURE_CONFIG" ;; esac
done
""",
    )
    capture = tmp_path / "build.yaml"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "TMPDIR": str(tmp_path),
        "CAPTURE_CONFIG": str(capture),
        "BUILD_SERVICE_ACCOUNT": service_account,
    }
    result = subprocess.run(
        ["bash", str(root / image / "build.sh")], env=env, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stdout + result.stderr
    config = yaml.safe_load(capture.read_text())
    assert config.get("serviceAccount", "") == service_account
    digest = hashlib.sha256((root / image / "Dockerfile").read_bytes() + b"test patch\n").hexdigest()[:8]
    prefix = {
        "image": "e259ed61-p",
        "image-olmo35-rf": "51f8cc4b-p907d86f3-rf",
        "image-olmo35-fs": "51f8cc4b-p907d86f3-rfb46e215e-bs3ec936c3-dc88c6c7ea-fs",
    }[image]
    assert config["images"][0].endswith(prefix + digest)
    assert not list(tmp_path.glob("maxtext-build.*"))


def checkpoint_stubs(monkeypatch, state, saves=True):
    saved = {}

    def tree_map(fn, value, is_leaf=None):
        if is_leaf and is_leaf(value):
            return fn(value)
        return (
            {key: tree_map(fn, child, is_leaf) for key, child in value.items()}
            if isinstance(value, dict)
            else fn(value)
        )

    class Manager:
        def __init__(self, *args, **kwargs):
            saved["out"] = args[0]

        def save(self, step, args):
            if saves:
                saved.update(args)
            return saves  # orbax returns False when the step already exists

        def wait_until_finished(self):
            pass

    checkpoint = types.SimpleNamespace(
        PyTreeCheckpointer=lambda: types.SimpleNamespace(
            metadata=lambda path: types.SimpleNamespace(tree=state), restore=lambda *a, **kw: copy.deepcopy(state)
        ),
        RestoreArgs=lambda **kw: None,
        CheckpointManager=Manager,
        args=types.SimpleNamespace(Composite=lambda **kw: kw, PyTreeSave=lambda value: value),
    )
    monkeypatch.setitem(sys.modules, "jax", types.SimpleNamespace(tree=types.SimpleNamespace(map=tree_map)))
    monkeypatch.setitem(sys.modules, "orbax", types.SimpleNamespace(checkpoint=checkpoint))
    return saved


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("remote", [False, True])
def test_resize_handles_tunix_and_converter_layouts(monkeypatch, tmp_path, wrapped, remote):
    def leaf(value):
        return {"value": value} if wrapped else value

    params = {
        "token_embedder": {"embedding": leaf(np.arange(8).reshape(4, 2))},
        "decoder": {"logits_dense": {"kernel": leaf(np.arange(8).reshape(2, 4))}},
    }
    state = params if wrapped else {"params": {"params": params}}
    saved = checkpoint_stubs(monkeypatch, state)
    out = "gs://test-bucket/new-checkpoint" if remote else str(tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["resize_vocab.py", "source", out, "3"])
    runpy.run_path(str(SCRIPTS / "resize_vocab.py"), run_name="__main__")
    result = saved["items"] if wrapped else saved["items"]["params"]["params"]
    assert saved["out"] == out
    embedding = result["token_embedder"]["embedding"]
    kernel = result["decoder"]["logits_dense"]["kernel"]
    if wrapped:
        embedding, kernel = embedding["value"], kernel["value"]
    np.testing.assert_array_equal(embedding, np.arange(8).reshape(4, 2)[:3])
    np.testing.assert_array_equal(kernel, np.arange(8).reshape(2, 4)[:, :3])


def test_resize_refuses_existing_step(monkeypatch, tmp_path):
    params = {
        "token_embedder": {"embedding": np.arange(8).reshape(4, 2)},
        "decoder": {"logits_dense": {"kernel": np.arange(8).reshape(2, 4)}},
    }
    checkpoint_stubs(monkeypatch, {"params": {"params": params}}, saves=False)
    monkeypatch.setattr(sys, "argv", ["resize_vocab.py", "source", str(tmp_path / "out"), "3"])
    with pytest.raises(SystemExit, match="already holds step 0"):
        runpy.run_path(str(SCRIPTS / "resize_vocab.py"), run_name="__main__")


def test_export_uses_sharded_reference_metadata_without_old_weights(monkeypatch, tmp_path):
    params = {
        "token_embedder": {"embedding": np.array([[1, 2]], dtype=np.float32)},
        "decoder": {
            "embedding_norm": {"scale": np.array([3, 4], dtype=np.float32)},
            "decoder_norm": {"scale": np.array([5, 6], dtype=np.float32)},
            "logits_dense": {"kernel": np.array([[7], [8]], dtype=np.float32)},
        },
    }
    checkpoint_stubs(monkeypatch, {"params": {"params": params}})
    ref = tmp_path / "ref"
    ref.mkdir()
    shapes = {
        "model.embed_tokens.weight": (1, 2),
        "model.embed_norm.weight": (2,),
        "model.norm.weight": (2,),
        "lm_head.weight": (1, 2),
    }
    weight_map = {}
    for i, (key, shape) in enumerate(shapes.items()):
        name = f"model-{i}.safetensors"
        safetensors_torch.save_file({key: torch.zeros(shape)}, str(ref / name))
        weight_map[key] = name
    (ref / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (ref / "config.json").write_text(
        json.dumps({"vocab_size": 1, "num_hidden_layers": 0, "num_attention_heads": 1, "head_dim": 2})
    )
    (ref / "chat_templates").mkdir()
    (ref / "chat_templates/default.jinja").write_text("template")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["export", "source", str(ref), str(out)])
    runpy.run_path(str(SCRIPTS / "olmo35/export_maxtext_to_hf.py"), run_name="__main__")
    assert sorted(path.name for path in out.glob("*.safetensors")) == ["model.safetensors"]
    assert not (out / "model.safetensors.index.json").exists()
    assert (out / "chat_templates/default.jinja").read_text() == "template"
    actual = safetensors_torch.load_file(str(out / "model.safetensors"))
    torch.testing.assert_close(actual["lm_head.weight"], torch.tensor([[7, 8]], dtype=torch.bfloat16))


def test_export_refuses_existing_output(monkeypatch, tmp_path):
    checkpoint_stubs(monkeypatch, {})
    (tmp_path / "existing.safetensors").write_bytes(b"old model")
    monkeypatch.setattr(sys, "argv", ["export", "source", "reference", str(tmp_path)])
    with pytest.raises(ValueError, match="new or empty"):
        runpy.run_path(str(SCRIPTS / "olmo35/export_maxtext_to_hf.py"), run_name="__main__")
    assert (tmp_path / "existing.safetensors").read_bytes() == b"old model"


def test_count_tokens_writes_only_the_requested_output(tmp_path):
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{% for message in messages %}{{ message.content }} {% endfor %}"
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    data = tmp_path / "data"
    data.mkdir()
    datasets.Dataset.from_list(
        [{"messages": [{"role": "assistant", "content": text}]} for text in ["hello", "hello world"]]
    ).to_parquet(data / "train.parquet")
    output = tmp_path / "lengths.npy"
    command = [
        sys.executable,
        str(SCRIPTS / "data/count_tokens.py"),
        str(data),
        "32",
        str(tmp_path / "tokenizer"),
        "--output",
        str(output),
        "--workers",
        "1",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_array_equal(np.load(output), [1, 2])
    assert not (data / "dolci_lens.npy").exists()
    previous = output.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert output.read_bytes() == previous


def test_olmo3_export_resolves_relative_paths_before_changing_directory(tmp_path):
    python = tmp_path / "python wrapper"
    trace = tmp_path / "trace.jsonl"
    executable(
        python,
        """#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path

with open(os.environ['TRACE'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1] == '-c':
    if sys.argv[2] == 'import sys; print(sys.executable)':
        print(Path(__file__).resolve())
    else:
        subprocess.run([sys.executable, *sys.argv[1:]], check=True)
elif sys.argv[1] == '-':
    code = sys.stdin.read()
    if 'PyTreeCheckpointer' in code:
        assert Path(sys.argv[2]).is_absolute()
        print(100352)
    elif 'PreTrainedTokenizerFast' in code:
        assert all(Path(arg).is_absolute() for arg in sys.argv[2:])
        print('metadata copied')
    else:
        sys.argv = sys.argv[1:]
        exec(code)
elif sys.argv[1].endswith('resize_vocab.py'):
    assert Path(sys.argv[1]).is_absolute() and Path(sys.argv[1]).is_file()
    assert Path(sys.argv[2]).is_absolute()
    (Path(sys.argv[3]) / '0/items').mkdir(parents=True)
elif sys.argv[1] == '-m':
    assert sys.argv[2] == 'maxtext.checkpoint_conversion.to_huggingface'
    out = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('base_output_directory='))
    Path(out).mkdir()
else:
    raise SystemExit('unexpected Python invocation')
""",
    )
    for name in ["params", "tokenizer", "base", "maxtext", "scratch"]:
        (tmp_path / name).mkdir()
    env = {
        **os.environ,
        "PY": str(python),
        "MAXTEXT_DIR": str(tmp_path / "maxtext"),
        "TMPDIR": str(tmp_path / "scratch"),
        "TRACE": str(trace),
    }
    script = os.path.relpath(SCRIPTS / "export_hf.sh", tmp_path)
    result = subprocess.run(
        ["bash", script, "params", "out", "base", "tokenizer"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(call[0].endswith("resize_vocab.py") for call in calls)
    assert (tmp_path / "out").is_dir()
    assert not list((tmp_path / "scratch").iterdir())
