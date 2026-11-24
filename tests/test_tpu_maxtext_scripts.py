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


@pytest.mark.parametrize("image", ["image", "image-olmo35-rf", "image-olmo35-fs", "image-olmo35-dr"])
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
        "image-olmo35-dr": "51f8cc4b-p907d86f3-rfb46e215e-bs3ec936c3-dc88c6c7ea-dr",
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


MM = SCRIPTS / "olmo35-mm"
MM_RUN = SCRIPTS / "k8s/olmo35-mm-sft-4x4x8"


@pytest.mark.parametrize("label", ["run", "e4", "gates", "small"])
@pytest.mark.parametrize("exists", [False, True])
def test_mm_image_build_pins_commits_and_never_rebuilds_a_tag(tmp_path, label, exists):
    pins = {
        line.split()[0]: line.split()[1:]
        for line in (SCRIPTS / "image-olmo35-mm/maxtext_source").read_text().splitlines()
    }
    repo, maxtext_ref, olmo_core_ref = pins[label]
    executable(
        tmp_path / "gcloud",
        """#!/bin/bash
case "$*" in
  *"images describe"*) exit "$DESCRIBE_RC" ;;
esac
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
        "DESCRIBE_RC": "0" if exists else "1",
    }
    result = subprocess.run(
        ["bash", str(SCRIPTS / "image-olmo35-mm/build.sh"), label], env=env, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stdout + result.stderr
    tag = f"us-docker.pkg.dev/ai2-olmo/olmo/maxtext:{maxtext_ref}-olmocore-{olmo_core_ref[:9]}"
    assert f"IMAGE={tag}" in result.stdout
    assert capture.exists() is not exists
    if not exists:
        config = yaml.safe_load(capture.read_text())
        assert config["images"] == [tag]
        assert "serviceAccount" not in config
        args = [step.get("args", []) for step in config["steps"]]
        assert ["clone", "--no-checkout", repo, "maxtext"] in args
        assert ["checkout", maxtext_ref] in args
        assert any(f"--build-arg=OLMO_CORE_REF={olmo_core_ref}" in a for a in args)
    assert not list(tmp_path.glob("maxtext-build.*"))


def test_mm_image_build_refuses_unknown_label(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPTS / "image-olmo35-mm/build.sh"), "nope"], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 2
    assert "unknown image" in result.stderr


def kustomize_literals(path, name="olmo35-mm-sft-env"):
    config = yaml.safe_load(path.read_text())
    literals = next(g["literals"] for g in config["configMapGenerator"] if g["name"] == name)
    return dict(item.split("=", 1) for item in literals)


def test_mm_gate_overlay_is_the_run_plus_the_gate_flags():
    run = kustomize_literals(MM_RUN / "kustomization.yaml")
    gate = kustomize_literals(SCRIPTS / "k8s/olmo35-mm-sft-4x4x8-gate/kustomization.yaml")
    assert run["RUN_NAME"] == gate["RUN_NAME"] == "CHANGE_ME"
    # The gate runs the production args with a larger MoE buffer (a dropless first step), plus the gate flags.
    assert "ragged_buffer_factor=3.0" in run["TRAIN_ARGS"]
    run_args = set(run["TRAIN_ARGS"].split()) - {"ragged_buffer_factor=3.0"}
    assert run_args <= set(gate["TRAIN_ARGS"].split())
    extra = set(gate["TRAIN_ARGS"].split()) - run_args
    assert extra == {
        "emo_min_document_expert_pool=512",
        "emo_max_document_expert_pool=512",
        "log_moe_dropped_tokens=true",
        "ragged_buffer_factor=5.0",
    }
    assert "save_checkpoint_on_completion=false" in gate["EXTRA_ARGS"].split()
    assert int(gate["STEPS"]) == 20 < int(gate["CHECKPOINT_PERIOD"])
    assert run["OLMO_MM_LOADER_WORLD"] == "32"


def mm_launcher_env(tmp_path, **overrides):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("WANDB", "OLMO", "TRAIN"))}
    env.update(kustomize_literals(MM_RUN / "kustomization.yaml"))
    env.update({"DRY_RUN": "1", "JOB_COMPLETION_INDEX": "0"}, **overrides)
    return env


def test_mm_launcher_refuses_placeholder_run_name(tmp_path):
    result = subprocess.run(
        ["bash", str(MM_RUN / "train_mm_sft.sh")],
        env=mm_launcher_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert "Set RUN_NAME" in result.stderr


def test_mm_launcher_dry_run_prints_the_train_command(tmp_path):
    result = subprocess.run(
        ["bash", str(MM_RUN / "train_mm_sft.sh")],
        env=mm_launcher_env(tmp_path, RUN_NAME="test-run"),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("DRY_RUN:"))
    assert "maxtext.trainers.pre_train.train /deps/src/maxtext/configs/olmo/olmo35_tiny_mm_sft_64k.yml" in line
    for arg in ["run_name=test-run", "steps=10962", "olmo_seed_token_rows=[]", "ragged_buffer_factor=3.0"]:
        assert f" {arg} " in line
    assert "OLMO_MM_LOADER_WORLD=32" in line


def test_mm_launcher_refuses_async_ragged_all_to_all(tmp_path):
    env = mm_launcher_env(tmp_path, RUN_NAME="test-run")
    env["LIBTPU_INIT_ARGS"] += " --xla_tpu_enable_async_ragged_all_to_all=true"
    result = subprocess.run(
        ["bash", str(MM_RUN / "train_mm_sft.sh")], env=env, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 1
    assert "must not set" in result.stderr


SMALL_RUN = SCRIPTS / "k8s/olmo35-small-mm-sft-8x8x8"
SMALL_ENV = "olmo35-small-mm-sft-env"


def small_literals(overlay="olmo35-small-mm-sft-8x8x8"):
    return kustomize_literals(SCRIPTS / f"k8s/{overlay}/kustomization.yaml", SMALL_ENV)


def test_small_mm_arms_train_the_same_text_rows():
    image, text = small_literals(), small_literals("olmo35-small-mm-sft-4x4x8-noimg")
    assert image["RUN_NAME"] == text["RUN_NAME"] == "CHANGE_ME"
    rows = 16
    assert int(image["OLMO_MM_SCHEDULE_STEPS"]) == int(image["STEPS"]) == 9632
    assert int(text["OLMO_MM_SCHEDULE_STEPS"]) == int(text["STEPS"]) == 8076
    assert int(text["OLMO_MM_IMAGE_ROWS"]) == 0 and text["OLMO_MM_CONFIG"] == ""
    text_rows = {arm["OLMO_MM_TEXT_ROWS"] for arm in (image, text)} | {
        str(9632 * rows - int(image["OLMO_MM_IMAGE_ROWS"])),
        str(8076 * rows),
    }
    assert text_rows == {"129216"}
    # The text arm overrides only its slice layout, schedule and data lines, plus EXTRA_ARGS to restore on every
    # replica (its data1 mesh has a single replica).
    assert "enable_single_replica_ckpt_restoring=false" in text["EXTRA_ARGS"].split()
    assert set(text) == {
        "EXTRA_ARGS",
        "RUN_NAME",
        "STEPS",
        "SHARD_ARGS",
        "TRAIN_ARGS",
        "OLMO_MM_CONFIG",
        "OLMO_MM_IMAGE_ROWS",
        "OLMO_MM_SCHEDULE_STEPS",
        "OLMO_MM_TEXT_ROWS",
    }
    assert "learning_rate_schedule_steps=8076" in text["TRAIN_ARGS"].split()
    shard = dict(arg.split("=") for arg in text["SHARD_ARGS"].split())
    assert (
        int(shard["ici_data_parallelism"]) * int(shard["ici_fsdp_parallelism"]) * int(shard["ici_expert_parallelism"])
        == 128
    )
    assert float(shard["per_device_batch_size"]) * 128 == rows


@pytest.mark.parametrize(
    "gate,run",
    [
        ("olmo35-small-mm-sft-8x8x8-gate", "olmo35-small-mm-sft-8x8x8"),
        ("olmo35-small-mm-sft-4x4x8-noimg-gate", "olmo35-small-mm-sft-4x4x8-noimg"),
    ],
)
def test_small_mm_gate_overlays_are_the_run_plus_the_gate_flags(gate, run):
    base = small_literals()
    run_env = {**base, **(small_literals(run) if run != "olmo35-small-mm-sft-8x8x8" else {})}
    gate_env = small_literals(gate)
    assert set(gate_env["TRAIN_ARGS"].split()) - set(run_env["TRAIN_ARGS"].split()) == {
        "emo_document_expert_pool_choices=[256]",
        "emo_document_expert_pool_probs=[1.0]",
    }
    assert set(run_env["TRAIN_ARGS"].split()) <= set(gate_env["TRAIN_ARGS"].split())
    restore = {"enable_single_replica_ckpt_restoring=false", "enable_single_replica_ckpt_restoring=true"}
    assert set(gate_env["EXTRA_ARGS"].split()) - set(base["EXTRA_ARGS"].split()) - restore == {
        "save_checkpoint_on_completion=false"
    }
    assert int(gate_env["STEPS"]) == 20 < int(gate_env["CHECKPOINT_PERIOD"])


def small_launcher_env(**overrides):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("WANDB", "OLMO", "TRAIN"))}
    env.update(small_literals())
    env.update(
        {
            "DRY_RUN": "1",
            "JOB_COMPLETION_INDEX": "0",
            "RUN_NAME": "test-run",
            "LOAD_PARAMETERS_PATH": "gs://bucket/checkpoints/4625/items",
        },
        **overrides,
    )
    return env


@pytest.mark.parametrize("name", ["LOAD_PARAMETERS_PATH", "OLMO_MM_CONFIG", "TRAIN_ARGS", "STEPS"])
def test_small_mm_launcher_refuses_placeholders(name):
    result = subprocess.run(
        ["bash", str(SMALL_RUN / "train_mm_sft.sh")],
        env=small_launcher_env(**{name: "x CHANGE_ME"}),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert f"Set {name}" in result.stderr


def test_small_mm_launcher_dry_run_prints_the_train_command():
    result = subprocess.run(
        ["bash", str(SMALL_RUN / "train_mm_sft.sh")],
        env=small_launcher_env(),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("DRY_RUN:"))
    assert "maxtext.trainers.pre_train.train /deps/src/maxtext/configs/olmo/olmo35_small_mm_sft_64k.yml" in line
    for arg in ["run_name=test-run", "steps=9632", "load_parameters_path=gs://bucket/checkpoints/4625/items"]:
        assert f" {arg} " in line
    assert (
        "olmo_mm_config_path=/deps/src/maxtext/configs/olmo/olmo_mm/olmo35_small_mmsft_images_s2v1_1p55b.json" in line
    )


@pytest.mark.parametrize(
    "script,args",
    [
        ("convert_b2e.sh", ["dcp", "out"]),
        ("export_to_olmocore.sh", ["gs://bucket/run/checkpoints/10961/items", "out", "config.json"]),
    ],
)
def test_mm_conversions_refuse_existing_output_and_dry_run(tmp_path, script, args):
    (tmp_path / "out").mkdir()
    result = subprocess.run(
        ["bash", str(MM / script), *args], cwd=tmp_path, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 2
    assert "nothing is overwritten" in result.stderr
    (tmp_path / "out").rmdir()
    result = subprocess.run(
        ["bash", str(MM / script), *args],
        cwd=tmp_path,
        env={**os.environ, "DRY_RUN": "1", "PY": "fake-python"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "out").exists()
    commands = [line for line in result.stdout.splitlines() if line.startswith("+ fake-python")]
    if script == "convert_b2e.sh":
        assert [c.split()[2].rsplit("/", 1)[1] for c in commands] == [
            "convert_b2e_to_maxtext.py",
            "verify_b2e_bitwise.py",
        ]
        assert "--save_overwrite false" in commands[0] and "--scan_layers true" in commands[0]
    else:
        assert "stage_items.sh gs://bucket/run/checkpoints/10961/items" in result.stdout
        assert [c.split()[2].rsplit("/", 1)[1] for c in commands] == ["fast_maxtext_loader.py", "verify_roundtrip.py"]
        assert "--mode maxtext_to_olmo_core" in commands[0] and "--save_overwrite false" in commands[0]


def write_gate_logs(directory, ce=1.0905, rbf=2.2, dropped=0, hashes=None, ranks=2, steps=3):
    hashes = hashes or {}
    lines = [f"I1120 train.py:1126] REQUIRED_RBF step=0 max={rbf} per_layer=[{rbf}] program=normal"]
    if dropped is not None:
        lines.append(f"I1120 train.py:1212] MOE-DROPPED step=0 dropped={dropped} routed=100 frac=0 per_layer=[0]")
    lines.append(f"I1120 train.py:1190] LOSS-PARTS step=1 ce={ce} moe_lb=0.24 z=0.0002 tokens=1936996")
    for step in range(1, steps + 1):
        for rank in range(ranks):
            record = {"step": step, "rank": rank, "hashes": hashes.get((step, rank), {"all": f"h{step}-{rank}"})}
            lines.append("I1120 olmo_mm_data_processing.py:530] MM-BATCH-HASH " + json.dumps(record))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pod-0.log").write_text("\n".join(lines) + "\n")
    # A second host that read the same ranks logs the same records.
    (directory / "pod-1.log").write_text("\n".join(lines[-ranks * steps :]) + "\n")


def write_gpu_hashes(directory, ranks=2, steps=3, step1_ce=1.0910, override=None):
    directory.mkdir(parents=True)
    for rank in range(ranks):
        with open(directory / f"rank{rank}.jsonl", "w") as f:
            for step in range(1, steps + 1):
                hashes = {"all": f"h{step}-{rank}", "input_ids": f"i{step}-{rank}"}
                if override and (step, rank) in override:
                    hashes = override[(step, rank)]
                f.write(json.dumps({"step": step, "rank": rank, "hashes": hashes}) + "\n")
    (directory / "step1_eval.json").write_text(json.dumps({"eval_mode_ce": step1_ce, "loss_weight": 1.0, "ranks": 2}))


def run_check_gates(tmp_path, *extra):
    module = load_script("olmo35-mm/check_gates.py")
    out = tmp_path / "gates.json"
    rc = module.main(["--tpu-logs", str(tmp_path / "tpu"), "--steps", "3", "--ranks", "2", "--out", str(out), *extra])
    return rc, json.loads(out.read_text())


def test_check_gates_passes_matching_runs(tmp_path):
    write_gate_logs(tmp_path / "tpu")
    write_gpu_hashes(tmp_path / "gpu")
    rc, summary = run_check_gates(
        tmp_path, "--gpu-hashes", str(tmp_path / "gpu"), "--gpu-step1", str(tmp_path / "gpu/step1_eval.json")
    )
    assert rc == 0, summary
    assert summary["step1_ce"]["abs_diff"] == pytest.approx(5e-4)
    assert summary["batch_hashes"]["equal"] == 6


@pytest.mark.parametrize(
    "logs,gpu,failing",
    [
        ({"ce": 1.0925}, {}, "step1_ce"),  # 1.5e-3 off
        ({"dropped": 7}, {}, "step1_ce"),  # first step dropped tokens
        ({"dropped": None, "rbf": 4.42}, {}, "step1_ce"),  # no MOE-DROPPED line: REQUIRED_RBF above the buffer
        ({}, {"override": {(2, 1): {"all": "other", "input_ids": "x"}}}, "batch_hashes"),
        ({"steps": 2}, {}, "batch_hashes"),  # step 3 missing on TPU
        ({"hashes": {(3, 0): {"all": "conflict"}}}, {}, "batch_hashes"),  # the hosts disagree
    ],
)
def test_check_gates_fails(tmp_path, logs, gpu, failing):
    write_gate_logs(tmp_path / "tpu", **logs)
    write_gpu_hashes(tmp_path / "gpu", **gpu)
    rc, summary = run_check_gates(
        tmp_path, "--gpu-hashes", str(tmp_path / "gpu"), "--gpu-step1", str(tmp_path / "gpu/step1_eval.json")
    )
    assert rc == 1
    assert summary[failing]["pass"] is False
    other = {"step1_ce", "batch_hashes"} - {failing}
    assert all(summary[name]["pass"] for name in other), summary


def test_check_gates_needs_the_gpu_hashes(tmp_path):
    write_gate_logs(tmp_path / "tpu")
    rc, summary = run_check_gates(tmp_path, "--reference-ce", "1.0910")
    assert rc == 1
    assert summary["step1_ce"]["pass"] and summary["batch_hashes"]["pass"]
