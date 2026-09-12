"""Researcher workflow ownership, source immutability and resume boundaries."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct.miles import run_data, workflow
from open_instruct.miles.errors import InputError


class Spec:
    def __init__(self, root, source, template=None):
        self.model = {"source": str(source), "format": "hf"}
        if template:
            self.model["hf_template"] = str(template)
        self.conversion = {"hf_output": str(root / "prepared" / "hf")}
        self.output = {"root": str(root), "export_hf": True, "hf_dir": str(root / "export")}
        self.launch = {"auto_resume": True}
        self.data = {"tasks": [{"task": "multiplication", "train_count": 1}]}

    def to_dict(self):
        return {key: getattr(self, key) for key in ("model", "conversion", "output", "launch", "data")}

    def plan(self):
        return self.to_dict()

    def compile(self, prepared=None):
        values = {"save": str(Path(self.output["root"]) / "checkpoints"), "seed": 17, "rollout_max_prompt_len": 2048}
        values.update(prepared or {})
        return SimpleNamespace(
            miles=values, core=SimpleNamespace(max_sequence_length=8192), plan=lambda: {"miles": values}
        )


@pytest.fixture
def spec(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"toy"}')
    (source / "model.safetensors").write_bytes(b"weight fixture")
    (source / "tokenizer.json").write_text('{"original":true}')
    (source / "tokenizer_config.json").write_text('{"chat_template":"original"}')
    (source / "chat_template.jinja").write_text("original")
    nested = source / "chat_templates"
    nested.mkdir()
    (nested / "tool.jinja").write_text("source tool template")
    return Spec(tmp_path / "run", source)


def test_template_directory_never_writes_through_source_links(spec, tmp_path, monkeypatch):
    template = tmp_path / "template"
    template.mkdir()
    (template / "tokenizer_config.json").write_text('{"chat_template":"replacement"}')
    spec.model["hf_template"] = str(template)
    source = Path(spec.model["source"])
    before = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}

    class Tokenizer:
        def save_pretrained(self, staging):
            (staging / "tokenizer.json").write_text("replacement vocab")
            (staging / "tokenizer_config.json").write_text("replacement config")
            (staging / "chat_template.jinja").write_text("replacement template")
            (staging / "chat_templates").mkdir()
            (staging / "chat_templates/tool.jinja").write_text("replacement tool template")

    monkeypatch.setattr(
        workflow.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())
        ),
    )
    target = Path(workflow.prepare_model(spec))
    after = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert before == after
    assert (target / "model.safetensors").is_symlink()
    assert not (target / "tokenizer.json").is_symlink()
    assert not (target / "chat_templates").is_symlink()
    assert (target / "tokenizer.json").read_text() == "replacement vocab"
    assert workflow.prepare_model(spec) == str(target)


def test_file_template_replaces_only_prepared_copy(spec, tmp_path):
    template = tmp_path / "replacement.jinja"
    template.write_text("replacement")
    spec.model["hf_template"] = str(template)
    target = Path(workflow.prepare_model(spec))
    assert (target / "chat_template.jinja").read_text() == "replacement"
    assert (Path(spec.model["source"]) / "chat_template.jinja").read_text() == "original"


@pytest.mark.parametrize("location", ["source", "prepared"])
def test_prepared_model_reuse_rejects_metadata_changes(spec, location):
    target = Path(workflow.prepare_model(spec))
    chosen = Path(spec.model["source"]) if location == "source" else target
    (chosen / "tokenizer_config.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        workflow.prepare_model(spec)


def test_incomplete_model_is_not_adopted(spec):
    Path(spec.conversion["hf_output"]).mkdir(parents=True)
    with pytest.raises(InputError, match="incomplete"):
        workflow.prepare_model(spec)


def test_run_directory_exclusive_and_failure_recorded(spec):
    with pytest.raises(RuntimeError, match="deliberate"), workflow.run_directory(spec):
        with pytest.raises(InputError, match="Another process"), workflow.run_directory(spec):
            pytest.fail("must not acquire the same lock twice")
        raise RuntimeError("deliberate")
    root = Path(spec.output["root"])
    state = json.loads((root / "workflow.json").read_text())
    assert state["status"] == "failed" and "deliberate" in state["error"]
    with workflow.run_directory(spec) as (_, retry):
        assert retry["status"] == "preparing"


def test_changed_run_spec_cannot_resume(spec):
    with workflow.run_directory(spec):
        pass
    spec.data["tasks"][0]["train_count"] = 2
    with pytest.raises(ValueError, match="configuration changed"), workflow.run_directory(spec):
        pytest.fail("changed spec accepted")


def test_execute_failure_then_resume_uses_checkpoint_and_same_preparation(spec, monkeypatch):
    root = Path(spec.output["root"])
    preparation_calls = []
    training_calls = []

    def prepare_data(data, hf_checkpoint, output, **kwargs):
        preparation_calls.append((data.copy(), hf_checkpoint, output, kwargs))
        output.mkdir(exist_ok=True)
        return {
            "prompt_data": str(output / "train.jsonl"),
            "eval_prompt_data": [],
            "reward_config": str(output / "verifiers.json"),
            "manifest": str(output / "manifest.json"),
        }

    def train(config, *, export_hf):
        training_calls.append((dict(config.miles), export_hf))
        if len(training_calls) == 1:
            checkpoint = Path(config.miles["save"])
            checkpoint.mkdir()
            (checkpoint / "core-latest.json").write_text("{}")
            raise RuntimeError("interrupted after durable save")
        return {"updates": 2}

    monkeypatch.setattr(run_data, "prepare_data", prepare_data)
    monkeypatch.setattr(workflow, "train_config", train)
    with pytest.raises(RuntimeError, match="interrupted"):
        workflow.execute(spec)
    assert json.loads((root / "workflow.json").read_text())["status"] == "failed"
    result = workflow.execute(spec)
    assert result["status"] == "complete"
    assert preparation_calls[0] == preparation_calls[1]
    assert "load" not in training_calls[0][0]
    assert training_calls[1][0]["load"] == str(root / "checkpoints")
    assert training_calls[1][1] == spec.output["hf_dir"]
    with pytest.raises(InputError, match="already completed"):
        workflow.execute(spec)


def test_auto_resume_false_blocks_retry(spec):
    spec.launch["auto_resume"] = False
    with workflow.run_directory(spec):
        pass
    with pytest.raises(InputError, match="auto_resume is false"), workflow.run_directory(spec):
        pytest.fail("retry accepted")


def test_template_directory_removes_stale_standalone_template(spec, tmp_path, monkeypatch):
    template = tmp_path / "template"
    template.mkdir()
    (template / "tokenizer_config.json").write_text('{"chat_template":"replacement"}')
    spec.model["hf_template"] = str(template)

    class Tokenizer:
        def save_pretrained(self, staging):
            (staging / "tokenizer_config.json").write_text('{"chat_template":"replacement"}')

    monkeypatch.setattr(
        workflow.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())
        ),
    )
    target = Path(workflow.prepare_model(spec))
    assert not (target / "chat_template.jinja").exists()
    assert not (target / "chat_templates").exists()
    assert (Path(spec.model["source"]) / "chat_template.jinja").read_text() == "original"


def test_nested_model_destination_cannot_write_source(spec):
    spec.conversion["hf_output"] = str(Path(spec.model["source"]) / "prepared")
    with pytest.raises(ValueError, match="read-only source"):
        workflow.prepare_model(spec)
    assert not list(Path(spec.model["source"]).glob(".*preparing*"))


def test_execute_uses_real_immutable_data_under_separate_prepared_directory(spec, monkeypatch):
    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, **kwargs):
            return messages[-1]["content"] + "|assistant:"

        def encode(self, text, **kwargs):
            return list(text.encode())

    monkeypatch.setattr(run_data, "_tokenizer", lambda path: Tokenizer())
    calls = []

    def train(config, **kwargs):
        data_path = Path(config.miles["prompt_data"])
        assert data_path == Path(spec.output["root"]) / "prepared/data/train.jsonl"
        calls.append(data_path.read_bytes())
        if len(calls) == 1:
            raise RuntimeError("failure before checkpoint")
        return None

    monkeypatch.setattr(workflow, "train_config", train)
    with pytest.raises(RuntimeError, match="before checkpoint"):
        workflow.execute(spec)
    assert workflow.execute(spec)["status"] == "complete"
    assert calls[0] == calls[1]


def test_native_converter_uses_saved_geometry_without_forward_claim(spec, tmp_path, monkeypatch):
    saved = {"model": {"width": 1024}, "dataset": {"tokenizer": {"identifier": "fixture"}}}
    calls = []

    def convert(source, target, model, tokenizer, **kwargs):
        calls.append((source, target, model.copy(), tokenizer.copy(), kwargs))
        model["width"] = 1
        tokenizer["identifier"] = "changed"

    modules = {
        "olmo_core.nn.hf.convert_checkpoint": SimpleNamespace(
            load_config=lambda path: saved, convert_checkpoint_to_hf=convert
        ),
        "olmo_core.config": SimpleNamespace(DType=lambda name: name),
        "torch": SimpleNamespace(device=lambda name: name),
    }
    monkeypatch.setattr(workflow.importlib, "import_module", modules.__getitem__)
    target = tmp_path / "native-export"
    workflow._convert_native(spec, target)
    assert saved["model"]["width"] == 1024
    assert saved["dataset"]["tokenizer"]["identifier"] == "fixture"
    assert calls[0][-1]["dtype"] == "bfloat16"
    assert calls[0][-1]["device"] == "cpu"
    assert calls[0][-1]["validate"] is False
