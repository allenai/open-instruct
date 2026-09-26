"""An HF input may carry a tokenizer/chat-template override, staged with its stop-token config."""

import json
from pathlib import Path

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.execution import workflow

EXAMPLE = Path(__file__).resolve().parents[2] / "configs/miles/examples/medium.toml"


def test_hf_input_accepts_a_template_override(tmp_path):
    template = tmp_path / "chat_template.jinja"
    template.write_text("{{ messages[0]['content'] }}<|im_start|>assistant\n<think>")
    spec = RunSpec.load(EXAMPLE, overrides=[f'model.hf_template="{template}"'])
    assert spec.model["hf_template"] == str(template)


def test_native_input_still_requires_the_template():
    with pytest.raises(InputError, match="hf_template is required"):
        RunSpec.load(EXAMPLE, overrides=['model.format="olmo_core"'])


def test_staging_copies_a_jinja_override_over_the_checkpoint_template(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "olmo3moe"}))
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "chat_template.jinja").write_text("OLD")
    (source / "tokenizer_config.json").write_text(json.dumps({"chat_template": "OLD"}))
    override = tmp_path / "new_template.jinja"
    override.write_text("NEW [CUTOFF_DATE]")
    spec = RunSpec.load(
        EXAMPLE,
        overrides=[f'model.source="{source}"', f'model.hf_template="{override}"', f'output.root="{tmp_path / "run"}"'],
    )
    warnings = []
    monkeypatch.setattr(workflow.logger, "warning", lambda *args: warnings.append(args[0] % args[1:]))
    staged = Path(workflow.prepare_model(spec))
    assert (staged / "chat_template.jinja").read_text() == "NEW [CUTOFF_DATE]"
    assert (staged / "model.safetensors").is_symlink()
    marker = json.loads((staged / "workflow-model.json").read_text())
    assert marker["identity"]["template"]["path"] == str(override)
    assert any("[CUTOFF_DATE]" in w for w in warnings)


@pytest.mark.parametrize(
    "template, message",
    [
        ("/weka/does/not/exist", "not found"),
        ("EMPTY_DIR", "no tokenizer assets"),
        ("notes.txt", "tokenizer directory or a .jinja"),
    ],
)
def test_bad_template_overrides_fail_as_input_errors(tmp_path, template, message):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weights")
    if template == "EMPTY_DIR":
        template = str(tmp_path / "empty")
        Path(template).mkdir()
    elif template == "notes.txt":
        template = str(tmp_path / "notes.txt")
        Path(template).write_text("not a template")
    spec = RunSpec.load(
        EXAMPLE,
        overrides=[f'model.source="{source}"', f'model.hf_template="{template}"', f'output.root="{tmp_path / "run"}"'],
    )
    with pytest.raises(InputError, match=message):
        workflow.prepare_model(spec)
