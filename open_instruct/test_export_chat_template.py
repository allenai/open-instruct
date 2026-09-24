"""CPU-only export tests: template behavior must not change tokenizer semantics."""

import hashlib
import json
import os
import pathlib
import runpy
import subprocess
import sys
import types
from unittest import mock

import pytest
from jinja2 import TemplateSyntaxError
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from torch.distributed.checkpoint import state_dict as dist_cp_sd
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from open_instruct import export_chat_template

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "scripts/tokenizers/templates/olmo_3_2_think_dev.jinja"
CONVERTER = ROOT / "scripts/train/debug/convert_moe_checkpoint_to_hf.py"
LAUNCHER = ROOT / "scripts/train/debug/oc_sft_olmoe3_kda_think.sh"
OLD_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "checkpoint"
    backend = Tokenizer(
        models.WordLevel({"[UNK]": 0, "<eos>": 1, "<pad>": 2, "<bos>": 3, "hello": 4}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 3)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", chat_template=OLD_TEMPLATE
    )
    tokenizer.save_pretrained(path)
    (path / "model.safetensors").write_bytes(b"weights must not be touched")
    (path / "generation_config.json").write_text('{"eos_token_id": 1}')
    return path


def snapshot(path):
    return {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}


@pytest.mark.parametrize("embedded", [False, True])
def test_install_preserves_tokenizer_and_renders_thinking(checkpoint, embedded, tmp_path):
    config_path = checkpoint / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    if embedded:
        config["chat_template"] = OLD_TEMPLATE
        config_path.write_text(json.dumps(config))
    before = snapshot(checkpoint)
    original = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    template = export_chat_template.read_export_chat_template(TEMPLATE_PATH)
    export_chat_template.install_export_chat_template(checkpoint, template)

    after = snapshot(checkpoint)
    allowed = {"chat_template.jinja", "tokenizer_config.json"} if embedded else {"chat_template.jinja"}
    assert {name for name in after if after[name] != before.get(name)} == allowed
    actual_config = json.loads(config_path.read_text())
    assert actual_config == (config | {"chat_template": template} if embedded else config)
    restored = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    assert restored.chat_template == template
    assert restored.get_vocab() == original.get_vocab()
    assert restored.all_special_ids == original.all_special_ids
    assert restored.backend_tokenizer.to_str() == original.backend_tokenizer.to_str()
    for text in ["hello", "<think>\nOkay", "12345 café 中文\n\n"]:
        assert restored.encode(text) == original.encode(text)
    messages = [{"role": "user", "content": "hello"}]
    rendered = restored.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert rendered.endswith("<|im_start|>assistant\n<think>")
    assert not restored.apply_chat_template(messages, tokenize=False, add_generation_prompt=False).endswith("<think>")
    # RL's save_pretrained must carry the template forward.
    next_export = tmp_path / "rl-checkpoint"
    restored.save_pretrained(next_export)
    assert AutoTokenizer.from_pretrained(next_export).chat_template == template


def test_pinned_template_provenance():
    assert hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest() == (
        "43b0c225dd327d4af450809bfb3abfbd828eb19d6546d3e9ae33782464874d6a"
    )


def test_no_override_is_byte_identical(checkpoint):
    before = snapshot(checkpoint)
    export_chat_template.install_export_chat_template(checkpoint, export_chat_template.read_export_chat_template(None))
    assert snapshot(checkpoint) == before


@pytest.mark.parametrize("link", [os.link, os.symlink])
def test_linked_metadata_does_not_modify_source(checkpoint, tmp_path, link):
    config_path = checkpoint / "tokenizer_config.json"
    config = json.loads(config_path.read_text()) | {"chat_template": OLD_TEMPLATE}
    config_path.write_text(json.dumps(config))
    before = snapshot(checkpoint)
    sibling = tmp_path / "rl"
    sibling.mkdir()
    for path in checkpoint.iterdir():
        link(path, sibling / path.name)
    export_chat_template.install_export_chat_template(sibling, TEMPLATE_PATH.read_text())
    assert snapshot(checkpoint) == before
    assert AutoTokenizer.from_pretrained(sibling).chat_template == TEMPLATE_PATH.read_text()


def test_invalid_metadata_fails_before_writing(checkpoint):
    (checkpoint / "tokenizer_config.json").write_text("not JSON")
    before = snapshot(checkpoint)
    with pytest.raises(ValueError):
        export_chat_template.install_export_chat_template(checkpoint, TEMPLATE_PATH.read_text())
    assert snapshot(checkpoint) == before


def test_named_templates_rejected(checkpoint):
    additional = checkpoint / "additional_chat_templates"
    additional.mkdir()
    (additional / "tool_use.jinja").write_text(OLD_TEMPLATE)
    before = snapshot(checkpoint)
    with pytest.raises(ValueError, match="single template"):
        export_chat_template.install_export_chat_template(checkpoint, TEMPLATE_PATH.read_text())
    assert snapshot(checkpoint) == before


@pytest.mark.parametrize("embedded", [{"default": OLD_TEMPLATE}, [{"name": "default", "template": OLD_TEMPLATE}]])
def test_embedded_named_templates_rejected(checkpoint, embedded):
    config_path = checkpoint / "tokenizer_config.json"
    config_path.write_text(json.dumps(json.loads(config_path.read_text()) | {"chat_template": embedded}))
    before = snapshot(checkpoint)
    with pytest.raises(ValueError, match="embedded named templates"):
        export_chat_template.install_export_chat_template(checkpoint, TEMPLATE_PATH.read_text())
    assert snapshot(checkpoint) == before


def test_empty_template_rejected(checkpoint, tmp_path):
    path = tmp_path / "empty.jinja"
    path.write_text(" \n")
    before = snapshot(checkpoint)
    with pytest.raises(ValueError, match="empty"):
        export_chat_template.read_export_chat_template(path)
    with pytest.raises(ValueError, match="empty"):
        export_chat_template.install_export_chat_template(checkpoint, path.read_text())
    assert snapshot(checkpoint) == before


def test_invalid_template_rejected_before_writes(checkpoint, tmp_path):
    path = tmp_path / "invalid.jinja"
    path.write_text("{% if %} ")
    before = snapshot(checkpoint)
    with pytest.raises(TemplateSyntaxError):
        export_chat_template.read_export_chat_template(path)
    with pytest.raises(TemplateSyntaxError):
        export_chat_template.install_export_chat_template(checkpoint, path.read_text())
    assert snapshot(checkpoint) == before


def test_generation_block_template_supported(checkpoint, tmp_path):
    path = tmp_path / "generation.jinja"
    path.write_text("{% generation %}{{ messages[0]['content'] }}{{ eos_token }}{% endgeneration %}")
    export_chat_template.install_export_chat_template(checkpoint, export_chat_template.read_export_chat_template(path))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    assert tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False) == "hello<eos>"


def test_special_token_snapshot_includes_roles_and_ids(checkpoint):
    state = export_chat_template.read_special_token_state(checkpoint)
    assert state["bos_token"] == ("<bos>", 3)
    assert state["eos_token"] == ("<eos>", 1)
    assert state["pad_token"] == ("<pad>", 2)


def test_existing_export_cli(checkpoint):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "open_instruct.export_chat_template",
            "--checkpoint-dir",
            str(checkpoint),
            "--export-chat-template",
            str(TEMPLATE_PATH),
        ],
        cwd=ROOT,
        check=True,
    )
    assert AutoTokenizer.from_pretrained(checkpoint).chat_template == TEMPLATE_PATH.read_text()


@pytest.mark.parametrize(
    "override",
    [False, True, "missing", "invalid", "hub-tokenizer", "backend-mismatch", "eos_token", "pad_token", "bos_token"],
)
def test_converter_integration(checkpoint, tmp_path, override):
    # The heavyweight converter is mocked; exercise the real CLI and post-export
    # code against an actual tokenizer without importing GPU-only OLMo-core.
    modules = {
        "olmo_core.config": types.SimpleNamespace(DType=types.SimpleNamespace(bfloat16="bf16")),
        "olmo_core.distributed.checkpoint": types.SimpleNamespace(
            get_checkpoint_metadata=mock.Mock(), load_keys=mock.Mock()
        ),
        "olmo_core.nn.hf": types.SimpleNamespace(convert_checkpoint_to_hf=mock.Mock()),
        "olmo_core.nn.transformer.config": types.SimpleNamespace(TransformerConfig=mock.Mock()),
        "olmo_core.utils": types.SimpleNamespace(prepare_cli_environment=mock.Mock()),
    }
    with mock.patch.dict(sys.modules, modules):
        namespace = runpy.run_path(str(CONVERTER))
    # Imported before patching sys.modules, so restoring it cannot evict torch
    # distributed modules and cause duplicate registrations in later tests.
    assert namespace["dist_cp_sd"] is dist_cp_sd
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": {}, "dataset": {"tokenizer": {}}}))
    argv = [str(CONVERTER), "-i", "/unused", "-o", str(checkpoint), "-c", str(config), "-t", str(checkpoint)]
    if override:
        argv += ["--export-chat-template", str(tmp_path / "missing.jinja" if override == "missing" else TEMPLATE_PATH)]
    if override == "invalid":
        invalid = tmp_path / "invalid.jinja"
        invalid.write_text("{% if %} ")
        argv[-1] = str(invalid)
    if override == "hub-tokenizer":
        argv[argv.index("-t") + 1] = "allenai/olmo-3-tokenizer-instruct-dev"
    before = snapshot(checkpoint)
    convert = modules["olmo_core.nn.hf"].convert_checkpoint_to_hf

    def rebuild_backend(**kwargs):
        # Model the dependency loading another tokenizer or reconstructing its
        # backend. The reference must have been snapshotted before this write.
        tokenizer_path = checkpoint / "tokenizer.json"
        backend = json.loads(tokenizer_path.read_text())
        backend["pre_tokenizer"] = {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        }
        tokenizer_path.write_text(json.dumps(backend))

    if override == "backend-mismatch":
        convert.side_effect = rebuild_backend

    def change_special_token(**kwargs):
        config_path = checkpoint / "tokenizer_config.json"
        metadata = json.loads(config_path.read_text())
        metadata[override] = "<pad>" if override != "pad_token" else "<eos>"
        config_path.write_text(json.dumps(metadata))

    special_mismatch = override in ("eos_token", "pad_token", "bos_token")
    if special_mismatch:
        convert.side_effect = change_special_token
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.dict(namespace["main"].__globals__, load_ddp_main_params=mock.Mock(return_value=None)),
    ):
        if override == "missing":
            with pytest.raises(FileNotFoundError):
                namespace["main"]()
            convert.assert_not_called()
        elif override == "invalid":
            with pytest.raises(TemplateSyntaxError):
                namespace["main"]()
            convert.assert_not_called()
        elif override == "hub-tokenizer":
            with pytest.raises(SystemExit, match="2"):
                namespace["main"]()
            convert.assert_not_called()
        elif override == "backend-mismatch":
            with pytest.raises(RuntimeError, match="differs from the saved training tokenizer"):
                namespace["main"]()
            convert.assert_called_once()
        elif special_mismatch:
            with pytest.raises(RuntimeError, match="special tokens or IDs differ"):
                namespace["main"]()
            convert.assert_called_once()
            assert (checkpoint / "tokenizer.json").read_bytes() == before["tokenizer.json"]
        else:
            namespace["main"]()
            convert.assert_called_once()
            assert convert.call_args.kwargs["tokenizer_id"] == str(checkpoint)
    if override is True:
        assert AutoTokenizer.from_pretrained(checkpoint).chat_template == TEMPLATE_PATH.read_text()
        assert (checkpoint / "tokenizer.json").read_bytes() == before["tokenizer.json"]
    elif override == "backend-mismatch" or special_mismatch:
        assert (checkpoint / "chat_template.jinja").read_bytes() == before["chat_template.jinja"]
    else:
        assert snapshot(checkpoint) == before


@pytest.mark.parametrize("mode", ["convert", "convert_rl", "convert_override"])
@pytest.mark.parametrize(
    "timeouts, expected",
    [({}, "2h"), ({"JOB_TIMEOUT": "3h"}, "3h"), ({"JOB_TIMEOUT": "3h", "CONVERT_TIMEOUT": "4h"}, "4h")],
)
def test_launcher_selects_template_only_for_rl(tmp_path, mode, timeouts, expected):
    # Substitute a local argv recorder for Python; never invoke mason/Beaker.
    recorder = tmp_path / "record"
    recorder.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$ARGV_OUTPUT"\n')
    recorder.chmod(0o755)
    output = tmp_path / "args"
    env = {key: value for key, value in os.environ.items() if not key.startswith(("EXPORT_", "CONVERT_"))}
    env.pop("JOB_TIMEOUT", None)
    env.update(timeouts)
    env.update(PY=str(recorder), CKPT_ROOT="/checkpoint", STEP="step42", ARGV_OUTPUT=str(output))
    if mode != "convert":
        env["EXPORT_TOKENIZER"] = "/training/tokenizer"
    if mode == "convert_override":
        env["EXPORT_CHAT_TEMPLATE"] = str(TEMPLATE_PATH.relative_to(ROOT))
    result = subprocess.run(
        ["bash", str(LAUNCHER), "unused-image", "convert" if mode == "convert_override" else mode],
        env=env,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    args = output.read_text().splitlines()
    assert args[args.index("--timeout") + 1] == expected
    assert args[args.index("-o") + 1] == "/checkpoint/hf_step42" + ("-think" if mode == "convert_rl" else "")
    if mode != "convert":
        assert args[args.index("--export-chat-template") + 1] == str(TEMPLATE_PATH.relative_to(ROOT))
        assert args[args.index("--tokenizer") + 1] == "/training/tokenizer"
        assert f"template: {TEMPLATE_PATH.relative_to(ROOT)}" in result.stdout
    else:
        assert "--export-chat-template" not in args
        assert "--tokenizer" not in args
        assert "template: unchanged" in result.stdout
