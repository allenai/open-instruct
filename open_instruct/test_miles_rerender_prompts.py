"""Re-rendered prompts must pass MILES's template check and keep every other field."""

import gzip
import json
from pathlib import Path

import pytest
from scripts.miles import rerender_prompts
from tokenizers import Tokenizer as FastTokenizer
from tokenizers import models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from open_instruct import toolrl_utils
from open_instruct.miles.datasets import run_data
from open_instruct.miles.errors import InputError

OLD_TEMPLATE = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
NEW_TEMPLATE = "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}{% if add_generation_prompt %}[assistant] <think>{% endif %}"


def checkpoint(tmp_path, template):
    vocab = {"[UNK]": 0, **{chr(c): i + 1 for i, c in enumerate(range(32, 127))}, "\n": 96}
    backend = FastTokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = template
    path = tmp_path / ("new" if template == NEW_TEMPLATE else "old")
    tokenizer.save_pretrained(path)
    return path


def rows(old):
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2+2?"}]
    text = old.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    base = {
        "input": text,
        "label": {"value": "4"},
        "metadata": {
            "sample_id": "s0",
            "messages": messages,
            "prompt_tokens": len(old.encode(text, add_special_tokens=False)),
            "rendering_template_sha256": rerender_prompts.template_sha256(old),
            "verifiers": [{"name": "wb_0", "target": {"value": "4"}}],
            "provenance": {"source": "fixture"},
        },
    }
    with_ids = json.loads(json.dumps(base))
    with_ids["metadata"]["sample_id"] = "s1"
    with_ids["metadata"]["prompt_token_ids"] = old.encode(text, add_special_tokens=False)
    return [base, with_ids]


def write(path, items):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item) + "\n")


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def test_rerendered_rows_pass_the_checkpoint_template_check(tmp_path):
    old = run_data._tokenizer(str(checkpoint(tmp_path, OLD_TEMPLATE)))
    new_path = checkpoint(tmp_path, NEW_TEMPLATE)
    new = run_data._tokenizer(str(new_path))
    source, output = tmp_path / "in.jsonl.gz", tmp_path / "out" / "in-new.jsonl.gz"
    original = rows(old)
    write(source, original)
    # The old rows are what MILES rejects for the new checkpoint.
    with pytest.raises(InputError, match="re-render inputs"):
        run_data._verify_rendering_template(original[0]["metadata"], new)

    report = rerender_prompts.main([str(source), str(new_path), str(output)])

    result = read(output)
    assert report["rows"] == 2 and report["inputs_changed"] == 2
    assert report["rendering_template_sha256"] == rerender_prompts.template_sha256(new)
    assert json.loads((tmp_path / "out" / "in-new.jsonl.gz.report.json").read_text())["output_sha256"]
    for before, after in zip(original, result, strict=True):
        run_data._verify_rendering_template(after["metadata"], new)
        expected = new.apply_chat_template(before["metadata"]["messages"], tokenize=False, add_generation_prompt=True)
        assert after["input"] == expected and expected.endswith("[assistant] <think>")
        ids = new.encode(expected, add_special_tokens=False)
        assert after["metadata"]["prompt_tokens"] == len(ids)
        if "prompt_token_ids" in before["metadata"]:
            assert after["metadata"]["prompt_token_ids"] == ids
        # Everything else is preserved.
        for key in ("input", "rendering_template_sha256", "prompt_tokens", "prompt_token_ids"):
            before["metadata"].pop(key, None)
            after["metadata"].pop(key, None)
        before.pop("input")
        after.pop("input")
        assert after == before


@pytest.mark.parametrize("kind", ["structured", "toolrl"])
def test_rerenders_tool_rows_with_their_native_schemas(tmp_path, kind):
    template = (
        "{% for tool in tools %}{{ tool.function.name }}{% endfor %}"
        "{% for m in messages %}{{ m.content }}{% if m.tool_calls %}"
        "<tool_call><function=lookup><parameter=n>3</parameter></function></tool_call>"
        "{% endif %}{% endfor %}{% if add_generation_prompt %}<think>{% endif %}"
    )
    tokenizer = run_data._tokenizer(str(checkpoint(tmp_path, template)))
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}}},
        }
    ]
    if kind == "structured":
        label = {"response_mode": "tool_call", "tool_call_format": "olmo_xml_v1", "tool_name": "lookup"}
    else:
        label = {
            "format": "olmo-toolrl-v1",
            "tools": tools,
            "calls": [{"name": "lookup", "parameters": {"n": 3}}],
            "tool_format": "olmo_xml_v1",
            "think_prefix_in_prompt": False,
        }
    row = {
        "input": "old",
        "label": label,
        "metadata": {
            "messages": [{"role": "user", "content": "Find number 3."}],
            "tools": tools,
            "verifiers": [{"target": label, "name": kind}],
        },
    }
    result, changed = rerender_prompts.rerender_row(row, tokenizer, rerender_prompts.template_sha256(tokenizer))
    assert changed and result["input"] == "lookupFind number 3.<think>"
    assert result["metadata"]["tools"] == tools
    if kind == "toolrl":
        expected = {**label, "think_prefix_in_prompt": True}
        action = "</think><tool_call><function=lookup><parameter=n>3</parameter></function></tool_call>"
        assert toolrl_utils.score_response(action, result["label"], think_prefix_in_prompt=True).score == 4
    else:
        expected = label
    assert result["label"] == result["metadata"]["verifiers"][0]["target"] == expected

    tokenizer.chat_template = NEW_TEMPLATE
    source = tmp_path / "in.jsonl.gz"
    write(source, [row])
    output = tmp_path / "incompatible.jsonl.gz"
    with pytest.raises(rerender_prompts.RerenderError, match="format does not match"):
        rerender_prompts.rerender(source, output, tokenizer)
    assert not output.exists()


def test_rejects_rows_it_cannot_rerender_and_in_place_writes(tmp_path):
    old = run_data._tokenizer(str(checkpoint(tmp_path, OLD_TEMPLATE)))
    new = run_data._tokenizer(str(checkpoint(tmp_path, NEW_TEMPLATE)))
    source = tmp_path / "in.jsonl.gz"
    write(source, rows(old))
    with pytest.raises(rerender_prompts.RerenderError, match="new path"):
        rerender_prompts.rerender(source, source, new)
    tool_row = rows(old)[0]
    tool_row["metadata"]["verifiers"][0]["target"] = {"tools": [], "tool_format": "olmo_xml_v1"}
    no_messages = rows(old)[0]
    del no_messages["metadata"]["messages"]
    for bad, match in ((tool_row, "tool definitions"), (no_messages, "no metadata.messages")):
        write(source, [bad])
        output = tmp_path / "bad.jsonl.gz"
        with pytest.raises(rerender_prompts.RerenderError, match=match):
            rerender_prompts.rerender(source, output, new)
        assert not output.exists() and not list(tmp_path.glob(".bad.jsonl.gz.*"))


def test_never_overwrites_existing_output_or_report(tmp_path):
    old = run_data._tokenizer(str(checkpoint(tmp_path, OLD_TEMPLATE)))
    new = run_data._tokenizer(str(checkpoint(tmp_path, NEW_TEMPLATE)))
    source = tmp_path / "in.jsonl.gz"
    write(source, rows(old))
    source_bytes = source.read_bytes()

    existing = tmp_path / "other.jsonl.gz"
    existing.write_bytes(b"another prepared dataset")
    with pytest.raises(rerender_prompts.RerenderError, match="already exists"):
        rerender_prompts.rerender(source, existing, new)
    assert existing.read_bytes() == b"another prepared dataset"

    # A report path that is a symlink to the source must not be followed.
    output = tmp_path / "out.jsonl.gz"
    report = tmp_path / "out.jsonl.gz.report.json"
    report.symlink_to(source)
    with pytest.raises(rerender_prompts.RerenderError, match="already exists"):
        rerender_prompts.rerender(source, output, new)
    assert source.read_bytes() == source_bytes and not output.exists()
    assert not list(tmp_path.glob(".out.jsonl.gz.*")) and not list(tmp_path.glob(".other.jsonl.gz.*"))


def test_null_metadata_reports_source_and_row(tmp_path):
    new = run_data._tokenizer(str(checkpoint(tmp_path, NEW_TEMPLATE)))
    source = tmp_path / "in.jsonl.gz"
    write(source, [{"input": "x", "metadata": None}])
    with pytest.raises(rerender_prompts.RerenderError, match="row 1: row has no metadata object"):
        rerender_prompts.rerender(source, tmp_path / "out.jsonl.gz", new)


def test_fails_closed_without_hard_links(tmp_path, monkeypatch):
    old = run_data._tokenizer(str(checkpoint(tmp_path, OLD_TEMPLATE)))
    new = run_data._tokenizer(str(checkpoint(tmp_path, NEW_TEMPLATE)))
    source = tmp_path / "in.jsonl.gz"
    write(source, rows(old))
    output = tmp_path / "out.jsonl.gz"

    def no_links(src, dst):
        # Another invocation publishes first; no fallback may replace it.
        Path(dst).write_bytes(b"concurrent output")
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(rerender_prompts.os, "link", no_links)
    with pytest.raises(rerender_prompts.RerenderError, match="hard links"):
        rerender_prompts.rerender(source, output, new)
    assert output.read_bytes() == b"concurrent output"
    assert not (tmp_path / "out.jsonl.gz.report.json").exists()
    assert not list(tmp_path.glob(".out.jsonl.gz.*"))
