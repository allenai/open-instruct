"""Re-render prepared MILES prompts under a new start checkpoint's chat template.

Prepared rows record the SHA-256 of the chat template they were rendered with
(`metadata.rendering_template_sha256`). When the start checkpoint changes and its
template differs, data preparation stops with "Prepared rendering template hash
differs from the checkpoint chat template; re-render inputs". This tool renders
each row's `metadata.messages` with the checkpoint tokenizer's chat template
(`add_generation_prompt=True`, as source preparation does) and rewrites only:

- `input` (the rendered prompt),
- `metadata.rendering_template_sha256`,
- `metadata.prompt_tokens` and `metadata.prompt_token_ids`, when present.

Tool rows also use their saved declarations and validate the target protocol.
ToolRL labels and verifier targets are updated if the prompt reasoning prefix changes.
The output is written atomically to a new path, with a JSON report beside it.

    python -m scripts.miles.rerender_prompts INPUT.jsonl[.gz] CHECKPOINT OUTPUT.jsonl[.gz]
"""

import argparse
import collections
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

from jinja2 import TemplateError

from open_instruct import tokenizer_utils
from open_instruct.miles.datasets import prompt_rendering


class RerenderError(ValueError):
    pass


def _open(path, mode):
    return (
        gzip.open(path, mode + "t", encoding="utf-8")
        if str(path).endswith(".gz")
        else open(path, mode, encoding="utf-8")
    )


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(staged, output):
    """Move `staged` to `output` atomically, never replacing an existing file."""
    try:
        os.link(staged, output)
    except FileExistsError:
        raise RerenderError(f"{output} already exists; write to a new path") from None
    except OSError as error:
        # No atomic no-clobber publish without hard links; fail closed.
        raise RerenderError(
            f"Cannot publish {output} without risking an overwrite ({error.strerror}); "
            "write the output on a filesystem that supports hard links"
        ) from error


def _stage(output, suffix):
    descriptor, staged = tempfile.mkstemp(prefix=f".{output.name}.", suffix=suffix, dir=output.parent)
    os.close(descriptor)
    return Path(staged)


def template_sha256(tokenizer):
    """The hash MILES compares against `metadata.rendering_template_sha256`."""
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise RerenderError("The checkpoint tokenizer has no chat template")
    return hashlib.sha256(template.encode()).hexdigest()


def _render_prompt(row, metadata, tokenizer):
    label = row.get("label")
    bindings = metadata.get("verifiers", [])
    targets = [label, *(binding.get("target") for binding in bindings if isinstance(binding, dict))]
    tools = metadata.get("tools")
    env = metadata.get("env_config") or {}
    has_tool_target = any(
        isinstance(target, dict) and ("tools" in target or target.get("response_mode") == "tool_call")
        for target in targets
    )
    if not (tools or env or has_tool_target):
        return tokenizer.apply_chat_template(metadata["messages"], tokenize=False, add_generation_prompt=True)

    if (
        not isinstance(label, dict)
        or len(bindings) != 1
        or not isinstance(bindings[0], dict)
        or bindings[0].get("target") != label
    ):
        raise RerenderError("row has tool definitions without a matching verifier target")
    structured = label.get("response_mode") == "tool_call"
    if env and env != {"kind": "non_executing_function_answer", "format": label.get("tool_call_format")}:
        raise RerenderError("unsupported tool environment")
    if not structured:
        if "tool_format" not in label or ("calls" in label) == ("expected_action" in label):
            raise RerenderError("unsupported offline tool verifier target")
        if tools is None:
            tools = label.get("tools")
    text = prompt_rendering.render_tool_prompt(metadata["messages"], tools, label, tokenizer)
    if not structured and "calls" in label:
        prefix = prompt_rendering.reasoning_prefix(text)
        row["label"] = {**label, "think_prefix_in_prompt": prefix}
        bindings[0]["target"] = {**bindings[0]["target"], "think_prefix_in_prompt": prefix}
    return text


def rerender_row(row, tokenizer, new_hash):
    """Return (row, changed); raises RerenderError for rows that cannot be re-rendered here."""
    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or not metadata.get("messages"):
        raise RerenderError("row has no metadata.messages to render")
    text = _render_prompt(row, metadata, tokenizer)
    if not isinstance(text, str) or not text:
        raise RerenderError("renderer produced an empty or non-text prompt")
    changed = text != row.get("input")
    row["input"] = text
    metadata["rendering_template_sha256"] = new_hash
    # MILES recomputes these when it builds a run-specific cache.
    metadata.pop("run_prompt_tokens", None)
    metadata.pop("run_prompt_token_ids_sha256", None)
    if "prompt_tokens" in metadata or metadata.get("prompt_token_ids") is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if "prompt_tokens" in metadata:
            metadata["prompt_tokens"] = len(ids)
        if metadata.get("prompt_token_ids") is not None:
            metadata["prompt_token_ids"] = ids
    return row, changed


def rerender(source, output, tokenizer, *, examples=3):
    source, output = Path(source), Path(output)
    report_path = output.with_name(output.name + ".report.json")
    if source.resolve() == output.resolve():
        raise RerenderError("Write to a new path; prepared inputs are never rewritten in place")
    for path in (output, report_path):
        if os.path.lexists(path):
            raise RerenderError(f"{path} already exists; write to a new path")
    new_hash = template_sha256(tokenizer)
    counts = collections.Counter()
    old_hashes = collections.Counter()
    changed_examples = []
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _stage(output, ".tmp.gz" if output.name.endswith(".gz") else ".tmp")
    staged_report = _stage(output, ".report.tmp")
    try:
        with _open(source, "r") as reader, _open(temporary, "w") as writer:
            for index, line in enumerate(reader, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                metadata = row.get("metadata") if isinstance(row, dict) else None
                if not isinstance(metadata, dict):
                    raise RerenderError(f"{source}: row {index}: row has no metadata object")
                old_hashes[str(metadata.get("rendering_template_sha256"))] += 1
                old_input = row.get("input")
                try:
                    row, changed = rerender_row(row, tokenizer, new_hash)
                except (ValueError, TypeError, KeyError, TemplateError) as error:
                    raise RerenderError(f"{source}: row {index}: {error}") from error
                counts["changed" if changed else "identical"] += 1
                if changed and len(changed_examples) < examples:
                    changed_examples.append(
                        {"row": index, "old_tail": str(old_input)[-300:], "new_tail": row["input"][-300:]}
                    )
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        report = {
            "source": str(source),
            "source_sha256": _sha256(source),
            "output": str(output),
            "output_sha256": _sha256(temporary),
            "rendering_template_sha256": new_hash,
            "rows": counts["changed"] + counts["identical"],
            "inputs_changed": counts["changed"],
            "inputs_identical": counts["identical"],
            "previous_template_hashes": dict(old_hashes),
            "changed_examples": changed_examples,
        }
        # Write the report in full before publishing anything, so a write error
        # leaves nothing behind that would block a retry.
        staged_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        _publish(temporary, output)
        _publish(staged_report, report_path)
    finally:
        temporary.unlink(missing_ok=True)
        staged_report.unlink(missing_ok=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="Prepared prompt JSONL (optionally .gz)")
    parser.add_argument(
        "checkpoint", help="New start checkpoint (HF directory or hub id) whose tokenizer renders the prompts"
    )
    parser.add_argument(
        "output", help="New prepared JSONL path (.gz to compress); a .report.json is written beside it"
    )
    parser.add_argument("--revision", help="Hub revision of the checkpoint")
    args = parser.parse_args(argv)
    tokenizer = tokenizer_utils.load_tokenizer(args.checkpoint, revision=args.revision, trust_remote_code=True)
    report = rerender(args.source, args.output, tokenizer)
    print(json.dumps({k: v for k, v in report.items() if k != "changed_examples"}, indent=2))
    return report


if __name__ == "__main__":
    main()
