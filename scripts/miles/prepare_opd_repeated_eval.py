"""Freeze the existing four-arm AIME evaluation panel and checkpoint identities."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(config, output):
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
    # Read the literal template without importing unrelated training/submission dependencies.
    source = Path(__file__).resolve().parents[2] / "open_instruct/dataset_transformation.py"
    templates = next(
        node.value
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "CHAT_TEMPLATES" for target in node.targets)
    )
    template = next(
        ast.literal_eval(value)
        for key, value in zip(templates.keys, templates.values, strict=True)
        if ast.literal_eval(key) == "qwen_instruct_user_boxed_math"
    )
    rows = []
    for dataset, location in config["prepared_sets"].items():
        for index, line in enumerate(Path(location).read_text().splitlines()):
            row = json.loads(line)
            rendered = tokenizer.apply_chat_template(
                row["metadata"]["opd_messages"], chat_template=template, tokenize=False, add_generation_prompt=True
            )
            if rendered != row["input"]:
                raise ValueError("Prepared AIME prompt differs from the original boxed-math template")
            rows.append(
                dict(
                    dataset=dataset,
                    question_id=f"{dataset}:{index}",
                    prompt=row["input"],
                    label=row["label"],
                    input_ids=tokenizer.encode(row["input"], add_special_tokens=False),
                )
            )
    expected = {"math_aime_2025": 30}
    counts = {name: sum(row["dataset"] == name for row in rows) for name in expected}
    if counts != expected or len({(row["dataset"], row["prompt"]) for row in rows}) != len(rows):
        raise ValueError(f"Unexpected evaluation panel: {counts}")
    if max(len(row["input_ids"]) for row in rows) > 2048:
        raise ValueError("Evaluation prompt exceeds the original prompt cap")
    manifest = {
        "config": config,
        "counts": counts,
        "checkpoints": {},
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
    }
    for name, location in config["checkpoints"].items():
        root = Path(location)
        if not any((root / marker).is_file() for marker in (".complete", ".checkpoint_complete")):
            raise ValueError(f"Incomplete or absent checkpoint: {root}")
        files = [root / "config.json", *sorted(root.glob("*.safetensors"))]
        if len(files) < 2:
            raise ValueError(f"No safetensors weights: {root}")
        manifest["checkpoints"][name] = {
            "path": str(root),
            "files": {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in files},
        }
    panel = output / "panel.jsonl"
    panel.write_text("".join(json.dumps(row) + "\n" for row in sorted(rows, key=lambda row: row["question_id"])))
    manifest["panel_sha256"] = digest(panel)
    (output / "verifiers.json").write_bytes(Path(config["registry"]).read_bytes())
    manifest["verifiers_sha256"] = digest(output / "verifiers.json")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.output)
