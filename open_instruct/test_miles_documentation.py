"""Documentation coverage and launch examples must follow the real CPU contract."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mkdocs.structure.files import Files
from scripts.miles import docs_site, generate_docs

from open_instruct.miles.configuration import constraints, options, run_spec
from open_instruct.miles.execution import launch

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/miles"


def test_references_build_without_runtime_or_native_help(monkeypatch, tmp_path):
    monkeypatch.delenv("MILES_NATIVE_HELP", raising=False)
    config = SimpleNamespace(
        site_dir=str(tmp_path), use_directory_urls=True, plugins=SimpleNamespace(_current_plugin=None)
    )
    files = docs_site.on_files(Files([]), config)
    assert {file.src_uri for file in files} == {
        "miles/configuration.md",
        "miles/native-options.md",
        "miles/native-training-options.md",
        "miles/native-serving-options.md",
    }
    native = files.get_file_from_path("miles/native-training-options.md").content_string
    assert "Not captured" in native and "global_batch_size" in native
    assert not (DOCS / "native-help.json").exists()
    assert all(not path.exists() for path in generate_docs.render())


def test_optional_help_requires_matching_provenance_and_complete_actions(tmp_path):
    schema_path = ROOT / "open_instruct/miles/configuration/options.json"
    capture = {
        "image": "test-image",
        "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        "options": [
            {"dest": r["dest"], "flags": r["flags"], "help": "Captured description", "default": 17}
            for r in options.load_schema()["options"]
        ],
    }
    path = tmp_path / "help.json"
    path.write_text(json.dumps(capture))
    rendered = generate_docs.render(path)
    assert "Captured description" in rendered[DOCS / "native-serving-options.md"]
    assert "test-image" in rendered[DOCS / "native-options.md"]
    capture["options"].pop()
    path.write_text(json.dumps(capture))
    with pytest.raises(ValueError, match="every pinned parser action"):
        generate_docs.render(path)
    capture["schema_sha256"] = "stale"
    path.write_text(json.dumps(capture))
    with pytest.raises(ValueError, match="provenance is stale"):
        generate_docs.render(path)


def test_structured_field_descriptions_cover_closed_schemas():
    descriptions = json.loads((DOCS / "reference-help.json").read_text())["structured"]
    for section, fields in constraints.STRUCTURED_FIELDS.items():
        for field in fields:
            assert f"{section}.{field}" in descriptions


def test_special_control_descriptions_cover_dispatch():
    described = json.loads((DOCS / "reference-help.json").read_text())["special"]
    assert constraints.SPECIAL_CONTROLS.keys() <= described.keys()


@pytest.mark.parametrize("path", sorted((ROOT / "configs/miles/examples").glob("*.toml")), ids=lambda p: p.stem)
def test_example_plans_and_launch_render_without_gpu_or_network(path):
    spec = run_spec.RunSpec.load(path)
    plan = spec.plan()
    tasks = launch.specification("test-image", spec, hostnames=[f"host-{i}" for i in range(32)])["tasks"]
    assert len(tasks) == 1
    assert tasks[0].get("replicas", 1) == plan["allocation"]["replicas"]
    assert all(task["resources"]["gpuCount"] > 0 for task in tasks)
    result = subprocess.run(
        [sys.executable, "-m", "open_instruct.miles", "validate", str(path)], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_current_guide_links_exist():
    generated = generate_docs.render()
    documents = {path: path.read_text() for path in DOCS.rglob("*.md")} | generated
    for path, text in documents.items():
        content = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        for link in re.findall(r"!?\[[^\]\n]*\]\(([^\s)]+)(?:[^)]*)\)", content):
            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", link) or link.startswith(("/", "#")):
                continue
            target = (path.parent / link.split("#")[0]).resolve()
            assert target.exists() or target in generated, (path, link)


def test_site_links_target_repository_without_changing_local_doc_links():
    class File:
        src_uri = "miles/index.md"

    class Page:
        file = File()

    value = "[config](../../configs/miles/README.md) [workflow](workflow.md)"
    result = docs_site.on_page_markdown(
        value,
        page=Page(),
        config={"docs_dir": str(ROOT / "docs"), "repo_url": "https://github.com/allenai/open-instruct"},
        files=None,
    )
    assert "https://github.com/allenai/open-instruct/blob/main/configs/miles/README.md" in result
    assert "[workflow](workflow.md)" in result
