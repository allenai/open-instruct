"""Documentation coverage and launch examples must follow the real CPU contract."""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.miles import docs_site, generate_docs

from open_instruct.miles import launch, run_spec

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/miles"


def test_generated_reference_is_current():
    for path, expected in generate_docs.render().items():
        assert path.read_text() == expected, f"Regenerate {path}"


def test_structured_field_descriptions_cover_closed_schemas():
    descriptions = json.loads((DOCS / "reference-help.json").read_text())["structured"]
    tree = ast.parse((ROOT / "open_instruct/miles/run_spec.py").read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_table"
            and len(node.args) >= 3
            and isinstance(node.args[2], ast.Set)
        ):
            section = ast.literal_eval(node.args[1])
            for field in ast.literal_eval(node.args[2]):
                assert f"{section}.{field}" in descriptions
        if isinstance(node, ast.FunctionDef) and node.name in ("_data", "_launch"):
            allowed = next(
                child
                for child in node.body
                if isinstance(child, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "allowed" for t in child.targets)
            )
            for field in ast.literal_eval(allowed.value):
                assert f"{node.name[1:]}.{field}" in descriptions
    # Judge/rubric schemas are separate from RunSpec.
    tree = ast.parse((ROOT / "open_instruct/miles/judging.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "fields":
            if len(node.args) != 3 or not isinstance(node.args[2], ast.Set):
                continue
            keys = ast.literal_eval(node.args[2])
            prefix = (
                "judges.NAME"
                if "prepared_dir" in keys
                else "rubrics.NAME"
                if "profile" in keys
                else "judging.bindings.VERIFIER"
                if "judge" in keys
                else None
            )
            if prefix:
                for key in keys:
                    assert f"{prefix}.{key}" in descriptions


def test_special_control_descriptions_cover_dispatch():
    described = json.loads((DOCS / "reference-help.json").read_text())["special"]
    tree = ast.parse((ROOT / "open_instruct/miles/run_spec.py").read_text())
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id == "RUN_SECTIONS"
    )
    for node in ast.walk(loop):
        if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name) or node.left.id != "key":
            continue
        for value in node.comparators:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                assert value.value in described
            elif isinstance(value, ast.Tuple):
                for key in ast.literal_eval(value):
                    assert key in described


@pytest.mark.parametrize("path", sorted((ROOT / "configs/miles/examples").glob("*.toml")), ids=lambda p: p.stem)
def test_example_plans_and_launch_render_without_gpu_or_network(path):
    spec = run_spec.RunSpec.load(path)
    plan = spec.plan()
    tasks = launch.specification("test-image", spec, hostnames=[f"host-{i}" for i in range(32)])["tasks"]
    assert len(tasks) == plan["allocation"]["replicas"]
    assert all(task["resources"]["gpuCount"] > 0 for task in tasks)
    result = subprocess.run(
        [sys.executable, "-m", "open_instruct.miles", "validate", str(path)], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_current_guide_links_exist():
    for path in DOCS.rglob("*.md"):
        content = re.sub(r"```.*?```", "", path.read_text(), flags=re.DOTALL)
        for link in re.findall(r"!?\[[^\]\n]*\]\(([^\s)]+)(?:[^)]*)\)", content):
            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", link) or link.startswith(("/", "#")):
                continue
            assert (path.parent / link.split("#")[0]).exists(), (path, link)


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
    assert "https://github.com/allenai/open-instruct/blob/robertb/miles-olmo-core/configs/miles/README.md" in result
    assert "[workflow](workflow.md)" in result
