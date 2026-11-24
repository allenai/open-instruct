"""scripts/eval/olmo35_mm/launch_text_cells.py: text eval cells never install anything from scaling-ladders."""

import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "olmo35_mm" / "launch_text_cells.py"
spec = importlib.util.spec_from_file_location("launch_text_cells", PATH)
assert spec is not None and spec.loader is not None
cells = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cells)

REF = "0123456789abcdef0123456789abcdef01234567"
TREE = "/weka/x/repos/scaling-ladders-hero/ladders/olmoe3"
TAR = "/weka/x/handoff/ladders-3542d99.tar"


def make_spec(arguments, env_value=""):
    return {"tasks": [{"arguments": arguments, "envVars": [{"name": "INSTALL", "value": env_value}, {"name": "S", "secret": "x"}]}]}


def test_vllm_cell_requirements_move_to_open_instruct():
    deps = (
        f"provider.dependencies=[vllm==0.19.1,olmoe3-vllm-plugin @ file://{TREE}/vllm_plugin,"
        f"olmoe3-transformers-plugin @ file://{TREE}/transformers_plugin]"
    )
    spec = make_spec(["olmo-eval", deps], env_value=f"uv pip install 'olmoe3-vllm-plugin @ file://{TREE}/vllm_plugin'")
    assert cells.use_open_instruct_plugins(spec, REF) == 3
    text = json.dumps(spec)
    assert "ladders" not in text
    for pkg in ("vllm_plugin", "transformers_plugin"):
        assert cells.plugin_url(REF, pkg) in text
    assert cells.plugin_url(REF, "vllm_plugin") == (
        f"git+https://github.com/allenai/open-instruct.git@{REF}#subdirectory=scripts/export/olmo35_mm/vllm_plugin"
    )


def test_bfcl_cell_drops_the_archive_and_installs_by_name():
    script = "\n".join(
        [
            "set -euo pipefail",
            f'echo "abc  {TAR}" | sha256sum -c -',
            f"mkdir -p /tmp/hero-ladder && tar -xf {TAR} -C /tmp/hero-ladder",
            "test -d /tmp/hero-ladder/ladders/olmoe3/vllm_plugin && test -d /tmp/hero-ladder/ladders/olmoe3/transformers_plugin",
            "uv pip install 'vllm==0.19.1' -e /tmp/hero-core -e /tmp/hero-ladder/ladders/olmoe3/vllm_plugin "
            "-e /tmp/hero-ladder/ladders/olmoe3/transformers_plugin",
            "python run_bfcl.py",
        ]
    )
    spec = make_spec([script])
    assert cells.use_open_instruct_plugins(spec, REF) == 2
    out = spec["tasks"][0]["arguments"][0]
    assert "ladders" not in out and "sha256sum" not in out and "tar -xf" not in out
    assert f"'olmoe3-vllm-plugin @ {cells.plugin_url(REF, 'vllm_plugin')}'" in out
    assert "-e /tmp/hero-core" in out and out.endswith("python run_bfcl.py")


def test_leftover_scaling_ladders_reference_is_refused():
    spec = make_spec(["see https://github.com/allenai/scaling-ladders for details"])
    with pytest.raises(SystemExit):
        cells.use_open_instruct_plugins(spec, REF)
