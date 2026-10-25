"""Fail closed when the binary image's FA4 wheel no longer matches our patch."""

import hashlib
from importlib import metadata, util
from pathlib import Path

import pytest

PATCH_PATH = Path(__file__).resolve().parents[2] / "runtime/miles/patch_flash_attn_auxdata.py"
SPEC = util.spec_from_file_location("patch_flash_attn_auxdata", PATCH_PATH)
patcher = util.module_from_spec(SPEC)
SPEC.loader.exec_module(patcher)


@pytest.fixture
def original_source(tmp_path):
    distribution = metadata.distribution("flash-attn-4")
    source = Path(distribution.locate_file("flash_attn/cute/utils.py")).read_text()
    # The production image is already patched; recover its pinned wheel bytes.
    source = source.replace(patcher.NEW_IMPORT, patcher.OLD_IMPORT).replace(patcher.NEW_CLASS, patcher.OLD_CLASS)
    assert hashlib.sha256(source.encode()).hexdigest() == patcher.SOURCE_SHA256
    path = tmp_path / "utils.py"
    path.write_text(source)
    return path


def test_patch_preserves_namedtuple_and_compiles(original_source):
    patcher.patch(original_source, patcher.VERSION)
    patched = original_source.read_text()
    compile(patched, str(original_source), "exec")
    assert patcher.NEW_CLASS in patched
    assert "class AuxData(NamedTuple):" in patched


@pytest.mark.parametrize("change", ["version", "source", "already_patched"])
def test_patch_refuses_unqualified_input_without_writing(original_source, change):
    version = patcher.VERSION
    if change == "version":
        version = "4.0.0b28"
    elif change == "source":
        original_source.write_text(original_source.read_text() + "\n# unexpected change\n")
    else:
        patcher.patch(original_source, version)
    before = original_source.read_bytes()
    with pytest.raises(ValueError, match="Unexpected flash-attn-4 source/version"):
        patcher.patch(original_source, version)
    assert original_source.read_bytes() == before
