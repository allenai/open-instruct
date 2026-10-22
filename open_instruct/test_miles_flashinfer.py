"""The image gate rejects mismatched FlashInfer packages before GPU scheduling."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from scripts.miles import check_flashinfer


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    (tmp_path / "python").mkdir()
    (tmp_path / "python/pyproject.toml").write_text('[project]\ndependencies = ["flashinfer_python[cu13]==0.6.18"]\n')
    versions = {"flashinfer-python": "0.6.18", "flashinfer-cubin": "0.6.18", "nvidia-cutlass-dsl": "4.6.2"}
    monkeypatch.setattr(check_flashinfer.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(
        check_flashinfer.metadata, "requires", lambda _: ['nvidia-cutlass-dsl[cu13]>=4.6.2; extra == "cu13"']
    )
    module = SimpleNamespace(
        __name__="flashinfer.gemm.kernels.test",
        default_tactic=object(),
        prefer_direct_bf16_gemm_sm100=object(),
        run_direct_dense=object(),
        SplitKTactic=object(),
        run_splitk_dense=object(),
    )
    imports = Mock(return_value=module)
    monkeypatch.setattr(check_flashinfer.importlib, "import_module", imports)
    return tmp_path, versions, module, imports


def test_matching_packages_and_kernel_symbols(runtime):
    root, _, _, imports = runtime
    check_flashinfer.check(root)
    assert imports.call_count == 2


@pytest.mark.parametrize("package", ["flashinfer-python", "flashinfer-cubin", "nvidia-cutlass-dsl"])
def test_mismatched_packages_fail_before_kernel_import(runtime, package):
    root, versions, _, imports = runtime
    versions[package] = "0.6.17"
    with pytest.raises(ValueError):
        check_flashinfer.check(root)
    imports.assert_not_called()


def test_missing_kernel_symbol_fails(runtime):
    root, _, module, _ = runtime
    del module.run_splitk_dense
    with pytest.raises(ImportError, match="run_splitk_dense"):
        check_flashinfer.check(root)
