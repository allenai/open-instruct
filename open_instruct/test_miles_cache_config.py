"""CPU-only cache configuration contracts."""

import pytest

from open_instruct.miles.configuration.config import CoreConfig


def test_cache_controls_validate_types():
    for key in ("compiler_cache", "compiler_cache_restore", "compiler_cache_diagnostics"):
        with pytest.raises(ValueError, match=key):
            CoreConfig(**{key: "false"})


def test_cache_is_opt_out():
    assert CoreConfig().compiler_cache is True
    assert CoreConfig(compiler_cache=False).compiler_cache is False


@pytest.mark.parametrize(
    "root", ("relative/cache", "/weka", "/weka/oe-training-default/cache", "/weka/tmp-0d/cache", "", 42)
)
def test_invalid_cache_root_rejected_before_launch(root):
    with pytest.raises(ValueError):
        CoreConfig(compiler_cache_root=root)
