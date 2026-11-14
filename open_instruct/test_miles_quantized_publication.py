"""Online publication preserves the initial loader's scoped weight conversion."""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from open_instruct.miles.execution import quantized_publication as publication


@pytest.mark.parametrize("failure", [False, True])
def test_native_loader_recipe_covers_split_buckets_and_restores_environment(monkeypatch, failure):
    seen = []
    staged = []
    expected = dict(publication.NVFP4_LOADER_ENV)

    @contextmanager
    def temporary_environment(**values):
        with monkeypatch.context() as scoped:
            for name, value in values.items():
                scoped.setenv(name, value)
            yield

    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "original")
    monkeypatch.delenv("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", raising=False)
    monkeypatch.setattr(
        publication.importlib, "import_module", lambda _: SimpleNamespace(temp_set_env=temporary_environment)
    )

    class Model:
        quant_config = SimpleNamespace(is_nvfp4_online=True)

        def load_weights(self, weights, *, sentinel):
            assert sentinel == "native options"
            seen.append({name: os.environ.get(name) for name in expected})
            staged.extend(weights)
            if failure:
                raise RuntimeError("original loader error")
            return staged

    publication.install_nvfp4_publication_context(Model)
    wrapped = Model.load_weights
    publication.install_nvfp4_publication_context(Model)
    assert Model.load_weights is wrapped
    model = Model()
    for shard in ("w1", "w3"):
        weights = iter([(shard, object())])
        if failure:
            with pytest.raises(RuntimeError, match="original loader error"):
                model.load_weights(weights, sentinel="native options")
        else:
            assert model.load_weights(weights, sentinel="native options") is staged
        assert os.environ["FLASHINFER_NVFP4_4OVER6"] == "original"
        assert "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH" not in os.environ
    assert seen == [expected, expected]
    assert [name for name, _ in staged] == ["w1", "w3"]


@pytest.mark.parametrize("config", [None, SimpleNamespace(), SimpleNamespace(is_nvfp4_online=False)])
def test_other_precisions_keep_original_iterator_and_result(monkeypatch, config):
    marker = object()
    iterator = iter([("expert", marker)])

    def unexpected_import(name):
        raise AssertionError(f"Unexpected runtime import: {name}")

    monkeypatch.setattr(publication.importlib, "import_module", unexpected_import)

    class Model:
        quant_config = config

        def load_weights(self, weights):
            assert weights is iterator
            return marker

    publication.install_nvfp4_publication_context(Model)
    assert Model().load_weights(iterator) is marker
