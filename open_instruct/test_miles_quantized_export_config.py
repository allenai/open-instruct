import pytest

from open_instruct.miles.configuration.config import CoreConfig


def test_quantized_export_is_opt_in():
    assert CoreConfig().quantized_export is False
    assert CoreConfig(quantized_export=True).quantized_export is True


@pytest.mark.parametrize(
    "options", [{"expert_publication": "fused"}, {"stream_moe_export": False}, {"weight_sync_mode": "per_tensor"}]
)
def test_quantized_export_rejects_unsupported_publication(options):
    with pytest.raises(ValueError, match="quantized_export"):
        CoreConfig(quantized_export=True, **options)
