import pytest
from scripts.miles.benchmark_quant_backend import measure, validate_outputs
from scripts.miles.prepare_quant_backend_runtime import patch_top_k


def test_adapter_topk_is_forwarded_without_changing_routing():
    source = "self.experts = expert_class(\n            num_experts=self.num_experts,\n            hidden_size=expert_hidden_size,\n)"
    patched = patch_top_k(source)
    assert patched.replace("            top_k=config.num_experts_per_tok,\n", "") == source
    with pytest.raises(ValueError, match="Unexpected"):
        patch_top_k(patched)


def test_truncated_or_missing_work_is_not_a_speed_result():
    with pytest.raises(ValueError, match="Missing"):
        validate_outputs([], 2, 10)
    with pytest.raises(ValueError, match="truncated"):
        validate_outputs([{"meta_info": {"completion_tokens": 9}}], 1, 10)


def test_queued_batch_has_exact_token_work_and_flushes_before_timing():
    class Engine:
        flushed = False

        def flush_cache(self):
            self.flushed = True

        def generate(self, *, input_ids, sampling_params, stream):
            assert self.flushed and not stream
            assert sampling_params["ignore_eos"]
            assert len(input_ids) == 2048
            return [{"meta_info": {"completion_tokens": 512}, "text": "x"} for _ in input_ids]

    result = measure(Engine(), [[1, 2]] * 2048, 512)
    assert result["output_tokens"] == 1048576
    assert result["requests"] == 2048
    assert result["output_tokens_per_second"] * result["seconds"] == pytest.approx(1048576)
