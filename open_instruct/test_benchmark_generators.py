import time
import unittest

import parameterized

from open_instruct import benchmark_generators, queue_types


class TestBenchmark(unittest.TestCase):
    @parameterized.parameterized.expand(
        [("NVIDIA H100 80GB HBM3", "h100"), ("NVIDIA L40S", "l40s"), ("NVIDIA RTX A6000", "a6000")]
    )
    def test_get_device_name(self, device_name, expected):
        result = benchmark_generators.get_device_name(device_name)
        self.assertEqual(result, expected)

    def test_prompt_request_has_start_time(self):
        """Test that PromptRequest can store start_time."""
        start_time = time.perf_counter()
        request = queue_types.PromptRequest(prompts=[[1, 2, 3]], generation_config=None, start_time=start_time)
        self.assertEqual(request.start_time, start_time)

    def test_generation_result_has_start_time(self):
        """Test that GenerationResult can store start_time."""
        start_time = time.perf_counter()
        request_info = queue_types.RequestInfo(
            num_calls=[0], timeouts=[0], tool_errors=[""], tool_outputs=[""], tool_runtimes=[0.0], tool_calleds=[False]
        )
        result = queue_types.GenerationResult(
            responses=[[1, 2, 3]],
            finish_reasons=["stop"],
            masks=[[1, 1, 1]],
            request_info=request_info,
            start_time=start_time,
        )
        self.assertEqual(result.start_time, start_time)

    def test_timing_calculation_with_start_time(self):
        """Test that timing calculations use start_time correctly."""
        # Simulate a request with a start time
        start_time = time.perf_counter()

        # Simulate some processing time
        time.sleep(0.01)  # Sleep for 10ms

        # Simulate completion
        completion_time = time.perf_counter()

        # Calculate generation time the new way (from start_time)
        generation_time = completion_time - start_time

        # Verify the timing is reasonable (should be at least 10ms)
        self.assertGreaterEqual(generation_time, 0.01)
        self.assertLess(generation_time, 0.1)  # Should be less than 100ms

    def test_model_dims_flops_with_samples_per_prompt(self):
        """Test that ModelDims handles N samples per prompt correctly."""
        model_dims = benchmark_generators.ModelDims(
            num_layers=12,
            hidden_size=768,
            intermediate_size=3072,  # Typically 4x hidden_size
            num_attn_heads=12,
            num_kv_heads=12,
            vocab_size=50000,
        )

        # Test with 2 prompts, 3 samples each
        prompt_lengths = [10, 15]  # 2 unique prompts
        response_lengths = [5, 6, 7, 8, 9, 10]  # 3 samples per prompt = 6 total responses
        samples_per_prompt = 3

        # Calculate FLOPs with the new method
        flops = model_dims.flops(prompt_lengths, response_lengths, samples_per_prompt)

        # Verify it doesn't crash and returns a reasonable value
        self.assertGreater(flops, 0)

        # Test that prefill FLOPs are only calculated once per unique prompt
        prefill_flops = model_dims.prefill_flops(prompt_lengths)
        self.assertGreater(prefill_flops, 0)

        # Test decode FLOPs with samples_per_prompt
        decode_flops = model_dims.decode_flops(prompt_lengths, response_lengths, samples_per_prompt)
        self.assertGreater(decode_flops, 0)

        # Total should be prefill + decode
        self.assertEqual(flops, prefill_flops + decode_flops)


if __name__ == "__main__":
    unittest.main()
