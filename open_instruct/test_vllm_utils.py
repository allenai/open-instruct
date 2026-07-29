import base64
import io
import logging
import unittest
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import vllm
from openai.types.completion import Completion
from parameterized import parameterized

from open_instruct import vllm_utils
from open_instruct.data_types import PromptRequest
from open_instruct.utils import ModelDims


class TestTruncateEnvOutputTokens(unittest.TestCase):
    @parameterized.expand(
        [
            ("no_truncation", [1, 2, 3, 4, 5], 10, 5, 100, 50, [1, 2, 3, 4, 5], 0),
            ("truncate_max_model_len", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95, 5, 100, 50, [1, 2, 3, 4, 5], 5),
            ("truncate_max_tokens", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10, 47, 100, 50, [1, 2, 3], 0),
            ("truncate_both_limits", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95, 47, 100, 50, [1, 2, 3], 5),
            ("no_space_for_model", [1, 2, 3], 100, 5, 100, 50, [], 3),
            ("no_remaining_response", [1, 2, 3], 10, 50, 100, 50, [], 0),
            ("empty_input", [], 10, 5, 100, 50, [], 0),
        ]
    )
    def test_truncate_tool_output_tokens(
        self, name, tokens, prompt_len, response_len, max_model_len, max_tokens, expected_tokens, expected_excess
    ):
        result, excess = vllm_utils.truncate_tool_output_tokens(
            tokens, prompt_len, response_len, max_model_len, max_tokens
        )
        self.assertEqual(result, expected_tokens)
        self.assertEqual(excess, expected_excess)


class TestDecodeRoutedExperts(unittest.TestCase):
    def test_decodes_vllm_numpy_payload(self):
        expected = np.arange(24, dtype=np.uint8).reshape(3, 4, 2)
        buffer = io.BytesIO()
        np.save(buffer, expected)
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")

        actual = vllm_utils.decode_routed_experts(payload)

        self.assertEqual(actual, expected.tolist())

    def test_accepts_legacy_list_payload(self):
        routes = [[[1, 2], [3, 4]]]

        self.assertIs(vllm_utils.decode_routed_experts(routes), routes)

    def test_rejects_invalid_shape(self):
        buffer = io.BytesIO()
        np.save(buffer, np.zeros((3, 4), dtype=np.uint8))
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")

        with self.assertRaisesRegex(ValueError, "must have shape"):
            vllm_utils.decode_routed_experts(payload)

    def test_openai_sdk_preserves_vllm_route_field(self):
        response = Completion.model_validate(
            {
                "id": "completion-id",
                "choices": [
                    {"finish_reason": "stop", "index": 0, "logprobs": None, "text": "", "routed_experts": "abc"}
                ],
                "created": 0,
                "model": "model",
                "object": "text_completion",
            }
        )

        self.assertEqual(response.choices[0].routed_experts, "abc")


class TestCompletionExtraBody(unittest.TestCase):
    def test_requests_routes_when_capture_is_enabled(self):
        extra_body = vllm_utils.get_completion_extra_body(
            base_request_id="train_0_0", min_tokens=1, return_routed_experts=True
        )

        self.assertIs(extra_body["return_routed_experts"], True)

    def test_omits_routes_when_capture_is_disabled(self):
        extra_body = vllm_utils.get_completion_extra_body(
            base_request_id="train_0_0", min_tokens=1, return_routed_experts=False
        )

        self.assertNotIn("return_routed_experts", extra_body)


class TestVllmWeightUpdate(unittest.TestCase):
    def _actor(self, split_reload_lifecycle: bool = True):
        actor = object.__new__(vllm_utils.LLMRayActor)
        actor.inflight_updates = True
        actor.active_tasks = set()
        actor.return_routed_experts = False
        actor.current_model_step = None
        engine_methods = ["update_weights", "sleep", "wake_up", "reset_prefix_cache"]
        if split_reload_lifecycle:
            engine_methods.extend(["start_weight_update", "finish_weight_update"])
        actor.llm_engine = mock.Mock(spec=engine_methods)
        actor._run_async = mock.Mock()
        return actor

    def test_uses_native_update_lifecycle_when_split_methods_are_absent(self):
        actor = self._actor(split_reload_lifecycle=False)
        update = object()
        actor.llm_engine.update_weights.return_value = update

        actor.update_weights({"update_info": {"names": []}}, model_step=3)

        actor._run_async.assert_called_once_with(update)
        self.assertEqual(actor.current_model_step, 3)

    def test_wraps_update_in_reload_lifecycle(self):
        actor = self._actor()
        actor.return_routed_experts = True
        start, update, finish = object(), object(), object()
        actor.llm_engine.start_weight_update.return_value = start
        actor.llm_engine.update_weights.return_value = update
        actor.llm_engine.finish_weight_update.return_value = finish

        actor.update_weights({"update_info": {"names": []}}, model_step=3)

        self.assertEqual(actor._run_async.call_args_list, [mock.call(start), mock.call(update), mock.call(finish)])
        self.assertEqual(actor.current_model_step, 3)

    def test_finishes_reload_lifecycle_when_update_fails(self):
        actor = self._actor()
        actor._run_async.side_effect = [None, RuntimeError("update failed"), None]

        with self.assertRaisesRegex(RuntimeError, "update failed"):
            actor.update_weights({"update_info": {"names": []}})

        self.assertEqual(actor._run_async.call_count, 3)
        actor.llm_engine.finish_weight_update.assert_called_once_with()

    def test_sleep_drains_active_tasks_before_suspending_scheduling(self):
        actor = self._actor()
        actor.inflight_updates = False
        actor.active_tasks = {object()}
        actor.check_background_threads = mock.Mock()
        sleep_request = object()
        actor.llm_engine.sleep.return_value = sleep_request

        with mock.patch.object(vllm_utils.time, "sleep", side_effect=lambda _: actor.active_tasks.clear()):
            actor.sleep()

        actor.check_background_threads.assert_called_once_with()
        actor.llm_engine.sleep.assert_called_once_with(level=0, mode="keep")
        actor._run_async.assert_called_once_with(sleep_request)


class TestCreateVllmEngines(unittest.TestCase):
    def test_passes_runtime_kernel_and_batch_settings_to_vllm(self):
        engine = MagicMock()
        actor_options = MagicMock()
        actor_options.remote.return_value = engine
        actor_cls = MagicMock()
        actor_cls.options.return_value = actor_options
        placement_group = MagicMock()

        with (
            mock.patch.object(vllm_utils.ray, "remote", return_value=actor_cls),
            mock.patch.object(vllm_utils.ray, "get"),
            mock.patch.object(vllm_utils.ray.runtime_env, "RuntimeEnv"),
            mock.patch.object(vllm_utils, "placement_group", return_value=placement_group),
            mock.patch.object(vllm_utils, "get_bundle_indices_list", return_value=[0]),
            mock.patch.object(vllm_utils, "get_cuda_arch_list", return_value="9.0"),
            mock.patch.object(vllm_utils, "ray_noset_visible_devices", return_value=False),
            mock.patch.object(vllm_utils, "PlacementGroupSchedulingStrategy"),
            mock.patch.object(vllm_utils.utils, "ray_get_with_progress"),
        ):
            engines = vllm_utils.create_vllm_engines(
                num_engines=1,
                tensor_parallel_size=1,
                enforce_eager=True,
                tokenizer_name_or_path="tokenizer",
                pretrain="model",
                revision=None,
                seed=42,
                enable_prefix_caching=False,
                max_model_len=384,
                prompt_queue=MagicMock(),
                results_queue=MagicMock(),
                eval_results_queue=MagicMock(),
                actor_manager=MagicMock(),
                vllm_moe_backend="triton",
                vllm_max_num_batched_tokens=384,
            )

        self.assertEqual(engines, [engine])
        engine_kwargs = actor_options.remote.call_args.kwargs
        self.assertEqual(engine_kwargs["moe_backend"], "triton")
        self.assertEqual(engine_kwargs["max_num_batched_tokens"], 384)


class TestVllmUtils3(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_process_outputs_with_tools(self):
        """Test that process_completed_request correctly handles outputs with tool attributes.

        Tests the new process_completed_request function which combined process_output and _process_completed_request.
        """

        def create_mock_logprobs(token_ids):
            return [-0.1 * tid for tid in token_ids]

        idx = 43039
        epoch = 0
        prompt_id = f"{epoch}_{idx}"

        mock_request = PromptRequest(
            prompt=[1, 2, 3], generation_config=None, is_eval=False, index=idx, prompt_id=prompt_id
        )
        request_id = vllm_utils.make_request_id(mock_request)

        mock_output1 = MagicMock(spec=vllm.CompletionOutput)
        mock_output1.token_ids = [1, 2, 3]
        mock_output1.logprobs = create_mock_logprobs([1, 2, 3])
        mock_output1.mask = [1, 1, 1]
        mock_output1.rollout_state = {
            "step_count": 1,
            "timeout": False,
            "tool_error": "",
            "tool_output": "result1",
            "tool_runtime": 0.5,
            "tool_call_stats": [],
            "rewards": [],
            "done": False,
            "info": {},
        }
        mock_output1.finish_reason = "stop"

        mock_output2 = MagicMock(spec=vllm.CompletionOutput)
        mock_output2.token_ids = [4, 5, 6]
        mock_output2.logprobs = create_mock_logprobs([4, 5, 6])
        mock_output2.mask = [1, 1, 1]
        mock_output2.rollout_state = {
            "step_count": 2,
            "timeout": False,
            "tool_error": "",
            "tool_output": "result2",
            "tool_runtime": 0.3,
            "tool_call_stats": [],
            "rewards": [],
            "done": False,
            "info": {},
        }
        mock_output2.finish_reason = "stop"

        mock_request_output = MagicMock(spec=vllm.RequestOutput)
        mock_request_output.request_id = request_id
        mock_request_output.outputs = [mock_output1, mock_output2]
        mock_request_output.prompt = "test prompt"
        mock_request_output.prompt_token_ids = [1, 2, 3]
        mock_request_output.finished = True

        request_metadata = {
            request_id: {
                "is_eval": False,
                "index": idx,
                "prompt_id": prompt_id,
                "prompt_token_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "start_time": 1000.0,
                "model_step": 0,
            }
        }

        result, is_eval, _example = vllm_utils.process_completed_request(
            request_id=request_id,
            outs=[mock_request_output],
            current_time=1001.0,
            use_tools=True,
            request_metadata=request_metadata,
        )

        # Verify is_eval is correct
        self.assertFalse(is_eval)

        # Verify that we get both responses
        self.assertEqual(len(result.responses), 2, "Expected exactly 2 responses")

        # Verify the responses are correct
        self.assertEqual(result.responses[0], [1, 2, 3])
        self.assertEqual(result.responses[1], [4, 5, 6])

        # Verify masks are correct
        self.assertEqual(len(result.masks), 2)
        self.assertEqual(result.masks[0], [1, 1, 1])
        self.assertEqual(result.masks[1], [1, 1, 1])

        # Verify request_info has correct tool attributes (read from rollout_state dicts)
        self.assertEqual(result.request_info.num_calls, [1, 2])
        self.assertEqual(result.request_info.tool_outputs, ["result1", "result2"])
        self.assertEqual(result.request_info.tool_runtimes, [0.5, 0.3])
        self.assertEqual(result.request_info.tool_calleds, [True, True])

    def test_process_outputs_without_tools(self):
        """Test that process_completed_request correctly handles outputs without tool attributes."""

        def create_mock_logprobs(token_ids):
            return [-0.1 * tid for tid in token_ids]

        idx = 200
        epoch = 0
        prompt_id = f"{epoch}_{idx}"

        mock_request = PromptRequest(
            prompt=[1, 2, 3], generation_config=None, is_eval=True, index=idx, prompt_id=prompt_id
        )
        request_id = vllm_utils.make_request_id(mock_request)

        mock_output1 = MagicMock(spec=vllm.CompletionOutput)
        mock_output1.token_ids = [1, 2, 3]
        mock_output1.logprobs = create_mock_logprobs([1, 2, 3])
        mock_output1.finish_reason = "stop"

        mock_output2 = MagicMock(spec=vllm.CompletionOutput)
        mock_output2.token_ids = [4, 5, 6]
        mock_output2.logprobs = create_mock_logprobs([4, 5, 6])
        mock_output2.finish_reason = "length"

        mock_request_output = MagicMock(spec=vllm.RequestOutput)
        mock_request_output.request_id = request_id
        mock_request_output.outputs = [mock_output1, mock_output2]
        mock_request_output.prompt = "test prompt"
        mock_request_output.prompt_token_ids = [1, 2, 3]
        mock_request_output.finished = True

        request_metadata = {
            request_id: {
                "is_eval": True,
                "index": idx,
                "prompt_id": prompt_id,
                "prompt_token_ids": [1, 2, 3, 4, 5],
                "start_time": 2000.0,
                "model_step": 0,
            }
        }

        result, is_eval, _example = vllm_utils.process_completed_request(
            request_id=request_id,
            outs=[mock_request_output],
            current_time=2000.5,
            use_tools=False,
            request_metadata=request_metadata,
        )

        # Verify is_eval is correct
        self.assertTrue(is_eval)

        # Verify that we get both responses
        self.assertEqual(len(result.responses), 2, "Expected exactly 2 responses")

        # Verify the responses are correct
        self.assertEqual(result.responses[0], [1, 2, 3])
        self.assertEqual(result.responses[1], [4, 5, 6])

        # Verify finish reasons
        self.assertEqual(result.finish_reasons[0], "stop")
        self.assertEqual(result.finish_reasons[1], "length")

        # Verify default masks (all 1s when no tools)
        self.assertEqual(result.masks[0], [1, 1, 1])
        self.assertEqual(result.masks[1], [1, 1, 1])

        # Verify request_info has default values when tools are not used
        self.assertEqual(result.request_info.num_calls, [0, 0])
        self.assertEqual(result.request_info.timeouts, [False, False])
        self.assertEqual(result.request_info.tool_errors, ["", ""])
        self.assertEqual(result.request_info.tool_outputs, ["", ""])
        self.assertEqual(result.request_info.tool_runtimes, [0.0, 0.0])
        self.assertEqual(result.request_info.tool_calleds, [False, False])
        self.assertEqual(result.request_info.rollout_states, [{}, {}])


class TestModelDimsFromVllmConfig(unittest.TestCase):
    def test_model_dims_from_vllm_config(self):
        expected_dims = ModelDims(
            num_layers=28,
            hidden_size=3584,
            intermediate_size=18944,
            vocab_size=152064,
            num_attn_heads=28,
            head_dim=128,
            num_kv_heads=4,
            device_name="h100",
        )

        mock_hf_text_config = mock.Mock()
        mock_hf_text_config.intermediate_size = 18944
        mock_hf_text_config.sliding_window = None
        mock_hf_text_config.num_attention_heads = 28
        mock_hf_text_config.num_key_value_heads = 4

        mock_model_config = mock.Mock()
        mock_model_config.get_hidden_size.return_value = 3584
        mock_model_config.get_num_layers.return_value = 28
        mock_model_config.get_vocab_size.return_value = 152064
        mock_model_config.get_head_size.return_value = 128
        mock_model_config.hf_text_config = mock_hf_text_config

        mock_vllm_config = mock.Mock()
        mock_vllm_config.model_config = mock_model_config
        mock_vllm_config.parallel_config = mock.Mock()

        with (
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA H100 80GB HBM3"),
            mock.patch("torch.cuda.is_available", return_value=True),
        ):
            vllm_dims = vllm_utils.model_dims_from_vllm_config(mock_vllm_config)

        self.assertEqual(vllm_dims, expected_dims)


if __name__ == "__main__":
    unittest.main()
