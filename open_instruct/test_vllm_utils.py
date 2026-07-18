import asyncio
import logging
import unittest
from collections import defaultdict
from unittest import mock
from unittest.mock import MagicMock

import vllm
from parameterized import parameterized

from open_instruct import vllm_utils
from open_instruct.data_types import PromptRequest
from open_instruct.utils import ModelDims


class TestVllmWorkerHelpers(unittest.TestCase):
    def test_get_kv_cache_spec_as_dict_normalizes_mapping_subclasses(self):
        worker = MagicMock()
        worker.get_kv_cache_spec.return_value = defaultdict(list, {"layer": "spec"})

        result = vllm_utils._get_kv_cache_spec_as_dict(worker)

        self.assertIs(type(result), dict)
        self.assertEqual(result, {"layer": "spec"})

    def test_kv_cache_spec_rpc_target_preserves_cuda_string_api(self):
        with mock.patch("open_instruct.vllm_utils.utils.get_accelerator_type", return_value="cuda"):
            self.assertEqual(vllm_utils._get_kv_cache_spec_rpc_target(), "get_kv_cache_spec")

    def test_kv_cache_spec_rpc_target_uses_npu_normalizer(self):
        with mock.patch("open_instruct.vllm_utils.utils.get_accelerator_type", return_value="npu"):
            self.assertIs(vllm_utils._get_kv_cache_spec_rpc_target(), vllm_utils._get_kv_cache_spec_as_dict)

    def test_update_weights_uses_complete_vllm_lifecycle(self):
        actor = object.__new__(vllm_utils.LLMRayActor)
        actor.inflight_updates = True
        actor.active_tasks = {}
        actor.current_model_step = 0
        actor.llm_engine = MagicMock()
        events = []

        async def start_weight_update(*, is_checkpoint_format):
            events.append(("start", is_checkpoint_format))

        async def update_weights(request):
            events.append(("update", request.update_info))

        async def finish_weight_update():
            events.append(("finish",))

        actor.llm_engine.start_weight_update = start_weight_update
        actor.llm_engine.update_weights = update_weights
        actor.llm_engine.finish_weight_update = finish_weight_update
        actor._run_async = asyncio.run
        update_info = {
            "update_info": {
                "names": ["layer.weight"],
                "dtype_names": ["bfloat16"],
                "shapes": [[2, 2]],
                "packed": False,
            }
        }

        actor.update_weights(update_info, model_step=3)

        self.assertEqual(events, [("start", True), ("update", update_info["update_info"]), ("finish",)])
        self.assertEqual(actor.current_model_step, 3)

    def test_update_weights_preserves_legacy_vllm_api(self):
        class LegacyEngine:
            async def update_weights(self, request):
                events.append(("update", request.update_info))

        actor = object.__new__(vllm_utils.LLMRayActor)
        actor.inflight_updates = True
        actor.active_tasks = {}
        actor.current_model_step = 0
        actor.llm_engine = LegacyEngine()
        actor._run_async = asyncio.run
        events = []
        update_info = {
            "update_info": {
                "names": ["layer.weight"],
                "dtype_names": ["bfloat16"],
                "shapes": [[2, 2]],
                "packed": False,
            }
        }

        actor.update_weights(update_info, model_step=3)

        self.assertEqual(events, [("update", update_info["update_info"])])
        self.assertEqual(actor.current_model_step, 3)

    def test_openai_loopback_client_ignores_proxy_environment(self):
        actor = object.__new__(vllm_utils.LLMRayActor)
        actor.server_port = 12345
        actor.llm_engine = MagicMock()
        actor.llm_engine.vllm_config.model_config.model = "local-model"
        http_client = MagicMock()

        with (
            mock.patch("open_instruct.vllm_utils.httpx.AsyncClient", return_value=http_client) as client_cls,
            mock.patch("open_instruct.vllm_utils.openai.AsyncOpenAI") as openai_cls,
            mock.patch("open_instruct.vllm_utils._check_health", new=mock.AsyncMock()),
        ):
            actor._init_openai_client()

        client_cls.assert_called_once_with(trust_env=False)
        openai_cls.assert_called_once_with(
            base_url="http://127.0.0.1:12345/v1", api_key="EMPTY", timeout=3600, http_client=http_client
        )
        self.assertEqual(actor.model_name, "local-model")


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
            mock.patch("open_instruct.utils.get_accelerator_type", return_value="cuda"),
        ):
            vllm_dims = vllm_utils.model_dims_from_vllm_config(mock_vllm_config)

        self.assertEqual(vllm_dims, expected_dims)


if __name__ == "__main__":
    unittest.main()
