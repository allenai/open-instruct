import logging
import unittest
from unittest import mock
from unittest.mock import MagicMock

import torch
import vllm
from parameterized import parameterized

from open_instruct import grpo_utils, vllm_utils
from open_instruct.data_types import PromptRequest
from open_instruct.utils import ModelDims
from open_instruct.weight_export import PassthroughWeightExportAdapter


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
        ):
            vllm_dims = vllm_utils.model_dims_from_vllm_config(mock_vllm_config)

        self.assertEqual(vllm_dims, expected_dims)


class TestWeightSyncLifecycle(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(2, 2, bias=False)
        self.model.config = mock.Mock(model_type="dense")
        self.engines = [mock.sentinel.engine_0, mock.sentinel.engine_1]

    def test_complete_ipc_sync_orders_lifecycle_and_model_step(self):
        events = []

        def call_engines(engines, method, *args):
            self.assertEqual(engines, self.engines)
            events.append((method, args))
            return []

        with (
            mock.patch.object(vllm_utils, "_call_engine_method", side_effect=call_engines),
            mock.patch.object(
                vllm_utils, "_broadcast_weights_ipc", side_effect=lambda *args: events.append(("send", ()))
            ),
            mock.patch.object(vllm_utils.torch.distributed, "is_initialized", return_value=False),
        ):
            refs = vllm_utils.broadcast_weights_to_vllm(
                self.model, self.engines, model_update_group=None, model_step=7
            )

        self.assertEqual(refs, [])
        self.assertEqual(
            events,
            [
                ("sleep", ()),
                ("wake_up_weights", ()),
                ("start_weight_update", ()),
                ("send", ()),
                ("finish_weight_update", ()),
                ("set_model_step", (7,)),
            ],
        )

    def test_transfer_failure_preserves_primary_error_and_does_not_stamp_step(self):
        events = []
        transfer_error = RuntimeError("primary transfer failure")

        def call_engines(_engines, method, *_args):
            events.append(method)
            if method == "finish_weight_update":
                raise RuntimeError("cleanup failure")
            return []

        with (
            mock.patch.object(vllm_utils, "_call_engine_method", side_effect=call_engines),
            mock.patch.object(vllm_utils, "_broadcast_weights_ipc", side_effect=transfer_error),
            mock.patch.object(vllm_utils.torch.distributed, "is_initialized", return_value=False),
            self.assertRaisesRegex(RuntimeError, "primary transfer failure") as raised,
        ):
            vllm_utils.broadcast_weights_to_vllm(self.model, self.engines, model_update_group=None, model_step=7)

        self.assertIs(raised.exception, transfer_error)
        self.assertEqual(events, ["sleep", "wake_up_weights", "start_weight_update", "finish_weight_update"])
        self.assertNotIn("set_model_step", events)

    def test_distributed_nonzero_rank_without_transfer_handle_uses_nccl_path(self):
        for param in self.model.parameters():
            param.ds_id = id(param)

        with (
            mock.patch.object(vllm_utils.torch.distributed, "is_initialized", return_value=True),
            mock.patch.object(vllm_utils.torch.distributed, "get_rank", return_value=1),
            mock.patch.object(vllm_utils.torch.distributed, "get_world_size", return_value=8),
            mock.patch.object(vllm_utils, "_distributed_barrier"),
            mock.patch.object(vllm_utils, "_send_nccl_weights") as send_nccl_weights,
            mock.patch.object(vllm_utils, "_broadcast_weights_ipc") as broadcast_weights_ipc,
        ):
            refs = vllm_utils.broadcast_weights_to_vllm(
                self.model, self.engines, model_update_group=None, model_step=7, gather_whole_model=False
            )

        self.assertEqual(refs, [])
        send_nccl_weights.assert_called_once()
        broadcast_weights_ipc.assert_not_called()

    def test_ipc_sender_uses_packed_send_mode_and_all_engine_handles(self):
        captured = {}

        def send_weights(iterator, trainer_args):
            captured["weights"] = list(iterator)
            captured["args"] = trainer_args

        with mock.patch.object(vllm_utils.IPCWeightTransferEngine, "trainer_send_weights", side_effect=send_weights):
            vllm_utils._broadcast_weights_ipc(
                self.model,
                self.engines,
                name_mapper=None,
                gather_whole_model=True,
                adapter=PassthroughWeightExportAdapter(),
            )

        self.assertEqual(captured["args"].send_mode, "ray")
        self.assertEqual(captured["args"].llm_handle, self.engines)
        self.assertTrue(captured["args"].packed)
        self.assertEqual([name for name, _ in captured["weights"]], ["weight"])

    def test_engine_sleep_and_wake_tags_match_vllm_weight_update_lifecycle(self):
        actor = object.__new__(vllm_utils.LLMRayActor)
        actor.llm_engine = mock.Mock()
        actor.llm_engine.sleep.return_value = mock.sentinel.sleep
        actor.llm_engine.wake_up.side_effect = [mock.sentinel.weights, mock.sentinel.inference]
        actor._run_async = mock.Mock()

        actor.sleep()
        actor.wake_up_weights()
        actor.wake_up()

        actor.llm_engine.sleep.assert_called_once_with(level=1)
        self.assertEqual(
            actor.llm_engine.wake_up.call_args_list,
            [mock.call(tags=["weights"]), mock.call(tags=["kv_cache", "scheduling"])],
        )
        self.assertEqual(
            actor._run_async.call_args_list,
            [mock.call(mock.sentinel.sleep), mock.call(mock.sentinel.weights), mock.call(mock.sentinel.inference)],
        )


class TestWeightGatherPolicy(unittest.TestCase):
    def test_zero3_gathers_exactly_one_parameter_at_a_time(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        params = list(model.named_parameters())
        for _, param in params:
            param.ds_id = id(param)

        gathered_batches = []

        class Gather:
            def __init__(self, parameters, enabled):
                self.parameters = list(parameters)
                self.enabled = enabled

            def __enter__(self):
                gathered_batches.append(self.parameters)

            def __exit__(self, *_args):
                return False

        with mock.patch.object(vllm_utils.deepspeed.zero, "GatheredParameters", Gather):
            exported = list(
                vllm_utils._iter_zero3_exported_tensors(
                    params, name_mapper=None, adapter=PassthroughWeightExportAdapter()
                )
            )

        self.assertEqual(len(gathered_batches), len(params))
        self.assertTrue(all(len(batch) == 1 for batch in gathered_batches))
        self.assertEqual([name for name, _ in exported], [name for name, _ in params])


class TestWeightSyncCoordination(unittest.TestCase):
    def test_actor_manager_is_paused_and_resumed_once(self):
        actor_manager = MagicMock()
        actor_manager.set_should_stop.remote.side_effect = ["pause", "resume"]

        with mock.patch.object(grpo_utils.ray, "get") as ray_get, grpo_utils.pause_actor_manager(actor_manager):
            pass

        self.assertEqual(actor_manager.set_should_stop.remote.call_args_list, [mock.call(True), mock.call(False)])
        self.assertEqual(ray_get.call_args_list, [mock.call("pause"), mock.call("resume")])

    def test_actor_manager_is_resumed_after_failure(self):
        actor_manager = MagicMock()

        with (
            mock.patch.object(grpo_utils.ray, "get"),
            self.assertRaisesRegex(RuntimeError, "sync failed"),
            grpo_utils.pause_actor_manager(actor_manager),
        ):
            raise RuntimeError("sync failed")

        self.assertEqual(actor_manager.set_should_stop.remote.call_args_list, [mock.call(True), mock.call(False)])


class TestMambaSpecCompatibilityPatch(unittest.TestCase):
    def test_patch_is_feature_gated_on_fixed_length_annotation(self):
        original_annotation = vllm_utils.MambaSpec.__annotations__["dtypes"]
        original_field_type = vllm_utils.MambaSpec.__dataclass_fields__["dtypes"].type
        try:
            vllm_utils.MambaSpec.__annotations__["dtypes"] = "tuple[torch.dtype]"
            vllm_utils.MambaSpec.__dataclass_fields__["dtypes"].type = "tuple[torch.dtype]"
            self.assertTrue(vllm_utils._patch_mamba_spec_dtypes_annotation())
            self.assertEqual(vllm_utils.MambaSpec.__annotations__["dtypes"], tuple[torch.dtype, ...])
            self.assertFalse(vllm_utils._patch_mamba_spec_dtypes_annotation())
        finally:
            vllm_utils.MambaSpec.__annotations__["dtypes"] = original_annotation
            vllm_utils.MambaSpec.__dataclass_fields__["dtypes"].type = original_field_type


if __name__ == "__main__":
    unittest.main()
