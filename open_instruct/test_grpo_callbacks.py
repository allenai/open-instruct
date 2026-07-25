import unittest
from unittest.mock import Mock, patch

from open_instruct import data_loader as data_loader_lib
from open_instruct.grpo_callbacks import DataPreparationActorCheckpointCallback, olmo_core_to_hf_name


class TestDataPreparationActorCheckpointCallback:
    def test_only_rank_zero_saves_actor_state(self):
        callback = DataPreparationActorCheckpointCallback()

        with (
            patch("open_instruct.grpo_callbacks.dist_utils.get_rank", return_value=1),
            patch("open_instruct.grpo_callbacks.ray.get_actor") as get_actor,
        ):
            assert callback.state_dict() == {}
            get_actor.assert_not_called()

    def test_rank_zero_saves_actor_state(self):
        callback = DataPreparationActorCheckpointCallback()
        actor = Mock()
        actor.get_state.remote.return_value = "state-ref"

        with (
            patch("open_instruct.grpo_callbacks.dist_utils.get_rank", return_value=0),
            patch("open_instruct.grpo_callbacks.ray.get_actor", return_value=actor) as get_actor,
            patch("open_instruct.grpo_callbacks.ray.get", return_value={"training_step": 100}) as ray_get,
        ):
            assert callback.state_dict() == {"data_prep_state": {"training_step": 100}}

        get_actor.assert_called_once_with(data_loader_lib.DATA_PREP_ACTOR_NAME)
        actor.get_state.remote.assert_called_once_with()
        ray_get.assert_called_once_with("state-ref")

    def test_only_rank_zero_restores_actor_state(self):
        callback = DataPreparationActorCheckpointCallback()
        state_dict = {"data_prep_state": {"training_step": 100}}

        with (
            patch("open_instruct.grpo_callbacks.dist_utils.get_rank", return_value=1),
            patch("open_instruct.grpo_callbacks.ray.get_actor") as get_actor,
        ):
            callback.load_state_dict(state_dict)
            get_actor.assert_not_called()

    def test_rank_zero_restores_actor_state(self):
        callback = DataPreparationActorCheckpointCallback()
        actor = Mock()
        actor.set_state.remote.return_value = "restore-ref"
        state = {"training_step": 100}

        with (
            patch("open_instruct.grpo_callbacks.dist_utils.get_rank", return_value=0),
            patch("open_instruct.grpo_callbacks.ray.get_actor", return_value=actor) as get_actor,
            patch("open_instruct.grpo_callbacks.ray.get") as ray_get,
        ):
            callback.load_state_dict({"data_prep_state": state})

        get_actor.assert_called_once_with(data_loader_lib.DATA_PREP_ACTOR_NAME)
        actor.set_state.remote.assert_called_once_with(state)
        ray_get.assert_called_once_with("restore-ref")


class OlmoCoreToHfNameTest(unittest.TestCase):
    """See open_instruct/olmo_core_utils.py's QWEN2_STYLE_HF_MODEL_TYPES docstring for why
    qwen2's w_out.bias needs special handling here: it's a synthetic param (olmo-core's
    AttentionConfig.bias applies uniformly to all four attention projections, but Qwen2 has no
    o_proj bias) with no destination in vLLM's HF-format model."""

    def test_top_level_names(self):
        self.assertEqual(olmo_core_to_hf_name("embeddings.weight"), "model.embed_tokens.weight")
        self.assertEqual(olmo_core_to_hf_name("lm_head.norm.weight"), "model.norm.weight")
        self.assertEqual(olmo_core_to_hf_name("lm_head.w_out.weight"), "lm_head.weight")

    def test_qkv_weight_and_bias_mapped(self):
        for proj in ("q", "k", "v"):
            self.assertEqual(
                olmo_core_to_hf_name(f"blocks.3.attention.w_{proj}.weight"),
                f"model.layers.3.self_attn.{proj}_proj.weight",
            )
            self.assertEqual(
                olmo_core_to_hf_name(f"blocks.3.attention.w_{proj}.bias"), f"model.layers.3.self_attn.{proj}_proj.bias"
            )

    def test_w_out_bias_has_no_destination(self):
        self.assertIsNone(olmo_core_to_hf_name("blocks.0.attention.w_out.bias"))
        self.assertIsNone(olmo_core_to_hf_name("blocks.27.attention.w_out.bias"))
        # w_out.weight is unaffected -- only the synthetic bias has no HF counterpart.
        self.assertEqual(
            olmo_core_to_hf_name("blocks.0.attention.w_out.weight"), "model.layers.0.self_attn.o_proj.weight"
        )

    def test_unrecognized_name_passes_through_unchanged(self):
        self.assertEqual(olmo_core_to_hf_name("some.unmapped.name"), "some.unmapped.name")
