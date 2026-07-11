from unittest.mock import Mock, patch

from open_instruct import data_loader as data_loader_lib
from open_instruct.grpo_callbacks import DataPreparationActorCheckpointCallback


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
