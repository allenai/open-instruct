import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from open_instruct import grpo_fast


def _build_mock_model():
    save_model_remote = MagicMock(return_value="save-ok")
    launch_eval_remote = MagicMock(return_value="eval-ok")
    model = SimpleNamespace(
        save_model=SimpleNamespace(remote=save_model_remote),
        launch_ai2_evals_on_weka_wrapper=SimpleNamespace(remote=launch_eval_remote),
    )
    return model, save_model_remote, launch_eval_remote


def test_get_unique_final_output_dir_returns_original_when_missing(tmp_path):
    output_dir = tmp_path / "new-final-output"

    resolved = grpo_fast.get_unique_final_output_dir(str(output_dir))

    assert resolved == str(output_dir)


def test_get_unique_final_output_dir_appends_suffix_when_exists(tmp_path):
    output_dir = tmp_path / "existing-output"
    output_dir.mkdir()

    resolved = grpo_fast.get_unique_final_output_dir(str(output_dir))
    resolved_path = Path(resolved)

    assert resolved != str(output_dir)
    assert resolved_path.parent == output_dir.parent
    assert resolved_path.name.startswith(f"{output_dir.name}-final-")
    assert not resolved_path.exists()


def test_save_final_model_uses_unique_dir_for_save_and_eval():
    args = SimpleNamespace(
        output_dir="/tmp/existing-output",
        world_size=2,
        try_launch_beaker_eval_jobs_on_weka=True,
        hf_repo_revision="main",
    )
    policy_group = SimpleNamespace(models=[])
    model_0, model_0_save_remote, model_0_eval_remote = _build_mock_model()
    model_1, model_1_save_remote, model_1_eval_remote = _build_mock_model()
    policy_group.models.extend([model_0, model_1])
    target_dir = "/tmp/existing-output-final-unique"

    with (
        patch.object(grpo_fast, "get_unique_final_output_dir", return_value=target_dir),
        patch.object(grpo_fast, "Timer", return_value=contextlib.nullcontext()),
        patch.object(grpo_fast, "ray_get_with_progress", return_value=None),
        patch.object(grpo_fast, "is_beaker_job", return_value=True),
    ):
        grpo_fast.save_final_model(
            args=args,
            policy_group=policy_group,
            tokenizer=MagicMock(),
            training_step=10,
            wandb_url="https://wandb.example/run",
            chat_template_name="chat-template",
        )

    model_0_save_remote.assert_called_once()
    model_1_save_remote.assert_called_once()
    assert model_0_save_remote.call_args.args[0] == target_dir
    assert model_1_save_remote.call_args.args[0] == target_dir

    model_0_eval_remote.assert_called_once()
    model_1_eval_remote.assert_called_once()
    assert model_0_eval_remote.call_args.args[0] == target_dir
    assert model_1_eval_remote.call_args.args[0] == target_dir
