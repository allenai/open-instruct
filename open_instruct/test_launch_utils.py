import pathlib
import tempfile
import time
import unittest
from unittest import mock

from parameterized import parameterized

from open_instruct import launch_utils


def _setup_beaker_mocks(mock_beaker_from_env, mock_is_beaker_job, initial_description):
    """Shared mock setup for beaker tests."""
    mock_is_beaker_job.return_value = True

    mock_client = mock.MagicMock()
    mock_beaker_from_env.return_value = mock_client

    mock_workload = mock.MagicMock()
    mock_client.workload.get.return_value = mock_workload

    mock_spec = mock.MagicMock()
    mock_spec.description = initial_description
    mock_client.experiment.get_spec.return_value = mock_spec

    description_history = []

    def track_description(workload, description=None):
        if description is not None:
            description_history.append(description)

    mock_client.workload.update.side_effect = track_description

    return mock_client, mock_spec, description_history


class TestBeakerDescription(unittest.TestCase):
    """Test the beaker description update function."""

    @mock.patch("os.environ.get")
    @mock.patch("beaker.Beaker.from_env")
    @mock.patch("open_instruct.launch_utils.is_beaker_job")
    def test_description_does_not_accumulate(self, mock_is_beaker_job, mock_beaker_from_env, mock_environ_get):
        """Test that the description doesn't accumulate git info and wandb URLs on repeated calls."""
        env_values = {"BEAKER_WORKLOAD_ID": "test-id-123", "GIT_COMMIT": "abc123", "GIT_BRANCH": "main"}
        mock_environ_get.side_effect = lambda key, default=None: env_values.get(key, default)

        mock_client, mock_spec, description_history = _setup_beaker_mocks(
            mock_beaker_from_env, mock_is_beaker_job, "Beaker-Mason job."
        )

        wandb_url = "https://wandb.ai/ai2-llm/open_instruct_internal/runs/1f3ow3oh"
        start_time = time.time()

        original_descriptions = {}

        for step in [10, 20, 30]:
            launch_utils.maybe_update_beaker_description(
                current_step=step,
                total_steps=100,
                start_time=start_time,
                wandb_url=wandb_url,
                original_descriptions=original_descriptions,
            )
            if description_history:
                mock_spec.description = description_history[-1]

        self.assertEqual(len(description_history), 3)

        for i, desc in enumerate(description_history):
            git_commit_count = desc.count("git_commit:")
            git_branch_count = desc.count("git_branch:")
            wandb_count = desc.count(wandb_url)

            self.assertEqual(
                git_commit_count,
                1,
                f"Step {(i + 1) * 10}: git_commit should appear once, but appears {git_commit_count} times in: {desc}",
            )
            self.assertEqual(
                git_branch_count,
                1,
                f"Step {(i + 1) * 10}: git_branch should appear once, but appears {git_branch_count} times in: {desc}",
            )
            self.assertEqual(
                wandb_count,
                1,
                f"Step {(i + 1) * 10}: wandb URL should appear once, but appears {wandb_count} times in: {desc}",
            )

            self.assertIn("Beaker-Mason job.", desc)
            self.assertIn("git_commit: abc123", desc)
            self.assertIn("git_branch: main", desc)
            self.assertIn(wandb_url, desc)
            self.assertIn(f"% complete (step {(i + 1) * 10}/100)", desc)

    @mock.patch("os.environ.get")
    @mock.patch("beaker.Beaker.from_env")
    @mock.patch("open_instruct.launch_utils.is_beaker_job")
    def test_description_without_progress(self, mock_is_beaker_job, mock_beaker_from_env, mock_environ_get):
        """Test description updates without progress information."""
        env_values = {"BEAKER_WORKLOAD_ID": "test-id-123", "GIT_COMMIT": "def456", "GIT_BRANCH": "dev"}
        mock_environ_get.side_effect = lambda key, default=None: env_values.get(key, default)

        mock_client, mock_spec, description_history = _setup_beaker_mocks(
            mock_beaker_from_env, mock_is_beaker_job, "Initial job description"
        )

        original_descriptions = {}

        launch_utils.maybe_update_beaker_description(
            wandb_url="https://wandb.ai/team/project/runs/xyz789", original_descriptions=original_descriptions
        )

        self.assertEqual(len(description_history), 1)
        desc = description_history[0]

        self.assertIn("Initial job description", desc)
        self.assertIn("git_commit: def456", desc)
        self.assertIn("git_branch: dev", desc)
        self.assertIn("https://wandb.ai/team/project/runs/xyz789", desc)
        self.assertNotIn("% complete", desc)

    @mock.patch("os.environ.get")
    @mock.patch("beaker.Beaker.from_env")
    @mock.patch("open_instruct.launch_utils.is_beaker_job")
    def test_description_does_not_duplicate_on_restart(
        self, mock_is_beaker_job, mock_beaker_from_env, mock_environ_get
    ):
        """Test that description doesn't duplicate when job restarts (fresh original_descriptions dict)."""
        env_values = {"BEAKER_WORKLOAD_ID": "test-id-123", "GIT_COMMIT": "abc123", "GIT_BRANCH": "main"}
        mock_environ_get.side_effect = lambda key, default=None: env_values.get(key, default)

        previous_run_description = (
            "Single GPU on Beaker with tool use test script. "
            "git_commit: e6df3c9c git_branch: finbarr/async-reward "
            "https://wandb.ai/ai2-llm/open_instruct_internal/runs/n53oxnzb "
            "[5.0% complete (step 1/20), eta 0m]"
        )
        mock_client, mock_spec, description_history = _setup_beaker_mocks(
            mock_beaker_from_env, mock_is_beaker_job, previous_run_description
        )

        wandb_url = "https://wandb.ai/ai2-llm/open_instruct_internal/runs/n53oxnzb"
        original_descriptions = {}

        launch_utils.maybe_update_beaker_description(
            current_step=2,
            total_steps=20,
            start_time=time.time(),
            wandb_url=wandb_url,
            original_descriptions=original_descriptions,
        )

        self.assertEqual(len(description_history), 1)
        desc = description_history[0]

        git_commit_count = desc.count("git_commit:")
        git_branch_count = desc.count("git_branch:")
        wandb_count = desc.count("wandb.ai")

        self.assertEqual(
            git_commit_count, 1, f"git_commit should appear once, but appears {git_commit_count} times in: {desc}"
        )
        self.assertEqual(
            git_branch_count, 1, f"git_branch should appear once, but appears {git_branch_count} times in: {desc}"
        )
        self.assertEqual(wandb_count, 1, f"wandb URL should appear once, but appears {wandb_count} times in: {desc}")
        self.assertIn("Single GPU on Beaker with tool use test script.", desc)


class TestDownloadFromGsBucket(unittest.TestCase):
    def test_download_from_gs_bucket(self):
        src_paths = ["gs://bucket/data1", "gs://bucket/data2"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "downloads"
            captured_cmd: dict[str, list[str]] = {}

            def mock_live_subprocess_output(cmd):
                captured_cmd["cmd"] = cmd

            with mock.patch.object(launch_utils, "live_subprocess_output", side_effect=mock_live_subprocess_output):
                launch_utils.download_from_gs_bucket(src_paths=src_paths, dest_path=str(dest_path))

            expected_cmd = [
                "gsutil",
                "-o",
                "GSUtil:parallel_thread_count=1",
                "-o",
                "GSUtil:sliced_object_download_threshold=150",
                "-m",
                "cp",
                "-r",
                *src_paths,
                str(dest_path),
            ]

            self.assertEqual(captured_cmd["cmd"], expected_cmd)
            self.assertTrue(dest_path.exists())


class TestWandbUrlToRunPath(unittest.TestCase):
    @parameterized.expand(
        [
            ("https://wandb.ai/org/project/runs/runid", "org/project/runid"),
            (
                "https://wandb.ai/ai2-llm/open_instruct_internal/runs/5nigq0mz",
                "ai2-llm/open_instruct_internal/5nigq0mz",
            ),
            (
                "https://wandb.ai/ai2-llm/open_instruct_internal/runs/vjyp36sp",
                "ai2-llm/open_instruct_internal/vjyp36sp",
            ),
        ]
    )
    def test_wandb_url_to_run_path(self, url: str, expected_run_path: str):
        self.assertEqual(launch_utils.wandb_url_to_run_path(url), expected_run_path)


if __name__ == "__main__":
    unittest.main()
