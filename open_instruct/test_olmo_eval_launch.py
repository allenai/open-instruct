import unittest

from open_instruct.olmo_eval_launch import (
    OlmoEvalLaunchConfig,
    build_olmo_eval_beaker_launch_command,
    default_olmo_eval_experiment_name,
    resolve_olmo_eval_model_path,
)


class TestOlmoEvalLaunch(unittest.TestCase):
    def test_resolve_olmo_eval_model_path_strips_trailing_slash(self):
        self.assertEqual(
            resolve_olmo_eval_model_path("/weka/oe-adapt-default/user/model/step_100/"),
            "/weka/oe-adapt-default/user/model/step_100",
        )

    def test_resolve_olmo_eval_model_path_rejects_beaker_dataset_id(self):
        with self.assertRaises(ValueError):
            resolve_olmo_eval_model_path("01KPTSPMHGEZVYCDNR0XBVJCGZ")

    def test_build_command_uses_checkpoint_path_and_config(self):
        config = OlmoEvalLaunchConfig(
            olmo_eval_tasks=["math:posttrain:dev"],
            olmo_eval_cluster="h100",
            olmo_eval_groups=["my-grpo-run"],
            olmo_eval_priority="urgent",
            olmo_eval_workspace="ai2/open-instruct-dev",
            olmo_eval_budget="ai2/oe-other",
        )
        cmd = build_olmo_eval_beaker_launch_command(
            "/weka/oe-adapt-default/user/model/step_100", config, experiment_name="my-run_step_100"
        )
        self.assertEqual(cmd[0:3], ["olmo-eval", "beaker", "launch"])
        self.assertIn("-m", cmd)
        self.assertIn("/weka/oe-adapt-default/user/model/step_100", cmd)
        self.assertIn("math:posttrain:dev", cmd)
        self.assertIn("h100", cmd)
        self.assertIn("ai2/open-instruct-dev", cmd)
        self.assertIn("urgent", cmd)
        self.assertIn("my-grpo-run", cmd)
        self.assertIn("ai2/oe-other", cmd)
        self.assertIn("--yes", cmd)
        self.assertIn("--no-follow", cmd)
        self.assertIn("my-run_step_100", cmd)

    def test_default_experiment_name_includes_step(self):
        self.assertEqual(default_olmo_eval_experiment_name("my-run", 100), "my-run_step_100")
        self.assertEqual(default_olmo_eval_experiment_name("my-run", None), "my-run")


if __name__ == "__main__":
    unittest.main()
