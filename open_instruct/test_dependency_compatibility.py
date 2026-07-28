"""Focused tests for dependency APIs exercised by Open Instruct."""

import os
import pathlib
import subprocess
import unittest

import torch
from datasets import Dataset

GPU_TEST_SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "test" / "run_gpu_pytest.sh"
QWEN3_MOE_PROFILE_SCRIPT = (
    pathlib.Path(__file__).parents[1] / "scripts" / "train" / "qwen" / "qwen3_30b_a3b_dapo_math_profile.sh"
)


class TestDependencyCompatibility(unittest.TestCase):
    def _run_gpu_test_script_function(
        self, function_name: str, argument: str, *, cuda_override: str | None = None
    ) -> str:
        env = os.environ.copy()
        if cuda_override is None:
            env.pop("OPEN_INSTRUCT_CUDA_VERSION", None)
        else:
            env["OPEN_INSTRUCT_CUDA_VERSION"] = cuda_override
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; "$2" "$3"', "_", str(GPU_TEST_SCRIPT), function_name, argument],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return result.stdout.strip()

    def test_datasets_torch_format(self):
        dataset = Dataset.from_dict({"tokens": [[1, 2], [3]]}).with_format("torch")

        self.assertEqual(dataset[0]["tokens"].tolist(), [1, 2])
        self.assertEqual(dataset[1]["tokens"].tolist(), [3])

    def test_gpu_launcher_recognizes_cuda_image_aliases(self):
        image_versions = {
            "user/open-instruct-integration-test-branch-cuda12": "12",
            "user/open-instruct-integration-test-branch-cuda13": "13",
            "user/open_instruct_auto_cuda13": "13",
            "open-instruct-auto-cuda13:abcdef": "13",
            "user/open-instruct-gpu-tests-abcdef-cuda13": "13",
        }

        for image_name, expected_version in image_versions.items():
            with self.subTest(image_name=image_name):
                self.assertEqual(
                    self._run_gpu_test_script_function("cuda_version_for_image", image_name), expected_version
                )

    def test_gpu_launcher_defaults_unversioned_images_to_cuda12(self):
        self.assertEqual(
            self._run_gpu_test_script_function("cuda_version_for_image", "user/open-instruct-integration-test"), "12"
        )

    def test_gpu_launcher_accepts_explicit_cuda13_for_unversioned_images(self):
        self.assertEqual(
            self._run_gpu_test_script_function(
                "cuda_version_for_image", "user/open-instruct-integration-test", cuda_override="13"
            ),
            "13",
        )

    def test_gpu_launcher_routes_cuda_variants_to_compatible_clusters(self):
        self.assertEqual(
            self._run_gpu_test_script_function("cuda_test_clusters", "12"), "ai2/jupiter ai2/ceres ai2/saturn"
        )
        self.assertEqual(self._run_gpu_test_script_function("cuda_test_clusters", "13"), "ai2/holmes")

    def test_container_cuda_variant_matches_torch_build(self):
        expected_cuda = os.environ.get("OPEN_INSTRUCT_CUDA_VERSION")
        if expected_cuda is None:
            self.skipTest("not running inside a versioned CUDA image")

        self.assertIsNotNone(torch.version.cuda)
        self.assertEqual(torch.version.cuda.split(".", maxsplit=1)[0], expected_cuda)

    def test_qwen3_moe_hardware_profiles(self):
        profiles = {"12": "ai2/jupiter|4|2|true", "13": "ai2/holmes|8|1|false"}

        for cuda_version, expected_profile in profiles.items():
            with self.subTest(cuda_version=cuda_version):
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$1"; qwen3_30b_a3b_hardware_profile "$2"',
                        "_",
                        str(QWEN3_MOE_PROFILE_SCRIPT),
                        cuda_version,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.stdout.strip(), expected_profile)

    def test_qwen3_moe_cuda13_image_selects_cuda13_profile(self):
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; qwen3_30b_a3b_cuda_version_for_image "$2"',
                "_",
                str(QWEN3_MOE_PROFILE_SCRIPT),
                "user/open-instruct-integration-test-branch-cuda13",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "13")


if __name__ == "__main__":
    unittest.main()
