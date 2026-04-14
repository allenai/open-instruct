Please launch the GPU tests manually on Beaker with:

./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh

Monitor the experiment using the monitor-experiment skill. When the test has passed, update the PR body with the update-pr-body skill to add the experiment ID at the very end of the PR body in this format: GPU_TESTS=[exp_id](beaker_url).
