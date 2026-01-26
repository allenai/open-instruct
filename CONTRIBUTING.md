# Contributing to Open Instruct

Thank you for your interest in contributing to Open Instruct!

## External contributors

### CI (Fork PRs)

When you submit a pull request from a fork, some CI checks behave differently due to GitHub's security restrictions on secrets:

#### GPU Tests

GPU tests require access to Beaker (our internal compute platform) and are **automatically skipped** for fork PRs. You'll see a message like:

```
Skipping GPU tests for fork PR
This PR is from a fork, and secrets are not available.
GPU tests will run automatically when this PR enters the merge queue.
```

This is expected behavior. A maintainer will manually run the GPU tests.

## Internal Contributors

Please name your branch `username/branch-description`. E.g. `finbarr/update-vllm-version`. 

### GPU_TESTS Override (Internal PRs Only)

For internal PRs, you can skip running GPU tests by providing a link to an existing successful Beaker experiment in your PR description. This is useful when you've already run the tests locally or want to reuse results from a previous run. The format is `GPU_TESTS=[EXPERIMENT_ID](https://beaker.org/ex/EXPERIMENT_ID)`. 

You can launch the GPU tests manually with `./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh`.

### GPU_TESTS Bypass (Internal PRs Only)

For changes that don't affect GPU functionality (e.g., documentation, CI config, minor refactors), you can bypass GPU tests entirely by adding to your PR description:

```
GPU_TESTS=bypass
```

**Warning**: Use this sparingly. Only bypass GPU tests when you are confident the changes cannot affect GPU-related code paths. When in doubt, let the tests run.
