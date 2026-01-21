# Contributing to Open Instruct

Thank you for your interest in contributing to Open Instruct!

## CI for External Contributors (Fork PRs)

When you submit a pull request from a fork, some CI checks behave differently due to GitHub's security restrictions on secrets:

### GPU Tests

GPU tests require access to Beaker (our internal compute platform) and are **automatically skipped** for fork PRs. You'll see a message like:

```
Skipping GPU tests for fork PR
This PR is from a fork, and secrets are not available.
GPU tests will run automatically when this PR enters the merge queue.
```

This is expected behavior. When your PR is approved and enters the merge queue, GPU tests will run automatically with full access to secrets.

### Unit Tests

Unit tests run normally for fork PRs. All test models and datasets are publicly available.

## CI for Internal Contributors

### GPU_TESTS Override (Internal PRs Only)

For internal PRs, you can skip running GPU tests by providing a link to an existing successful Beaker experiment in your PR description. This is useful when you've already run the tests locally or want to reuse results from a previous run.

**Format** (must be a markdown link):
```
GPU_TESTS=[EXPERIMENT_ID](https://beaker.org/ex/EXPERIMENT_ID)
```

**Example**:
```
GPU_TESTS=[01KFGG2Q8XX0VHTP8QNYBAB3C9](https://beaker.org/orgs/ai2/workspaces/open-instruct-dev/experiments/01KFGG2Q8XX0VHTP8QNYBAB3C9)
```

**Requirements for the override experiment**:
- The experiment description must contain "GPU tests"
- The experiment must have exit code 0 (success)

### GPU_TESTS Bypass (Internal PRs Only)

For changes that don't affect GPU functionality (e.g., documentation, CI config, minor refactors), you can bypass GPU tests entirely by adding to your PR description:

```
GPU_TESTS=bypass
```

**Warning**: Use this sparingly. Only bypass GPU tests when you are confident the changes cannot affect GPU-related code paths. When in doubt, let the tests run.

## Running Tests Locally

```bash
# Run unit tests
uv run pytest

# Run linter and formatter
make style && make quality
```
