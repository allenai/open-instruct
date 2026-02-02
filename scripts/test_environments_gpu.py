#!/usr/bin/env python3
"""
GPU test script for RL environments.

This script tests environment functionality with Ray actors on GPU nodes.
It can be run standalone or as part of the CI pipeline.

Usage:
    python scripts/test_environments_gpu.py

Requirements:
    - Ray installed and initialized
    - GPU available (for full test)
"""

import asyncio
import logging
import sys
import time

import ray

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def test_counter_env_basic():
    """Test basic CounterEnv functionality without Ray."""
    logger.info("Testing CounterEnv basic functionality...")

    from open_instruct.environments import ToolCall
    from open_instruct.environments.examples import CounterEnv

    env = CounterEnv(target=3)

    # Reset
    result = await env.reset()
    assert "0" in result.observation, f"Expected '0' in observation: {result.observation}"
    assert len(result.tools) == 3, f"Expected 3 tools, got {len(result.tools)}"
    logger.info(f"  Reset: {result.observation}")

    # Increment to target
    for i in range(3):
        step = await env.step(ToolCall(name="increment", args={}))
        logger.info(f"  Step {i+1}: {step.observation}, reward={step.reward}")

    # Submit
    step = await env.step(ToolCall(name="submit", args={}))
    assert step.done, "Expected done=True after submit"
    assert step.reward == 1.0, f"Expected reward=1.0, got {step.reward}"
    logger.info(f"  Submit: {step.observation}, reward={step.reward}")

    # Check metrics
    metrics = env.get_metrics()
    assert metrics["reached_target"] == 1.0, f"Expected reached_target=1.0, got {metrics}"
    logger.info(f"  Metrics: {metrics}")

    logger.info("CounterEnv basic test passed!")


async def test_guess_number_env_basic():
    """Test basic GuessNumberEnv functionality without Ray."""
    logger.info("Testing GuessNumberEnv basic functionality...")

    from open_instruct.environments import ToolCall
    from open_instruct.environments.examples import GuessNumberEnv

    env = GuessNumberEnv(min_val=1, max_val=10)

    # Reset with known secret
    result = await env.reset(task_id="5")
    assert env._secret == 5, f"Expected secret=5, got {env._secret}"
    logger.info(f"  Reset: {result.observation}")

    # Binary search
    guesses = [5]  # We know the answer
    for guess in guesses:
        step = await env.step(ToolCall(name="guess", args={"number": guess}))
        logger.info(f"  Guess {guess}: {step.observation}")

        if step.done:
            assert step.reward == 1.0, f"Expected reward=1.0, got {step.reward}"
            break

    logger.info("GuessNumberEnv basic test passed!")


async def test_environment_pool():
    """Test EnvironmentPool with Ray actors."""
    logger.info("Testing EnvironmentPool with Ray actors...")

    # Initialize Ray if not already
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    from open_instruct.environments import EnvironmentPool, ResetResult, StepResult, ToolCall

    # Create pool
    pool = EnvironmentPool(
        pool_size=4,
        env_name="counter",
        target=5,
    )

    try:
        # Initialize
        await pool.initialize()
        logger.info(f"  Pool initialized with {len(pool)} actors")

        # Acquire and test
        env_actor = await pool.acquire()
        logger.info("  Acquired environment actor")

        # Reset - ray.get() the ObjectRef
        reset_ref = env_actor.reset.remote()
        result = ray.get(reset_ref)
        assert isinstance(result, ResetResult), f"Expected ResetResult, got {type(result)}"
        logger.info(f"  Reset result: {result.observation[:50]}...")

        # Step - ray.get() the ObjectRef
        step_ref = env_actor.step.remote(ToolCall(name="increment", args={}))
        step_result = ray.get(step_ref)
        assert isinstance(step_result, StepResult), f"Expected StepResult, got {type(step_result)}"
        logger.info(f"  Step result: {step_result.observation}")

        # Release
        pool.release(env_actor)
        logger.info("  Released environment actor")

    finally:
        # Shutdown
        await pool.shutdown()
        logger.info("  Pool shutdown complete")

    logger.info("EnvironmentPool test passed!")


async def test_environment_verifiers():
    """Test environment verifiers."""
    logger.info("Testing environment verifiers...")

    from open_instruct.environments import EnvironmentState
    from open_instruct.ground_truth_utils import LastRewardEnvVerifier, SumRewardEnvVerifier

    # Test LastRewardEnvVerifier
    last_verifier = LastRewardEnvVerifier()
    state = EnvironmentState(rewards=[0.1, 0.2, 0.8])
    result = last_verifier([], "", state)
    assert result.score == 0.8, f"Expected 0.8, got {result.score}"
    logger.info(f"  LastRewardEnvVerifier: {result.score}")

    # Test SumRewardEnvVerifier
    sum_verifier = SumRewardEnvVerifier()
    state = EnvironmentState(rewards=[0.1, 0.2, 0.3])
    result = sum_verifier([], "", state)
    assert abs(result.score - 0.6) < 0.001, f"Expected 0.6, got {result.score}"
    logger.info(f"  SumRewardEnvVerifier: {result.score}")

    logger.info("Environment verifiers test passed!")


async def test_parallel_rollouts():
    """Test parallel environment rollouts."""
    logger.info("Testing parallel rollouts...")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    from open_instruct.environments import EnvironmentPool, ToolCall

    pool = EnvironmentPool(
        pool_size=8,
        env_name="counter",
        target=3,
    )

    try:
        await pool.initialize()

        # Run parallel rollouts
        async def run_episode(episode_id: int):
            env = await pool.acquire()
            try:
                # Reset
                ray.get(env.reset.remote())

                # Run steps
                total_reward = 0.0
                for _ in range(3):
                    result = ray.get(env.step.remote(ToolCall(name="increment", args={})))
                    total_reward += result.reward

                # Submit
                result = ray.get(env.step.remote(ToolCall(name="submit", args={})))
                total_reward += result.reward

                return episode_id, total_reward, result.done
            finally:
                pool.release(env)

        # Run 8 parallel episodes
        start = time.time()
        tasks = [run_episode(i) for i in range(8)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        success_count = sum(1 for _, reward, done in results if done and reward > 0)
        logger.info(f"  Completed {len(results)} episodes in {elapsed:.2f}s")
        logger.info(f"  Success rate: {success_count}/{len(results)}")

        assert success_count == 8, f"Expected 8 successes, got {success_count}"

    finally:
        await pool.shutdown()

    logger.info("Parallel rollouts test passed!")


async def test_env_registry():
    """Test environment registry."""
    logger.info("Testing environment registry...")

    from open_instruct.environments import ENV_REGISTRY, get_env_class

    # Check registered envs
    expected = ["counter", "guess_number", "sandbox", "openenv", "openenv_repl"]
    for name in expected:
        assert name in ENV_REGISTRY, f"Expected '{name}' in registry"
        logger.info(f"  Found: {name}")

    # Test get_env_class
    cls = get_env_class(env_name="counter")
    assert cls.__name__ == "CounterEnv"
    logger.info("  get_env_class works")

    logger.info("Environment registry test passed!")


async def run_all_tests():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("RL Environment GPU Tests")
    logger.info("=" * 60)

    tests = [
        ("Counter Env Basic", test_counter_env_basic),
        ("Guess Number Env Basic", test_guess_number_env_basic),
        ("Environment Registry", test_env_registry),
        ("Environment Verifiers", test_environment_verifiers),
        ("Environment Pool", test_environment_pool),
        ("Parallel Rollouts", test_parallel_rollouts),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        logger.info("")
        logger.info(f"Running: {name}")
        logger.info("-" * 40)
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            logger.error(f"FAILED: {e}")
            failed += 1
            import traceback

            traceback.print_exc()

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    return failed == 0


def main():
    """Main entry point."""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
