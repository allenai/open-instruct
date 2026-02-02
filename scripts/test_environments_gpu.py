#!/usr/bin/env python3
"""
GPU test script for RL environments with Ray actors.

Usage: python scripts/test_environments_gpu.py
"""

import asyncio
import logging
import sys

import ray

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_basic_envs():
    """Test environments without Ray."""
    from open_instruct.environments import ToolCall
    from open_instruct.environments.examples import CounterEnv

    env = CounterEnv(target=3)
    await env.reset()
    for _ in range(3):
        await env.step(ToolCall(name="increment", args={}))
    result = await env.step(ToolCall(name="submit", args={}))

    assert result.done and result.reward == 1.0
    logger.info("Basic env test passed")


async def test_env_pool():
    """Test EnvironmentPool with Ray actors."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    from open_instruct.environments import EnvironmentPool, ToolCall

    pool = EnvironmentPool(pool_size=4, env_name="counter", target=3)
    await pool.initialize()

    try:
        env = await pool.acquire()
        ray.get(env.reset.remote())

        for _ in range(3):
            ray.get(env.step.remote(ToolCall(name="increment", args={})))

        result = ray.get(env.step.remote(ToolCall(name="submit", args={})))
        assert result.done and result.reward == 1.0

        pool.release(env)
        logger.info("Environment pool test passed")
    finally:
        await pool.shutdown()


async def main():
    logger.info("Running environment tests...")

    await test_basic_envs()
    await test_env_pool()

    logger.info("All tests passed!")
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    finally:
        if ray.is_initialized():
            ray.shutdown()
