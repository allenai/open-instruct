"""
Tests for the RL environments module.
"""

import pytest

from open_instruct.data_types import EnvironmentState
from open_instruct.environments.adapter import EnvironmentAdapter, EnvironmentPool
from open_instruct.environments.base import RLEnvironment, StepResult


class MockEnv(RLEnvironment):
    """Simple mock environment for testing."""

    def __init__(self, max_steps: int = 3, reward_on_done: float = 1.0):
        self.max_steps = max_steps
        self.reward_on_done = reward_on_done
        self.step_count = 0

    def reset(self) -> StepResult:
        self.step_count = 0
        return StepResult(observation="Start", reward=0.0, done=False)

    def step(self, action: dict) -> StepResult:
        self.step_count += 1
        done = self.step_count >= self.max_steps
        reward = self.reward_on_done if done else 0.0
        return StepResult(
            observation=f"Step {self.step_count}",
            reward=reward,
            done=done,
        )

    def close(self):
        pass


class TestStepResult:
    def test_step_result_creation(self):
        result = StepResult(observation="test", reward=0.5, done=False)
        assert result.observation == "test"
        assert result.reward == 0.5
        assert result.done is False
        assert result.info == {}

    def test_step_result_with_info(self):
        result = StepResult(observation="test", reward=1.0, done=True, info={"key": "value"})
        assert result.info == {"key": "value"}


class TestEnvironmentState:
    def test_final_reward_empty(self):
        state = EnvironmentState(env_name="test")
        assert state.final_reward == 0.0

    def test_final_reward_with_rewards(self):
        state = EnvironmentState(env_name="test", rewards=[0.0, 0.0, 1.0])
        assert state.final_reward == 1.0

    def test_final_reward_single(self):
        state = EnvironmentState(env_name="test", rewards=[0.5])
        assert state.final_reward == 0.5


class TestEnvironmentAdapter:
    def test_setup_and_step(self):
        import asyncio

        async def _test():
            adapter = EnvironmentAdapter(lambda **kwargs: MockEnv(max_steps=2, **kwargs))
            info = {"env_config": {}}

            # Setup
            result = await adapter.setup(info)
            assert result.observation == "Start"
            assert result.reward == 0.0
            assert adapter.done is False

            # Step 1
            result = await adapter.step(action={"move": "forward"})
            assert result.observation == "Step 1"
            assert result.reward == 0.0
            assert adapter.done is False

            # Step 2 (final)
            result = await adapter.step(action={"move": "forward"})
            assert result.observation == "Step 2"
            assert result.reward == 1.0
            assert adapter.done is True

            # Check state
            state = adapter.get_state()
            assert state["step_count"] == 2
            assert state["rewards"] == [0.0, 0.0, 1.0]
            assert state["done"] is True

        asyncio.run(_test())

    def test_cleanup(self):
        import asyncio

        async def _test():
            adapter = EnvironmentAdapter(MockEnv)
            await adapter.setup({"env_config": {}})
            assert adapter.env is not None
            adapter.cleanup()
            assert adapter.env is None

        asyncio.run(_test())


class TestEnvironmentPool:
    def test_pool_acquire_and_release(self):
        import asyncio

        async def _test():
            pool = EnvironmentPool(MockEnv, pool_size=2)
            await pool.initialize()

            # Acquire first
            result1 = await pool.acquire("req1", {"env_config": {}})
            assert result1.observation == "Start"

            # Acquire second
            result2 = await pool.acquire("req2", {"env_config": {}})
            assert result2.observation == "Start"

            # Step first
            step_result = await pool.step("req1", action={})
            assert step_result.observation == "Step 1"

            # Check states
            assert not pool.is_done("req1")
            assert not pool.is_done("req2")

            # Release
            await pool.release("req1")
            await pool.release("req2")

        asyncio.run(_test())


# Note: EnvVerifier tests are in the main test suite which requires heavy torch imports
# They are tested via CI, not locally on machines without GPU support
