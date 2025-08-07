"""Simple test for ActorManager functionality."""

import time
import unittest

import ray

from open_instruct.vllm_utils3 import ActorManager


class TestActorManager(unittest.TestCase):
    """Test ActorManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not ray.is_initialized():
            ray.init()

    def tearDown(self):
        """Clean up after tests."""
        if ray.is_initialized():
            ray.shutdown()

    def test_actor_manager_weight_sync_coordination(self):
        """Test that ActorManager can coordinate weight sync via should_stop flag."""
        # Create actor manager
        actor_manager = ActorManager.remote()

        # Verify initial state
        self.assertFalse(ray.get(actor_manager.should_stop.remote()))

        # Test the coordination pattern used in sync_weights_and_prepare_prompts
        # Set should_stop to True (simulating weight sync start)
        ray.get(actor_manager.set_should_stop.remote(True))
        self.assertTrue(ray.get(actor_manager.should_stop.remote()))

        # Simulate weight sync happening
        time.sleep(0.1)

        # Set should_stop back to False (simulating weight sync complete)
        ray.get(actor_manager.set_should_stop.remote(False))
        self.assertFalse(ray.get(actor_manager.should_stop.remote()))

    def test_set_should_stop(self):
        """Test the basic set_should_stop functionality."""
        actor_manager = ActorManager.remote()

        # Initially should be False
        self.assertFalse(ray.get(actor_manager.should_stop.remote()))

        # Set to True
        ray.get(actor_manager.set_should_stop.remote(True))
        self.assertTrue(ray.get(actor_manager.should_stop.remote()))

        # Set back to False
        ray.get(actor_manager.set_should_stop.remote(False))
        self.assertFalse(ray.get(actor_manager.should_stop.remote()))


if __name__ == "__main__":
    unittest.main()
