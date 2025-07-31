"""Simple test for ActorManager sync_weights functionality."""

import time
import unittest

import ray

from open_instruct.vllm_utils3 import ActorManager


class TestActorManager(unittest.TestCase):
    """Test ActorManager sync_weights functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not ray.is_initialized():
            ray.init()

    def tearDown(self):
        """Clean up after tests."""
        if ray.is_initialized():
            ray.shutdown()

    def test_actor_manager_sync_weights(self):
        """Test that ActorManager.sync_weights calls broadcast and manages flags correctly."""
        # Create actor manager
        actor_manager = ActorManager.remote()

        # Create mock policy models as Ray actors
        @ray.remote
        class MockModel:
            def __init__(self):
                self.broadcast_called = False

            def broadcast_to_vllm(self):
                self.broadcast_called = True

                time.sleep(0.1)  # Simulate some work
                return "broadcast_done"

            def was_broadcast_called(self):
                return self.broadcast_called

        mock_model1 = MockModel.remote()
        mock_model2 = MockModel.remote()

        policy_models = [mock_model1, mock_model2]

        # Verify initial state
        self.assertFalse(ray.get(actor_manager.should_update_weights.remote()))

        # Call sync_weights and wait for completion
        ray.get(actor_manager.sync_weights.remote(policy_models, training_step=100))

        # Verify broadcast was called on both models
        self.assertTrue(ray.get(mock_model1.was_broadcast_called.remote()))
        self.assertTrue(ray.get(mock_model2.was_broadcast_called.remote()))

        # Verify flag is reset after sync
        self.assertFalse(ray.get(actor_manager.should_update_weights.remote()))

    def test_set_should_update_weights(self):
        """Test the basic set_should_update_weights functionality."""
        actor_manager = ActorManager.remote()

        # Initially should be False
        self.assertFalse(ray.get(actor_manager.should_update_weights.remote()))

        # Set to True
        ray.get(actor_manager.set_should_update_weights.remote(True))
        self.assertTrue(ray.get(actor_manager.should_update_weights.remote()))

        # Set back to False
        ray.get(actor_manager.set_should_update_weights.remote(False))
        self.assertFalse(ray.get(actor_manager.should_update_weights.remote()))


if __name__ == "__main__":
    unittest.main()
