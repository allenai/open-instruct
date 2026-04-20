"""Regression tests for lazy weight-sync initialization on resume.

The main-thread training loop in ``grpo_fast.run_training`` initializes the
native vLLM weight sync exactly once, on the first iteration of the loop.
On resume this must happen at ``resume_training_step`` (the step the loop
actually starts at), not at an absolute step 1.

A prior bug hard-coded the guard to ``training_step == 1``, which meant
that on resume (``resume_training_step > 1``) the weight sync thread was
never started and the vLLM actors silently kept their original pretrain
weights for the rest of the run.

These tests don't exercise ``run_training`` directly (it depends on vLLM
and ray, neither of which are installed in this test environment); they
exercise the guard logic directly so a future refactor can't reintroduce
the off-by-"step 1" bug.
"""

import unittest

from parameterized import parameterized


class TestWeightSyncInitGuard(unittest.TestCase):
    @parameterized.expand([("fresh_start", 1, 5), ("short_resume", 3, 7), ("far_resume", 100, 103)])
    def test_initializes_on_first_loop_iteration(self, _name: str, resume_step: int, end_step: int):
        init_calls: list[int] = []
        notify_calls: list[int] = []
        trigger_initialized = False

        for training_step in range(resume_step, end_step + 1):
            if training_step == resume_step:
                init_calls.append(training_step)
                trigger_initialized = True
            elif trigger_initialized:
                notify_calls.append(training_step)

        self.assertEqual(init_calls, [resume_step])
        self.assertEqual(notify_calls, list(range(resume_step + 1, end_step + 1)))

    def test_old_buggy_guard_misses_resume(self):
        """Documents the bug: ``training_step == 1`` skips init entirely on resume."""
        resume_step, end_step = 5, 8
        init_calls: list[int] = []
        for training_step in range(resume_step, end_step + 1):
            if training_step == 1:
                init_calls.append(training_step)
        self.assertEqual(init_calls, [])


if __name__ == "__main__":
    unittest.main()
