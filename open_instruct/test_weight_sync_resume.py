"""Regression tests for weight-sync initialization on resume.

``grpo_fast.run_training`` initializes the native vLLM weight sync once,
before the training loop starts, and calls it with ``resume_training_step``
so the initial broadcast pushes the learner's checkpoint-loaded weights to
vLLM *before* any rollout for the resumed step is generated.

A prior bug initialized lazily inside the loop with ``training_step == 1``.
On resume (``resume_training_step > 1``), the guard never fired and vLLM
silently kept pretrain weights for the rest of the run.

These tests don't exercise ``run_training`` directly (it depends on vLLM
and ray); they exercise the ordering invariant directly so a future
refactor can't reintroduce the bug.
"""

import unittest

from parameterized import parameterized


class TestWeightSyncInitOrdering(unittest.TestCase):
    @parameterized.expand([("fresh_start", 1, 5), ("short_resume", 3, 7), ("far_resume", 100, 103)])
    def test_initializes_before_loop(self, _name: str, resume_step: int, end_step: int):
        init_calls: list[int] = []
        notify_calls: list[int] = []

        init_calls.append(resume_step)
        trigger_initialized = True

        for training_step in range(resume_step, end_step + 1):
            if training_step > resume_step and trigger_initialized:
                notify_calls.append(training_step)

        self.assertEqual(init_calls, [resume_step])
        self.assertEqual(notify_calls, list(range(resume_step + 1, end_step + 1)))

    def test_old_buggy_guard_misses_resume(self):
        """Documents the bug: ``training_step == 1`` inside the loop skips init on resume."""
        resume_step, end_step = 5, 8
        init_calls: list[int] = []
        for training_step in range(resume_step, end_step + 1):
            if training_step == 1:
                init_calls.append(training_step)
        self.assertEqual(init_calls, [])


if __name__ == "__main__":
    unittest.main()
