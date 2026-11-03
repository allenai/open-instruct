"""ReMind-style FIFO review with a continuous prompt budget (not update bursts)."""

import copy
import random
from collections import deque
from fractions import Fraction

from open_instruct.miles.configuration import validation
from open_instruct.miles.rollout.prompt_provider import REVIEW, SELECTION_ID, ShuffledPromptProvider


class ReMindPromptProvider:
    """Review previously mastered prompts using fresh generations and binary rewards.

    Spend one slot in every 50 on review by default; unused slots return to the
    ordinary shuffled stream. Selection is independent of request chunk size.
    Feedback uses native inference records enriched with the selection ID carried
    by each prompt. The caller serializes selection, feedback, and checkpoints.
    The dataset must remain fixed because review history refers to its row IDs.
    """

    requires_feedback = True

    def __init__(self, get_dataset, *, shuffle, seed, review_fraction=0.02, enqueue_probability=0.25):
        dataset = get_dataset()
        if dataset is None or not len(dataset):
            raise ValueError("ReMind requires a nonempty global prompt dataset")
        validation.number(review_fraction, "review_fraction", maximum=1, exclusive_min=True)
        validation.number(enqueue_probability, "enqueue_probability", maximum=1)
        validation.integer(seed, "seed", minimum=0)
        self._get_dataset = get_dataset
        self._dataset = dataset
        self.base = ShuffledPromptProvider(self._review_dataset, shuffle=shuffle)
        self.samples = dataset.origin_samples
        self.row_ids = {id(sample): i for i, sample in enumerate(self.samples)}
        self.config = dict(seed=seed, review_fraction=review_fraction, enqueue_probability=enqueue_probability)
        self.fraction = Fraction(str(review_fraction))
        self.rng = random.Random(seed)
        self.draws = 0
        self.queue = deque()
        self.tracked = set()  # Queued or currently being reviewed; never enqueue twice.
        self.pending = {}  # Selection ID -> (dataset row, review).

    def _review_dataset(self):
        dataset = self._get_dataset()
        if dataset is not self._dataset or dataset.origin_samples is not self.samples:
            raise ValueError("ReMind dataset cannot be replaced; construct a new provider for the new dataset")
        return dataset

    def next_samples(self, n):
        if type(n) is not int or n < 0:
            raise ValueError("n must be a nonnegative integer")
        self._review_dataset()
        result = []
        numerator, denominator = self.fraction.numerator, self.fraction.denominator
        for _ in range(n):
            slot = (self.draws + 1) * numerator // denominator > self.draws * numerator // denominator
            review = bool(slot and self.queue)
            row = self.queue.popleft() if review else self.row_ids[id(self.base.next_samples(1)[0])]
            sample = copy.copy(self.samples[row])
            sample.metadata = {**(sample.metadata or {}), SELECTION_ID: self.draws, REVIEW: review}
            self.pending[self.draws] = (row, review)
            self.draws += 1
            result.append(sample)
        return result

    def feedback(self, records):
        self._review_dataset()
        for record in records:
            if record.get("kind") == "disposition" and record.get("disposition") == "acknowledged":
                # The runtime can discard a group before recording (e.g. mixed-policy
                # rejection). Release unobserved selections without losing reviews.
                selection = self.pending.pop(record.get(SELECTION_ID), None)
                if selection is not None and selection[1]:
                    self.queue.append(selection[0])
                continue
            if record.get("kind") != "group" or record.get("filter_decision") == "aborted":
                continue  # Aborted groups keep their selection ID when retried.
            selection = self.pending.pop(record.get(SELECTION_ID), None)
            if selection is None:
                # First scored generation counts; regeneration after staleness is a duplicate.
                continue
            row, review = selection
            responses = record.get("responses") or []
            rewards = []
            for response in responses:
                reward = response.get("reward")
                if isinstance(reward, dict):
                    reward = reward.get(record.get("reward_key"))
                valid = (response.get("validity") or {}).get("valid") is True
                # Deliberate truncation-to-zero is a failure; service errors are not.
                if response.get("truncated") and record.get("reward_zero_truncated") and reward == 0:
                    valid = True
                if response.get("status") not in ("completed", "truncated") or not valid or reward not in (0, 1):
                    break
                rewards.append(reward)
            valid = len(rewards) == len(responses) == record.get("group_size") and len(rewards) > 1
            valid = valid and record.get("expected_group_size", len(responses)) == len(responses)
            mastered = valid and all(reward == 1 for reward in rewards)
            if review:
                if mastered:
                    self.tracked.remove(row)
                else:
                    # Unknown outcomes do not establish regression, but still need review.
                    self.queue.append(row)
            elif mastered and row not in self.tracked and self.rng.random() < self.config["enqueue_probability"]:
                self.tracked.add(row)
                self.queue.append(row)

    def state_dict(self):
        return {
            "kind": "remind",
            "config": dict(self.config),
            "base": self.base.state_dict(),
            "draws": self.draws,
            "queue": list(self.queue),
            "pending": dict(self.pending),
            "rng": self.rng.getstate(),
        }

    def load_state_dict(self, state):
        self._review_dataset()
        if state.get("kind") != "remind" or state.get("config") != self.config:
            raise ValueError("Prompt provider checkpoint does not match this ReMind configuration")
        draws, queue, pending = state["draws"], state["queue"], state["pending"]
        if type(draws) is not int or draws < 0:
            raise ValueError("Invalid ReMind draw counter")
        rows = list(queue)
        for selection, (row, review) in pending.items():
            if type(selection) is not int or not 0 <= selection < draws or type(review) is not bool:
                raise ValueError("Invalid pending ReMind selection")
            if type(row) is not int or not 0 <= row < len(self.samples):
                raise ValueError("Invalid ReMind row")
            if review:
                rows.append(row)
        if len(set(rows)) != len(rows) or any(
            type(row) is not int or not 0 <= row < len(self.samples) for row in rows
        ):
            raise ValueError("Invalid ReMind review queue")
        self.base.load_state_dict(state["base"])
        self.rng.setstate(state["rng"])
        self.draws, self.queue, self.pending, self.tracked = draws, deque(queue), dict(pending), set(rows)
