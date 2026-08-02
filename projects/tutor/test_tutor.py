"""Tests for the tutoring project.

Runs on a laptop with nothing installed - the judge is stubbed and the leak rule
needs no model at all.

    python -m unittest projects.tutor.test_tutor -v
"""

from __future__ import annotations

import asyncio
import json
import random
import unittest

from open_instruct.scored_rewards import registry
from open_instruct.scored_rewards.types import TRANSCRIPT_KEY, Sample
from projects.tutor import build_dataset, leak, plugin, rubric, student

ITEM = {
    "question": "Which object is the best conductor of electricity?",
    "choices": ["a rubber band", "a copper wire", "a glass cup", "a wooden spoon"],
    "gold_idx": 1,
    "grade": 6,
}


def run(coro):
    return asyncio.run(coro)


def make_sample(turns, item=ITEM, index=0, group_size=1):
    """A sample as the partner environment would have left it."""
    transcript = [{"who": "partner", "text": "I don't get this."}]
    for turn in turns:
        transcript.append({"who": "policy", "text": turn})
        transcript.append({"who": "partner", "text": "ok..."})
    return Sample(
        completion=" ".join(turns),
        label=json.dumps(item),
        rollout={"info": {TRANSCRIPT_KEY: json.dumps(transcript), "scored_policy_text": "\n".join(turns)}},
        index=index,
        group_size=group_size,
    )


class TestLeakRule(unittest.TestCase):
    def flag(self, text, item=ITEM, **kwargs):
        return leak.leaked_item(text, item, **kwargs)

    def test_verbatim_gold_is_a_leak(self):
        self.assertTrue(self.flag("The answer is a copper wire."))

    def test_identifying_word_alone_is_a_leak(self):
        """`copper` appears in gold and in no distractor - naming it is enough."""
        self.assertTrue(self.flag("Think about what copper is used for."))

    def test_conceptual_hint_is_not_a_leak(self):
        self.assertFalse(self.flag("Which of these do you think lets electricity flow easily?"))

    def test_a_word_the_question_already_said_is_not_identifying(self):
        self.assertFalse(self.flag("Remember what electricity needs in order to move."))

    def test_generic_words_do_not_fire(self):
        self.assertFalse(self.flag("Take your time and think about each one carefully."))

    def test_stemming_catches_inflections(self):
        item = {"question": "What happens to the ice?", "choices": ["it melts", "it grows"], "gold_idx": 0}
        self.assertTrue(self.flag("What is melting here?", item=item))

    def test_numeric_gold_is_caught(self):
        item = {"question": "A box holds 4 rows of 6.", "choices": ["24", "18", "30"], "gold_idx": 0}
        self.assertTrue(self.flag("You should end up with 24.", item=item))

    def test_a_number_from_the_question_is_not_a_leak(self):
        item = {"question": "A box holds 4 rows of 6.", "choices": ["24", "18", "30"], "gold_idx": 0}
        self.assertFalse(self.flag("You have 4 rows - what does that tell you?", item=item))

    def test_overlap_and_elimination_are_off_by_default(self):
        """Naming every wrong option is elimination. Calibration turned it off."""
        text = "It is not a rubber band, not a glass cup, and not a wooden spoon."
        self.assertFalse(self.flag(text))
        self.assertTrue(self.flag(text, use_elimination=True))

    def test_signals_are_all_reported_even_when_they_do_not_fire(self):
        signals = leak.leak_signals(
            "It is not a rubber band, not a glass cup, and not a wooden spoon.",
            leak.gold_text(ITEM),
            leak.distractors_of(ITEM),
            ITEM["question"],
        )
        self.assertEqual(signals["elimination"], 1.0)
        self.assertEqual(signals["verbatim"], 0.0)

    def test_long_gold_with_one_distinctive_word(self):
        """The case `identifying` exists for: overlap dilutes, this does not."""
        item = {
            "question": "Which grievance is listed in the Declaration?",
            "choices": [
                "For depriving us in many cases of the benefits of Trial by Jury",
                "For raising the price of imported tea",
            ],
            "gold_idx": 0,
        }
        text = "They couldn't have a fair trial - does that remind you of anything about juries?"
        self.assertTrue(self.flag(text, item=item))


class TestStudent(unittest.TestCase):
    def test_intents_contain_no_way_to_teach(self):
        self.assertNotIn("explain", " ".join(student.INTENTS))

    def test_mastery_needs_both_a_question_and_understanding(self):
        state = student.initial_state(ITEM, random.Random(0), level="high")
        state = student.update_mastery(state, "Just do it this way.", "Oh so I would add the two sides first?")
        self.assertEqual(state.mastery, "none")  # the tutor asked nothing
        state = student.update_mastery(state, "What would you add first?", "idk")
        self.assertEqual(state.mastery, "none")  # the student showed nothing
        state = student.update_mastery(state, "What would you add first?", "Oh so I would add the two sides first?")
        self.assertEqual(state.mastery, "partial")

    def test_mastery_never_falls(self):
        state = student.initial_state(ITEM, random.Random(0), level="high", mastery="solid")
        state = student.update_mastery(state, "no question here", "idk")
        self.assertEqual(state.mastery, "solid")

    def test_profile_pins_at_a_level(self):
        state = student.initial_state(ITEM, random.Random(1), level="low")
        self.assertEqual(state.ability, 5)

    def test_intent_is_from_the_closed_set(self):
        rng = random.Random(3)
        state = student.initial_state(ITEM, rng)
        for _ in range(50):
            self.assertIn(student.plan_intent(state, rng), student.INTENTS)

    def test_opening_line_is_deterministic_and_names_the_misconception(self):
        item = dict(ITEM, choose_pick=0)
        first = student.opening_line(item, seed=7)
        self.assertEqual(first, student.opening_line(item, seed=7))
        self.assertIn("rubber band", first)

    def test_director_never_shows_the_options(self):
        director = student.StudentDirector()
        rng = random.Random(0)
        director.system(ITEM, 0, rng)
        shown = director.user(ITEM, [{"who": "policy", "text": "hello"}], 1, rng)
        self.assertNotIn("copper wire", shown)
        self.assertIn(ITEM["question"], shown)


class TestRubric(unittest.TestCase):
    def test_paid_dimensions_exclude_complete(self):
        self.assertNotIn("complete", rubric.PAID_DIMENSIONS)
        self.assertIn("leak", rubric.PAID_DIMENSIONS)

    def test_turn_body_shows_gold_and_the_prior_context(self):
        body = rubric.turn_body(ITEM, [{"who": "partner", "text": "I'm stuck"}], "What conducts?")
        self.assertIn("Correct answer: a copper wire", body)
        self.assertIn("Student: I'm stuck", body)
        self.assertIn("What conducts?", body)

    def test_opening_turn_says_so_rather_than_showing_nothing(self):
        self.assertIn("opening the conversation", rubric.turn_body(ITEM, [], "hi"))

    def test_completion_gate(self):
        self.assertEqual(rubric.completion_gate(0.0), 0.0)
        self.assertEqual(rubric.completion_gate(1.0), 1.0)
        # a parse failure is our bug, not an abandoned dialogue
        self.assertEqual(rubric.completion_gate(None), 1.0)


class TestEpisode(unittest.TestCase):
    def test_each_tutor_turn_gets_the_context_before_it(self):
        sample = make_sample(["first turn", "second turn"])
        item, turns = plugin.episode_of(sample)
        self.assertEqual(item["question"], ITEM["question"])
        self.assertEqual([t for _, t in turns], ["first turn", "second turn"])
        self.assertEqual(turns[0][0], [{"who": "partner", "text": "I don't get this."}])
        self.assertEqual(len(turns[1][0]), 3)  # opener, first turn, the reply to it

    def test_no_environment_falls_back_to_the_whole_completion(self):
        item, turns = plugin.episode_of(Sample(completion="one shot hint", label=json.dumps(ITEM)))
        self.assertEqual(turns, [([], "one shot hint")])


class TestScorer(unittest.TestCase):
    def scorer(self, **kwargs):
        return plugin.build_tutor_scorer(stub=True, **kwargs)

    def test_leak_floors_every_dimension(self):
        inner = plugin.JudgedDialogue(stub=True)
        group = [
            make_sample(["Think about what copper does."], index=0, group_size=2),
            make_sample(["What flows easily?"], index=1, group_size=2),
        ]
        results = run(inner.score_group(group))
        self.assertEqual(results[0].info["leaked"], 1.0)
        self.assertEqual(set(results[0].dimensions.values()), {plugin.LEAK_FLOOR})
        self.assertEqual(results[1].info["leaked"], 0.0)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in results[1].dimensions.values()))

    def test_leaking_scores_strictly_worse_than_the_group(self):
        group = [
            make_sample(["The answer is a copper wire."], index=0, group_size=3),
            make_sample(["What lets electricity move?"], index=1, group_size=3),
            make_sample(["Which of these is a metal you can bend?"], index=2, group_size=3),
        ]
        scores = [r.score for r in run(self.scorer().score_group(group))]
        self.assertEqual(min(scores), scores[0])

    def test_group_scores_are_zero_mean(self):
        group = [make_sample([f"turn {i}"], index=i, group_size=4) for i in range(4)]
        scores = [r.score for r in run(self.scorer().score_group(group))]
        self.assertAlmostEqual(sum(scores), 0.0, places=9)

    def test_leak_only_ablation_needs_no_judge(self):
        scorer = registry.build("tutor_leak_only")
        group = [make_sample(["copper is the key"]), make_sample(["what do you think?"])]
        scores = [r.score for r in run(scorer.score_group(group))]
        self.assertEqual(scores, [plugin.LEAK_FLOOR, 0.0])

    def test_registered_under_its_name(self):
        self.assertIn("tutor", registry.available())


class TestDataset(unittest.TestCase):
    def test_rows_carry_the_item_the_env_and_the_opener(self):
        rows = build_dataset.build([dict(ITEM, choose_pick=0)], turns=3)
        row = rows[0]
        self.assertEqual(row["dataset"], "passthrough")
        self.assertEqual(row["env_config"]["env_configs"][0]["env_name"], "tutor_student")
        self.assertEqual(row["env_config"]["max_steps"], 3)
        self.assertIn("Student:", row["messages"][-1]["content"])
        self.assertEqual(json.loads(row["ground_truth"])["gold_idx"], 1)

    def test_the_tutor_never_sees_the_gold_marked_as_gold(self):
        prompt = "\n".join(m["content"] for m in build_dataset.build([ITEM])[0]["messages"])
        self.assertNotIn("Correct answer", prompt)
        self.assertNotIn("gold_idx", prompt)

    def test_single_turn_mode_has_no_env(self):
        self.assertNotIn("env_config", build_dataset.build([ITEM], env=None)[0])


if __name__ == "__main__":
    unittest.main()
