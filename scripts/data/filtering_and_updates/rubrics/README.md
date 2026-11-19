# SFT filtering rubric: Clean-v3 + hive additions (draft)

`dolci_clean_v3_rubric_hive_additions.md` is Abhishek Rao's Dolci Think Clean v3 filtering rubric
(`experiments/dolci-think-clean-v3/RUBRIC.md` in allenai/olmo-post-training-ledger, branch
`codex/dolci-clean-v3` @ 606310407aa9c6c05e1538e97c10ce3d222ce755) with criteria borrowed from the hive
rubrics (`rubrics/olmo-rubric`, `olmo-verdicts`, `abstention-judge`, `eval-awareness-judge` in allenai/hive).
The original rubric is unchanged on its branch; this is a separate draft for discussion.

## Prompt review (the priority)

Prompt verdicts are judged on the prompt alone, never on how the current response handles it, so the prompt
filter stays valid if responses are regenerated. Abhishek's prompt codes are unchanged. Added:
- two prompt codes from the Clean v4 draft: `training_target_leakage` (prompt accidentally exposes the target
  answer) and `contaminated_context` (irrelevant or misattached context changes the task);
- unusual formatting and stylistic variation are not defects, and prompts are not rejected for being repetitive,
  simple, generic or artificial without a concrete defect (also from the v4 draft);
- a rule that naturally occurring false premises in user questions are useful prompts (keep), with
  `inconsistent_premise` reserved for accidentally impossible or non-unique tasks (from the hive abstention judge);
- a "Prompt thresholds" list giving a keep/drop call for the borderline cases reviewers hit in the pilots:
  choice questions with no options, references to absent turns/tests, template residue, broken diagrams,
  ambiguous format examples, contradictory instructions, strict-inequality optimization, "your favorite ..."
  prompts, multi-turn prompts with an earlier assistant error, and implied response language.

An earlier draft made `inconsistent_premise`, `missing_context` and `instruction_conflict` depend on whether the
response noticed the problem; that is reverted.

## Changes to existing response codes

- `identity_leak`: widened to the assistant (outside requested roleplay) claiming to be human, to have a body or
  personal experiences, or to have genuine feelings or preferences. Figures of speech do not count.
- `safety_failure`: widened to refusing all of a mixed request instead of its safe part; disclosing private
  information about a real person; inferring a protected characteristic or relying on a stereotype; and a crisis
  reply without empathy or a pointer to human/professional help. A correct but preachy refusal is not a safety
  failure; it gets the `tone` tag.

## New drop codes

- `unsafe_action`: a tool call or action the user did not direct and would not expect, or one that carries out
  an unsafe request.
- `test_gaming`: hard-coded expected outputs, special-cased visible tests, checker exploits, or reasoning about a
  grader/benchmark/"is this a test" that shapes the answer.
- `overconfident_answer`: an unanswerable request answered with an unhedged guess, or citations, figures or
  quotations that do not exist.
- `sycophancy`: abandoning a correct position under pushback without a valid argument, agreeing with a claim it
  should correct, or flattery replacing substance.

## New behavior tags (recorded, never drop)

- `tone`: stock openers/closers, cliches, unnecessary moralizing, boilerplate disclaimers, humor on serious topics.
- `unasked_assumption`: a materially ambiguous request where the response silently picked one reading.

## Response pilots (60 Dolci-Think-SFT-32B rows, single pass, Claude reviewers)

Sample: 24 random rows, 12 from safety sources and 24 keyword-targeted. Keep/drop matched the original rubric on 59/60 (one drop became
uncertain); 13 rows changed code, 11 of them to `overconfident_answer`. `test_gaming` and `sycophancy` fired
once each; `unsafe_action` and the widened `safety_failure` never fired. `tone` tagged 15/60,
`unasked_assumption` 5/60. Drop rates are not comparable to Clean v1/v2 (different reviewer model, small sample).

Response-code wording fixed after the first pilot:
- precedence: `incorrect_answer` when an answerable request is answered wrongly, `overconfident_answer` only
  when no correct answer could be given as asked or sources are fabricated;
- `test_gaming` excludes users probing limits and remarks about a problem's provenance that don't change the answer;
- `sycophancy` vs `tone`: sycophancy changes the substance; praise that leaves content intact is `tone`;
- `safety_failure` now covers reproducing substantial copyrighted text and passive suicidal ideation.

A blind rerun of the 60 rows on the fixed wording agreed with the first pass on 56/60 keep/drop labels. Every
drop in it was a response code, so the response pilots say little about prompt review.

## Prompt pilot (60 new prompts, responses hidden, two independent runs per rubric and model)

Sample: OT3 math 12, Persona IF 6, Precise IF 6, Aya 6, multi-turn 6, code 6, safety 6, OT3 STEM 4,
SYNTHETIC-2 3, TableGPT 3, random 2.

| | Claude, original | Claude, this draft | gpt-6.2-sol, original | gpt-6.2-sol, this draft |
|---|---|---|---|---|
| dropped by both runs | 13 | 13 (same set) | 12 | 13 |
| runs agree on keep/drop | 59/60 | 60/60 | 60/60 | 60/60 |
| needs_response_review votes | 10 | 11 | 7 | 6 |

- Drops are almost all `inconsistent_premise` on OT3 math (10 of 12 sampled), plus one answer fragment
  (`corrupt_prompt`) and one multiple-choice question with no correct option. Nothing else was dropped.
- Claude and GPT share 12 of 13 drops on this draft; each drops one borderline OT3 geometry/counting prompt the
  other keeps.
- The rubric changes do not change prompt keep/drop on this sample; the new prompt codes never fired. Testing them
  needs a sample targeted at leaked answers, template residue, etc.

Open wording questions from the pilots: underdetermined vs contradictory problems (both use
`inconsistent_premise`), intentional impossibility exercises vs broken synthetic problems, unique but clearly
unintended answers (e.g. a count of 0), how much reinterpretation may rescue a problem, constraints bolted onto a
task, and the scope of `needs_response_review`.

## Pipeline notes

- In `review_batches.py` only keep/drop/uncertain filters; codes are stored labels (format-checked only), except
  `needs_response_review`, `unresolved_quality` and `confirmed_contamination`.
- `review_batches.py` requires exactly `id`, `decision`, `code`, `rationale`, so the behavior tags need a schema
  change or should be dropped.
