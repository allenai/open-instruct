# SFT filtering rubric: Clean-v3 + hive additions (draft)

`dolci_clean_v3_rubric_hive_additions.md` is Abhishek Rao's Dolci Think Clean v3 filtering rubric
(`experiments/dolci-think-clean-v3/RUBRIC.md` in allenai/olmo-post-training-ledger, branch
`codex/dolci-clean-v3` @ 606310407aa9c6c05e1538e97c10ce3d222ce755) with criteria borrowed from the hive
rubrics (`rubrics/olmo-rubric`, `olmo-verdicts`, `abstention-judge`, `eval-awareness-judge` in allenai/hive).
The original rubric is unchanged on its branch; this is a separate draft for discussion.

## Changes to existing codes

- `inconsistent_premise`: narrowed. Drop only when the response does not recognize the conflict; if it says
  there is no unique answer, corrects the premise or asks for what is missing, keep (useful abstention). If it
  answers as if the task were well posed, the drop code is `overconfident_answer`.
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

## Pilot (60 Dolci-Think-SFT-32B rows, single pass, Claude reviewers)

Sample: 24 random rows, 12 from safety sources and 24 keyword-targeted. Keep/drop matched the original rubric on 59/60 (one drop became
uncertain); 13 rows changed code, 11 of them to `overconfident_answer`. `test_gaming` and `sycophancy` fired
once each; `unsafe_action` and the widened `safety_failure` never fired. `tone` tagged 15/60,
`unasked_assumption` 5/60. Drop rates are not comparable to Clean v1/v2 (different reviewer model, small sample).

Wording fixed after the pilot (not yet re-piloted):
- "recognizes the problem" now refers to the final answer, not only the reasoning;
- precedence: `incorrect_answer` when an answerable request is answered wrongly, `overconfident_answer` only
  when no correct answer could be given as asked or sources are fabricated;
- the noticed-problem carve-out now also covers `missing_context` and `instruction_conflict`;
- `test_gaming` excludes users probing limits and remarks about a problem's provenance that don't change the answer;
- `sycophancy` vs `tone`: sycophancy changes the substance; praise that leaves content intact is `tone`;
- `safety_failure` now covers reproducing substantial copyrighted text and passive suicidal ideation.
