# Shared grading inputs and behavior changes

MILES training combines mathematical, code, instruction-following, tool and model-judge rewards. Historically these graders selected their own answer regions, so a correct answer in thinking could earn credit even when the delivered answer was wrong or unfinished. This implementation gives final-answer grading one shared representation of thinking and the delivered response, with explicit exceptions for objectives that need more context. This page records the effective grading changes against open-instruct `46d2dbbb5bc13baa7c821afb82cd16a95be52e41`, including the paths whose behavior is preserved.

## Configuration and eligibility

The production default remains `reward_final_answer_only=false`. That mode retains existing grader dispatch, boundary extraction, token inputs, gates and judge payloads. Registry `grading` policies are validated but do not change legacy scoring. Direct verifier calls, Core/vLLM and DeepSpeed/vLLM trainers, and the OPD wrapper retain their existing behavior. OPD explicitly requests legacy scoring.

For a thinking-policy run:

```toml
[core]
reward_final_answer_only = true
reward_reasoning_mode = "required"
```

For an explicitly non-thinking run, use `reward_reasoning_mode="none"`. This accepts the whole response without requiring `</think>`. It rejects reasoning control tags and a prompt that prefills `<think>` as `unexpected_reasoning`; it does not infer the reasoning convention from absent generated tags.

`ResponseParts` contains the original `raw` text, terminal-cleaned `full` text, `thinking`, delivered `response`, the response's character offset, prompt-opening/closing state, closing-marker count, and completion status. A tool call is a delivered response; the model need not append a prose summary to earn action correctness credit. The thinking opener may be in the prompt or response. If the rendered prompt already ends in a closed thinking section, generated text starts in the response region; no prompt tokens are counted as generated thinking. A missing delivered response is distinct from a present but empty one.

Selection retains the last literal `</think>` convention. Missing closure and whitespace-only or terminator-only tails receive `missing_final_answer`. A new `<think>` at the beginning of the selected suffix, after whitespace, receives `unfinished_reasoning`. An interior literal mention in explanatory prose does not trigger that shared gate. Repeated closing markers still select the last suffix and are recorded rather than globally rejected. Selection does not extract a scalar, program, JSON document, answer tag or tool call: those remain task parsers' responsibilities.

The two ordinary input views are **full response** and **after thinking**. Selection preserves leading/trailing whitespace, answer wrappers, code fences, and explanatory prose in the selected text. The sole transport cleanup removes the existing terminal `<|endoftext|>` / `<|im_end|>` markers; it does not generally trim whitespace. Testing whether a response is empty uses whitespace inspection without modifying the delivered text. Thinking-only input remains an explicit exception for process objectives, rather than another answer-extraction heuristic.

`reward_final_answer_only=false` sends terminal-cleaned full text into the historical consumer path. `true` selects the after-thinking view by default, with the registry's explicit process exceptions. These statements describe **input selection**, not a guarantee that every historical consumer grades that input verbatim: the consumer review below makes those remaining parsers visible. Non-thinking runs and prompt-prefilled closed thinking select the complete generated text as the delivered response.

Only the trusted reward registry can select another grading view:

```json
{
  "answer": {
    "factory": "open_instruct.miles.rewards.task_verifiers.MultiplicationVerifier"
  },
  "whole_response_length": {
    "factory": "open_instruct.ground_truth_utils.UpToMaxLenVerifier",
    "grading": {"scope": "full", "require_response": false}
  }
}
```

| Setting | Meaning |
|---|---|
| `scope="response"` | Default: grade the delivered answer or action. A completed nonempty response is mandatory. |
| `scope="thinking"` | Deliver the thinking region to a process grader. This is an explicit change of objective. |
| `scope="full"` | Deliver terminal-cleaned full text, with original response/trajectory tokens. Native task parsers may still extract their own answer from this input. |
| `require_response=true` | Default for every scope: a missing/empty/unfinished delivered response gates this component at its verifier floor. |
| `require_response=false` | Explicitly allow thinking/full objectives to grade without a completed delivered response. Cannot be used with `scope="response"`. |

Full-text access does not itself bypass eligibility. Setting `require_response=false` requires an explicit `scope="thinking"` or `scope="full"`, even when a factory defaults to full scope; preparation and runtime enforce the same declaration. `reward_zero_truncated` still gates **every** component, including explicit thinking/full exceptions. It is independent of response selection. Gates retain each verifier's target-dependent floor and weight; they are neither verifier failures nor ungraded-response omissions.

Do not add `grading` to per-row verifier metadata: targets and weights do not authorize a different input policy. A malformed registry policy fails registry loading/prepared-data validation. Token-based verifiers reject scopes other than `full` at registry load. ToolRL supports `response` and `full`, not `thinking`. Its native format objective requires reasoning; final mode rejects ToolRL or a thinking-scope objective combined with `reward_reasoning_mode="none"` before grading/gating.

## Verifier interface

Ordinary verifiers continue to receive `async_call(tokens, prediction, label, **kwargs)`, with `prediction` selected by the registry policy. For `scope="response"`, an existing `async_call_final_answer` override still takes precedence. R1 format and native manifest IF keep those overrides.

A grader needing both pieces can implement:

```python
async def async_call_parts(self, tokens, parts, label, *, scope, query=None, rollout_state=None):
    prediction = parts.text(scope)
    # parts.thinking and parts.response are independently available.
```

The dispatcher uses this method only in final-answer mode, after the component's eligibility gate. It takes precedence over the final-answer/string methods. Legacy mode always uses `async_call`. No adapter to an external experimental verifier is included. A factory outside the `open_instruct` namespace must explicitly declare registry `grading` or implement the parts protocol before it can score in final mode. An undeclared contract raises before the missing-response gate, rather than silently returning all zeros. An explicit declaration is an assertion of compatibility, not automatic qualification: external graders still need positive/negative controls. Optional string fallback is retained for in-repository text graders.

Only full-scope consumers receive original tokens. Response/thinking scopes receive `[]`, as the sample does not provide reliable decoded-character-to-token alignment. The implementation does not retokenize generated text or fabricate token slices. A token-dependent custom grader must declare `grading_uses_tokens=True` and `grading_scope="full"`; the registry rejects a selected-text scope for it. Text-based thinking grading is supported; token-based thinking-only grading requires an explicit future alignment contract.

Original samples, labels and training loss masks are preserved. `metadata.grading` records the parser version, reasoning convention, boundary status, character counts, and every component's scope, eligibility requirement and gate. It does not duplicate the response text. `reward_gate` remains a compatibility summary when any component is gated; the per-component record is authoritative for mixed objectives.

The Open Instruct recorder integration retains this summary in each saved response, including when raw-response recording is disabled. New final-mode group rows also carry `grading_policy` and extend their evidence `input_key` with the selector version, reasoning convention and truncation policy. This prevents pooling new final-mode evidence with legacy results or a different declared reasoning convention. The feedback recorder and persisted row use the same extended key. Default-off rows and the pinned base recorder's manifest format are unchanged.

## Consumer review: selection versus task parsing

The first simplification separates the shared selector from historical consumer normalization. The selector owns the reasoning boundary and eligibility; it does not decide how an answer should be presented. The old suffix-and-trim helper now lives in `deterministic_verifiers.answer_text`, alongside the consumers that still use it. This relocation preserves its historical output. The only new consumer behavior in this step is that named quality judges receive their selected view intact. Contract version **2** distinguishes that behavior from the first implementation of this branch (`a3fcb7a6e60ad84dc277f2582d0b0ba4cdf252d5`, version 1).

The remaining review is deliberately per consumer. A full-scope declaration currently controls the string delivered to the consumer; it does **not** disable the consumer's internal extraction. Do not treat a full-scope task parser as a verbatim whole-response objective until it is reviewed. Removing a parser requires evidence from the actual prompt contract and positive/negative examples. Supplying fabricated thinking delimiters to satisfy an old parser would hide the input contract and is not the migration strategy.

| Consumer / implementation | Current interpretation after selection | Decision in this step / remaining review |
|---|---|---|
| Named quality, reference quality and WebInstruct judges: `general_judge`, `judge_registry` | Historical helper preferred the first answer wrapper and otherwise selected after the first closing think tag; its trimming depended on which branch matched. | **Changed for flag-on, every scope:** bypass that helper and render the exact selected text. No implicit presentation requirement. Flag-off and direct helper callers retain historical behavior. Templates, reference targets, judge models and verdict parsing are unchanged. |
| Legacy `LMJudgeVerifier` | Uses the same generic answer-wrapper helper inside its own async method. | **Pending:** provide an explicit intact-input route; assess each registered prompt before retaining wrapper extraction. It does not inherit the named-judge fix. |
| Legacy `RubricVerifier` | Selects the last answer wrapper, unlike the first-wrapper rule above. | **Pending:** review its rubric prompts and remove implicit wrapper extraction for ordinary quality grading. |
| MILES Math-Verify, legacy strict math and GSM8K | Mathematical answer anchors, equivalence, or last-number parsing within the supplied text. | **Retained:** these are task parsers. Review allowable answer presentation separately; shared selection must not choose mathematical candidates. |
| Legacy code client / complete-code | Program extraction; complete-code also calls the historical suffix-and-trim helper. | **Retained:** code extraction is needed for execution, but redundant reasoning selection can be removed when an explicit selected-input route is introduced. Verify fence and bare-code expectations against prompts. |
| Deterministic MCQA / numeric / exact text / countdown / knights | Historical suffix-and-trim helper, then target-specific answer declarations, boxes, wrappers or assignment parsing. | **Pending:** separate suffix selection from each target's presentation parser. Last-line parsers need particular care around trailing newlines; do not blanket-remove whitespace normalization and silently change grades. |
| Deterministic formatting | Historical helper before prefix/line/regex constraints. | **Priority review:** trimming can change whether a presentation constraint is satisfied. Review the prepared instruction before changing that behavior. |
| Manifest IF and direct legacy IFEval / IFEvalOld | `remove_thinking_section` strips assistant/answer tags and trims text before checking constraints. | **Priority review:** the native Manifest IF final override still delegates to this normalization. Decide which transformations the IF prompt actually permits; an explicit selected-input method must avoid another thinking-boundary pass. |
| NeMo instruction / calendar | Historical helper before native instruction checking or calendar JSON parsing. | **Pending:** review native normalization independently from reasoning selection. Preserve native constraint aggregation and calendar semantics. |
| JSON Schema | Historical helper, then JSON decoding and schema validation. | **Pending:** selected-input route should retain JSON's own permitted whitespace while rejecting extra prose; no generic object extraction should be introduced. |
| Structured JSON / YAML / XML / TOML / CSV | Historical helper, then the format's parser, contract and optional fence handling. | **Pending per format:** check fence and surrounding-text requirements against the rendered prompt; retain the actual format parser. |
| Litmus | Historical helper, then its declared answer regex. | **Pending:** remove redundant boundary selection while retaining the prompt's regex contract. |
| Reasoning Gym | Historical helper, then family-specific boxes, answer tags and native scoring. | **Pending per family:** wrapper expectations differ across tasks; no global extraction replacement. |
| Long-context tasks | Historical helper, then kind-specific answer declarations and normalization. | **Pending per kind:** preserve ranking, partial credit and comparison rules while reviewing answer presentation. |
| NeMo single-step tools | Own boundary/opening-state handling, then native call/message parsing and comparison. | **Pending:** selected actions need a declared route that skips boundary discovery while retaining action syntax and extra-call checks. |
| ToolRL | Structured-parts route uses actual boundary information for format and selected action for correctness. | **Already explicit:** retain both pieces rather than inventing tags. Format/correctness objectives and full-scope native behavior remain separately documented. |
| R1 format / multiplication | R1 has an explicit final-answer method; multiplication expects answer tags. | **Retained pending prompt review:** wrapper format is the objective for R1 and a historical task requirement for multiplication. Do not make it a universal grading rule. |
| Sudoku / Flan / string / F1 / puzzle / search | Task-specific board, phrase, answer/finish-tag or text normalization rules. | **Pending per factory:** inspect their prepared prompts and registered use before changing presentation requirements. |
| Length / passthrough / environment | Original trajectory length or external state, rather than answer extraction. | **Retained:** explicit full-trajectory exceptions where required; do not manufacture selected token IDs or environment results. |
| External factories | Trusted grading declaration or structured-parts interface required in flag-on mode. | **Retained:** the declaration is an asserted contract, not proof that an old parser accepts it. Require consumer controls during migration. |

This checklist records known implementations and remaining decisions; it does not claim every pending consumer has been migrated or every registry route exercised. The effective-change register below describes the currently implemented behavior, including retained historical parsers.

The version-2 follow-up replay used the same locally sampled 1,365 stored responses as the version-1 consumer panel: 1,344 stratified responses across 30 observed paths plus 21 targeted math examples. Across both flag settings it produced 2,070 numeric outcomes and 660 captured service inputs, with no errors, timeouts or additional changes relative to `a3fcb7a6e60ad84dc277f2582d0b0ba4cdf252d5`. The selected judge answers in this panel do not expose the wrapper-extraction difference; this is a coverage limit, not evidence that removing extraction is universally inert. Focused tests exercise answer tags, text outside them, exact whitespace preservation, all three named judge types, every scope and both flag settings. No submitted programs were executed and no live judge verdicts were requested. The CPU checks for this follow-up passed 650 tests, including seven against the pinned recorder; five optional tests were skipped and five existing tests remained expected failures.

## Effective grading change register

The comparison below distinguishes the old flag-on implementation from flag-off compatibility. The new default-off implementation is unchanged in every row unless a future, separately documented correctness fix changes it.

| Path | Previous flag-on behavior | New flag-on behavior / effective change | Validation |
|---|---|---|---|
| General MILES math (including remapped legacy factory), strict math, GSM8K | Grade the last closing-marker suffix. | Same selected text, anchor rules and mathematical equivalence. Correct thinking cannot rescue a wrong delivered answer. | Existing symbolic worker regression and math tests; selection tests. |
| Legacy assertion/stdio code | Extract a program from the selected suffix. | Same suffix and code/service contract. No execution-service, timeout or test changes. | Final-answer code-dispatch tests. |
| Complete-code | Select suffix, then its task parser selects again. | Same effective text/program; shared selection precedes the unchanged code contract. | Complete-code tests and shared selection tests. |
| Deterministic MCQA, numeric, exact text, countdown, formatting, knights | Select last suffix; parse according to the target. | Same for closed responses. Direct-call whole-response fallback is preserved through the shared compatibility extractor. | Deterministic verifier tests. |
| NeMo IF / calendar | Select last suffix and apply the target. | Same for closed responses; no fence, tolerance or constraint-aggregation changes. | NeMo constraints tests. |
| Dolci manifest IF / legacy IFEval | Outer selection followed by IF normalization. | Same native normalization and constraint aggregation. No historical qualified-module adapter. | Real native IF wrapper/copy-span parity tests. |
| JSON Schema / structured output | Select suffix and validate parsed content. | Same parsing and schema semantics; schema-only content weaknesses are not changed. | Structured/JSON-schema tests. |
| Litmus / Reasoning Gym / long context | Select suffix, then task extraction/scoring. | Same closed-response behavior, tolerances and native partial credit. | Family tests. |
| Named judges: response scope | Select suffix; rubric extracts answer tags within it and inconsistently trims whitespace. | **Intentional input change:** send the entire selected response intact, including wrappers, whitespace, and text outside answer tags. No implicit answer extraction. Answer tags in thinking remain inaccessible. | All three named judge templates, every scope, both flag settings, and exact payload preservation tests. |
| Named judges: explicit thinking/full scope | No per-grader scope. | New explicit objective: selected view is sent intact, with answer extraction disabled. Rubric templates, bindings and score parsers are unchanged. | New judge view/payload tests. |
| Legacy `LMJudgeVerifier` / `RubricVerifier` registered directly | Receive the outer suffix, then apply their own first-answer/last-answer tag rules. | Same delivered input and native wrapper rules. They do not use named-judge extraction bypass; explicit thinking/full scopes still pass through their native task parser. Direct legacy callers remain unchanged. | Source tracing and shared string-dispatch regressions; no live legacy-judge qualification claimed. |
| NeMo single-step tools | Outer suffix; prompt-opening state disabled after selection. | Same normal action/message comparison and prompt state for response scope. Full scope retains actual prompt-opening state. Similarity thresholds and extra-call behavior are unchanged. | NeMo worker/prompt-opener tests. |
| ToolRL response scope | Lost the raw boundary; valid no-call format could score 0 and call rewards could lose format credit. | **Correctness fix:** raw boundary supplies format credit; shared selected action supplies correctness. Prompt-prefilled reasoning comes from the rendered prompt, rather than a copied label flag. Reward scaling, floors and partial-credit formula are unchanged. | New call/no-call and prompt-prefix tests; existing ToolRL scale/floor/parser tests. |
| ToolRL repeated markers | Suffix-only input lost format context and native parsing otherwise used the first marker. | Format still fails repeated boundaries; correctness uses the shared last suffix in response scope. Explicit full scope deliberately retains native first-marker action parsing. | Repeated-marker action-selection test. |
| R1 format | Optional suffix override checks answer tags. | Same response-scope check; empty answer tags/trailing text retain existing format semantics. Explicit full scope uses the existing raw-boundary check. | Existing final-answer format tests. |
| Multiplication / Sudoku / Flan / string/search/F1/puzzle tasks | Parse task-specific answer tags, finish tags or text from suffix. | Same selected input and task parsers. No new participation floors or answer-wrapper rules. | Existing task tests and shared selection tests. |
| Passthrough/environment objectives | Passthrough returns zero independently of text; legacy environment workflows aggregate rollout rewards separately. | Same scoring and rollout-state forwarding. Explicit full-scope process graders can retain trajectory input; no environment reward aggregation or per-turn decomposition is changed. | Source tracing and shared dispatch/state-preservation regressions. |
| MaxLen / UpToMaxLen registered in MILES | Received `[]`, silently scoring zero-length input. | **Correctness fix:** declared full scope retains actual trajectory tokens. Still requires a delivered response unless explicitly exempted. Legacy callers and formulas are unchanged. | New actual-factory length tests and invalid-scope rejection. |
| Leading reopened reasoning after closure | Generic suffix graders could award credit; NeMo already rejected it. | **Intentional eligibility change:** response-required components receive their floor. Interior literal mentions do not trigger the shared gate. Explicit thinking/full exceptions may still grade. | Reopened-reasoning, literal-mention and legacy-parity tests. |
| Declared non-thinking runs | Flag-on required a closing tag and zeroed valid tagless answers. | **New opt-in:** `reward_reasoning_mode="none"` accepts a completed tagless response, reserving reasoning tags as control syntax. | Non-thinking/parser tests. |
| Prompt-prefilled closed thinking | Flag-on required another closing marker in generated text. | **Intentional boundary correction:** a rendered prompt ending in `</think>` establishes that generated text is already the delivered response. | Prompt-prefilled closed-thinking test. |
| Mixed response/process objectives | One global missing-final gate suppressed all components. | **New opt-in:** each declared exception can grade; response-required components still receive their floor. Truncation remains global. | Mixed-component/floor/truncation tests. |
| External custom verifiers | Could silently reapply a raw-boundary requirement to a selected suffix. | **Compatibility requirement:** final mode now rejects undeclared external input contracts before eligibility gating; legacy mode remains supported. | External-verifier declared/full-scope and missing-response tests. |
| Verifier/service failures, omissions, weights, floors | Existing exception, exclusion and target-dependent floor behavior. | Unchanged. New input provenance does not convert infrastructure failures to model mistakes. | Existing reward-failure/omission suites. |
| Inference records and feedback | Pinned recorder drops arbitrary metadata and its protocol identity omits global grading flags. | **New-contract provenance:** persist per-response selection; extend final-mode evidence keys with the effective global grading policy. Default-off keys, admission and rewards are unchanged. | Actual pinned-recorder persistence, feedback/key parity and contract-separation tests. |
| OPD, direct callers, non-MILES GRPO | Their own extraction contracts. | Unchanged; no global migration of these workflows is claimed. | OPD and direct-call tests. |

## Limits and remaining qualification

Literal thinking delimiters inside quotations, code, JSON strings or tool arguments are not distinguishable from structural tags using decoded text alone. Last-marker selection retains that existing limitation; a literal opener at the start of a selected response now triggers the documented reopened-reasoning gate. Interior openers are not interpreted by the shared eligibility check; native task parsers may impose additional format restrictions (NeMo tools retain their conservative opener check). Do not treat marker counts as a complete conversation parser. Multi-turn trajectories are retained for full-scope token consumers, not decomposed into per-turn thinking/response pairs.

Bad targets can still be hidden by an eligibility gate if that verifier has no independent target validation before scoring. Returned timeout semantics, permissive native checker rules, schema-only rewards and judge score parsing remain separate audit items. This refactor does not certify service deployment, gold-label correctness, external experimental modules or all native Reasoning Gym subtasks.

The pinned recorder's original protocol omission also affects historical/default-off evidence under different truncation settings. That existing upstream issue is not retrospectively repaired here; old records must not be assumed comparable merely because their old input keys match. New per-row grading identity supplements the base manifest, and runtime/verifier revision provenance is still required. The selector schema version must change if a later implementation changes its grading semantics.

H180 found mixed accuracy and termination changes and retained an IF normalization confound even after reward flow was restored. It does not establish that the cleaner objective improves stopping. These code changes introduce no format bonus, new training run, sampling change or production-default change. Saved-response replay and a controlled training comparison remain separate evidence from CPU regression tests; passing input-contract tests does not establish training quality.

## Saved-text selection replay

The [compact replay receipt](grading-selection-replay.json) covers the retained November 8 export of 39,535 math draws from 27 sources (screen and historical Dolci). Compared with the baseline revision, legacy helper text changes **0/39,535** times. Text is identical for all responses eligible under both final-selection policies. The shared leading-reopen gate newly excludes **35** archived responses; missing-final counts remain **249**, and **110** responses contain multiple closing markers. The gate changes were inspected: an earlier broad substring rule also caught an explanatory mention of `<think>`, so the implemented rule is anchored to the beginning of the selected response.

This is an input/eligibility replay, not executed verifier scores, false-positive adjudication or a population error rate. Historical archived text may be shortened, and this export does not provide prompt-prefill or finish-state evidence. No truncation status is inferred, no missing text is reconstructed, and no measured training effect is claimed. ToolRL/judge/length behavior is covered by actual dispatch regressions, not this math-only corpus.

Reproduce on the retained export, or another JSONL export with the same fields:

```bash
python -m scripts.test.replay_grading_selection draws.jsonl --output selection.json
```

The script streams the input and records its SHA-256, source counts and a bounded set of changed coordinates without copying raw responses. The original export is a local audit artifact; its preparation script is retained under `outputs/math-verify-false-credit-20261108/scripts/export.py` in the original workspace. The receipt binds the exact available bytes but does not establish durable archival availability.
