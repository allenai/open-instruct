# Shared grading inputs and behavior changes

MILES training combines mathematical, code, instruction-following, tool and model-judge rewards. Historically these graders selected their own answer regions, so a correct answer in thinking could earn credit even when the delivered answer was wrong or unfinished. This implementation gives final-answer grading one shared representation of thinking and the delivered response, with explicit answer, process, format and composite objective types for components that need different context. This page records the effective grading changes against open-instruct `46d2dbbb5bc13baa7c821afb82cd16a95be52e41`, including the paths whose behavior is preserved.

## Configuration and eligibility

The library default remains `reward_final_answer_only=false`; the maintained `dev`, `node` and `three-node` examples explicitly enable it for thinking policies. The disabled mode retains existing grader dispatch, boundary extraction, token inputs, gates and judge payloads. Registry `grading` policies are validated but do not change legacy scoring. Direct verifier calls, Core/vLLM and DeepSpeed/vLLM trainers, and the OPD wrapper retain their existing behavior. OPD explicitly requests legacy scoring.

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

Factory declarations and the trusted reward registry determine objective types and input views. A reasoning/whole-trajectory exception must declare `objective="process"` or `objective="format"`; `scope="full"` alone no longer authorizes an answer grader to see thinking:

```json
{
  "answer": {
    "factory": "open_instruct.miles.rewards.task_verifiers.MultiplicationVerifier"
  },
  "whole_response_length": {
    "factory": "open_instruct.ground_truth_utils.UpToMaxLenVerifier",
    "grading": {"objective": "process", "scope": "full", "require_response": false}
  }
}
```

| Setting | Meaning |
|---|---|
| `objective="answer"` | Default: correctness, quality or constraints on the presented answer/action. Requires `scope="response"`; it cannot opt back into thinking. |
| `objective="process"` | Explicit reasoning or trajectory objective. Requires `scope="thinking"` (default) or `scope="full"`. Length verifiers declare process/full in their factory. |
| `objective="format"` | Explicit presentation/structure objective. R1 checks answer presentation in the response; a boundary-format component such as ToolRL declares full context. |
| `objective="composite"` | A structured consumer with separately declared component objective types and views. ToolRL combines answer correctness on response with format on full text. A registry cannot relabel it as a single objective. |
| `scope="response"` | Default selected string: the delivered answer or action. A completed nonempty response is mandatory. |
| `scope="thinking"` | Deliver the thinking region to a process grader. This is an explicit change of objective. |
| `scope="full"` | Deliver terminal-cleaned full text, with original response/trajectory tokens. Native task parsers may still extract their own answer from this input. |
| `require_response=true` | Default for every scope: a missing/empty/unfinished delivered response gates this component at its verifier floor. |
| `require_response=false` | Explicitly allow thinking/full objectives to grade without a completed delivered response. Cannot be used with `scope="response"`. |

Full-text access does not itself bypass eligibility. Setting `require_response=false` requires an explicit `scope="thinking"` or `scope="full"`, even when a factory defaults to full scope; preparation and runtime enforce the same declaration. `reward_zero_truncated` still gates **every** component, including explicit thinking/full exceptions. It is independent of response selection. Gates retain each verifier's target-dependent floor and weight; they are neither verifier failures nor ungraded-response omissions.

Do not add `grading` to per-row verifier metadata: targets and weights do not authorize a different input policy. Preparation validates grading keys, types and enum values without importing factories. Runtime registry loading resolves factory defaults and validates objective/view combinations, token requirements and composite declarations. Token-based verifiers reject scopes other than `full` at registry load. ToolRL declares a composite objective: correctness always uses `response`, and format uses `full`. A full/thinking override of the composite is rejected rather than letting correctness see thinking. Its native format objective requires reasoning; final mode rejects ToolRL or a thinking-scope objective combined with `reward_reasoning_mode="none"` before grading/gating.

Named-judge bindings may replace a registry's factory/configuration, but preserve its trusted `grading` declaration unless the override explicitly supplies another one. For prepared inputs with a reward registry, inherited policies are included in the preparation contract. A cached preparation from before this preservation fix is rejected when it would have dropped an explicit policy; use a fresh output directory. Preparations without such declarations keep their existing contract.

## Verifier interface

Ordinary verifiers continue to receive `async_call(tokens, prediction, label, **kwargs)`, with `prediction` selected by the registry policy. For `scope="response"`, an existing `async_call_final_answer` override still takes precedence. R1 format and native manifest IF keep those overrides. Migrated consumers declare `grading_accepts_selected_input=True`; the dispatcher then supplies `selected_input=True` for any selected view, including process/full inputs. The consumer skips all thinking-boundary discovery but retains its task parser. This flag is internal to the trusted adapter and cannot be supplied by dataset rows. Flag-off dispatch omits it, preserving historical calls.

A grader needing both pieces can implement:

```python
async def async_call_parts(self, tokens, parts, label, *, scope, query=None, rollout_state=None):
    prediction = parts.text(scope)
    # parts.thinking and parts.response are independently available.
```

The dispatcher uses this method only in final-answer mode, after the component's eligibility gate. It takes precedence over the final-answer/string methods. Legacy mode always uses `async_call`. No adapter to an external experimental verifier is included. A factory outside the `open_instruct` namespace must explicitly declare registry `grading` or implement the parts protocol before it can score in final mode. The native reward bridge validates its registry in driver startup, before allocating workers: undeclared external contracts and incompatible reasoning conventions fail there. Direct scoring calls retain the same checks before eligibility gates. An explicit declaration is an assertion of compatibility, not automatic qualification: external graders still need positive/negative controls. Optional string fallback is retained for in-repository text graders.

Composite factories also declare `grading_components`, mapping names to `{objective, scope}`. Their parts method receives a `component_inputs` dictionary of the exact selected strings; ToolRL reads `format` and `correctness` from that dictionary. The current composite score retains its joint eligibility gate and native aggregation formula: sub-objectives do not silently acquire independent rewards, weights or gates. Ordinary independently weighted verifiers still have their own eligibility policies.

Only full-scope consumers receive original tokens. Response/thinking scopes receive `[]`, as the sample does not provide reliable decoded-character-to-token alignment. The implementation does not retokenize generated text or fabricate token slices. A token-dependent custom grader must declare `grading_uses_tokens=True` and `grading_scope="full"`; the registry rejects a selected-text scope for it. Text-based thinking grading is supported; token-based thinking-only grading requires an explicit future alignment contract.

Original samples, labels and training loss masks are preserved. `metadata.grading` records the parser version, reasoning convention, boundary status, character counts, and every component's objective type, scope, eligibility requirement and gate. Composite records additionally identify each sub-objective and its input view. It does not duplicate the response text. `reward_gate` remains a compatibility summary when any component is gated; the per-component record is authoritative for mixed objectives.

The Open Instruct recorder integration retains this summary in each saved response, including when raw-response recording is disabled. New final-mode group rows also carry `grading_policy` and extend their evidence `input_key` with the selector version, reasoning convention and truncation policy. This prevents pooling new final-mode evidence with legacy results or a different declared reasoning convention. The feedback recorder, persisted row and prompt-exclusion selector use the same extended key. Exclusion tables carry their grading contract, reject cross-contract use and require `--grading-policy` when the chosen lineage/protocol contains mixed grading evidence. New tables use schema 2; historical unversioned schema-1 tables remain usable only in legacy mode. Rebuild old final-mode tables from matching records instead of silently accepting incompatible keys. Default-off rows and the pinned base recorder's manifest format are unchanged.

## Consumer migration: selection versus task parsing

The shared selector owns thinking boundaries and eligibility. Consumers may still extract mathematical answers, code, JSON or answer wrappers from the supplied string, but migrated paths never look for `</think>` again. They retain their existing task-level whitespace normalization; this step does not change every formatting rubric. Contract version **3** identifies typed components and this migration, separating its evidence from versions 1 and 2 of this branch.

The 16 entry points below previously interpreted thinking boundaries after shared selection. They now accept the selected-input contract. The count includes Manifest IF separately from its native IFEval delegate; NeMo's extra parser is already included among the ten common-helper consumers.

| Consumers | Count | Flag-on behavior after migration |
|---|---:|---|
| Deterministic, complete-code, Litmus, JSON Schema, structured formats, long context, Reasoning Gym, NeMo calendar, NeMo instruction, NeMo single-step tools | 10 | `answer_text(selected_input=True)` retains task whitespace normalization and skips thinking discovery. All async/worker paths forward the contract. |
| Manifest IF, native IFEval, IFEvalOld, F1, puzzle matcher | 5 | Keep answer/assistant-tag cleanup and native normalization; skip the thinking split. The async bridge forwards the contract to opted-in synchronous implementations. |
| Legacy LM judge | 1 | Keep first-answer-wrapper extraction and assistant-tag cleanup; disable its thinking-boundary fallback. No live legacy-judge qualification is claimed. |
| NeMo's additional action parser | Included above | Bypass its own split and blanket opener rejection; shared eligibility is authoritative. Action syntax/comparison is unchanged. |
| Long-context opener rejection | Included above | Shared eligibility is authoritative; literal interior opener text is no longer automatically treated as unfinished thinking. |

| Other consumers | Objective / decision |
|---|---|
| Named quality/reference/WebInstruct judges | Answer by default; explicitly declared process/format views are possible. Their selected input stays intact, as in version 2. |
| ToolRL | Composite: full context for native boundary-format credit, selected response for action correctness. No local reasoning split or blanket opener rejection in selected-action correctness. Native format rules, scaling, partial credit and joint missing-response floor remain. |
| R1 format | Declares format/response; retains its answer-presentation check. An explicit format/full declaration retains the native raw-boundary check. |
| Math, strict math, GSM8K, legacy code, rubric judge, multiplication, Sudoku, Flan, string/search | No explicit thinking split to migrate in these consumers. Keep their downstream mathematical/program/wrapper/phrase parsing. |
| MaxLen / UpToMaxLen | Declare process/full; retain actual trajectory tokens. Other state/passthrough objectives retain their native contract. |
| External factories | Still require a trusted declaration or parts interface. An asserted contract does not automatically qualify an external parser; positive/negative controls remain required. |

Flag-off behavior is intentionally unchanged, including consumers that historically stripped thinking despite receiving full text. Making flag-off mean that every consumer actually grades full text would be a separate objective migration, reviewed per path rather than hidden in this change.

Validation results and saved-response measurements are recorded in [PR #2250](https://github.com/allenai/open-instruct/pull/2250); this page describes the grading contract and effective behavior changes.

## Effective grading change register

The comparison below distinguishes the old flag-on implementation from flag-off compatibility. The new default-off implementation is unchanged in every row unless a future, separately documented correctness fix changes it.

| Path | Previous flag-on behavior | New flag-on behavior / effective change | Validation |
|---|---|---|---|
| General MILES math (including remapped legacy factory), strict math, GSM8K | Grade the last closing-marker suffix. | Same selected text, anchor rules and mathematical equivalence. Correct thinking cannot rescue a wrong delivered answer. | Existing symbolic worker regression and math tests; selection tests. |
| Legacy assertion/stdio code | Extract a program from the selected suffix. | Same suffix and code/service contract. No execution-service, timeout or test changes. | Final-answer code-dispatch tests. |
| Complete-code | Select suffix, then its task parser selects again. | No second thinking split in selected-input mode. Program extraction and execution-service semantics remain unchanged. | Complete-code tests and shared selection tests. |
| Deterministic MCQA, numeric, exact text, countdown, formatting, knights | Select last suffix; parse according to the target. | Use the selected view without a second thinking split. Existing task whitespace, declarations and comparison rules remain; legacy/direct calls retain suffix fallback. | Deterministic verifier tests. |
| NeMo IF / calendar | Select last suffix and apply the target. | Skip local thinking selection on every selected view; retain native parsing, tolerances and constraint aggregation. | NeMo constraints tests. |
| Dolci manifest IF / legacy IFEval | Outer selection followed by IF normalization. | Skip local thinking selection; retain answer/assistant-tag cleanup, native whitespace normalization and constraint aggregation. No historical qualified-module adapter. | Real native IF wrapper/copy-span parity tests. |
| JSON Schema / structured output | Select suffix and validate parsed content. | Skip local thinking selection; format/schema parsing and schema-only content weaknesses are unchanged. | Structured/JSON-schema tests. |
| Litmus / Reasoning Gym / long context | Select suffix, then task extraction/scoring. | Skip local thinking selection; retain answer extraction, tolerances and partial credit. Long-context interior literal openers are now governed by the shared gate. | Family tests. |
| Named judges: response scope | Select suffix; rubric extracts answer tags within it and inconsistently trims whitespace. | **Intentional input change:** send the entire selected response intact, including wrappers, whitespace, and text outside answer tags. No implicit answer extraction. Answer tags in thinking remain inaccessible. | All three named judge templates, every scope, both flag settings, and exact payload preservation tests. |
| Named judges: explicit thinking/full scope | No per-grader scope. | New explicit objective: selected view is sent intact, with answer extraction disabled. Rubric templates, bindings and score parsers are unchanged. | New judge view/payload tests. |
| Legacy `LMJudgeVerifier` / `RubricVerifier` registered directly | Receive the outer suffix, then apply their own first-answer/last-answer tag rules. | LM judge skips its thinking fallback for selected views but retains first-answer-wrapper extraction. Rubric keeps its last-answer-wrapper rule. Process views are not re-split; wrapper extraction remains task parsing. Direct legacy calls are unchanged. | Source tracing and shared string-dispatch regressions; no live legacy-judge qualification claimed. |
| NeMo single-step tools | Outer suffix; prompt-opening state disabled after selection. | Skip both local thinking split and blanket opener rejection on selected views. Shared gates decide eligibility, so literal interior openers in messages/arguments are content. Native syntax, similarity thresholds and extra-call behavior remain unchanged. | NeMo worker/prompt-opener tests. |
| ToolRL response scope | Lost the raw boundary; valid no-call format could score 0 and call rewards could lose format credit. | **Correctness fix:** raw boundary supplies format credit; shared selected action supplies correctness. Prompt-prefilled reasoning comes from the rendered prompt, overriding both a copied label flag and registry `think_prefix_in_prompt` configuration. Reward scaling, floors and partial-credit formula are unchanged. | New call/no-call and prompt-prefix tests; existing ToolRL scale/floor/parser tests. |
| ToolRL repeated markers | Suffix-only input lost format context and native parsing otherwise used the first marker. | Format still fails repeated boundaries; correctness uses the shared last suffix. Composite full overrides are rejected; only flag-off/direct native calls retain first-marker action parsing. | Repeated-marker action-selection test. |
| R1 format | Optional suffix override checks answer tags. | Same response-scope check; empty answer tags/trailing text retain existing format semantics. Explicit full scope uses the raw-boundary check and rejects non-thinking mode, while response-format checking remains usable without thinking. | Existing final-answer format tests. |
| Multiplication / Sudoku / Flan / string/search/F1/puzzle tasks | Parse task-specific answer tags, finish tags or text from suffix. | Retain task parsers and answer-wrapper rules. F1 and puzzle skip their local thinking split on selected inputs. No new participation floors. | Existing task tests and shared selection tests. |
| Passthrough/environment objectives | Passthrough returns zero independently of text; legacy environment workflows aggregate rollout rewards separately. | Same scoring and rollout-state forwarding. Explicit full-scope process graders can retain trajectory input; no environment reward aggregation or per-turn decomposition is changed. | Source tracing and shared dispatch/state-preservation regressions. |
| MaxLen / UpToMaxLen registered in MILES | Received `[]`, silently scoring zero-length input. | **Correctness fix:** declared full scope retains actual trajectory tokens. Still requires a delivered response unless explicitly exempted. Legacy callers and formulas are unchanged. | New actual-factory length tests and invalid-scope rejection. |
| Leading reopened reasoning after closure | Generic suffix graders could award credit; NeMo already rejected it. | **Intentional eligibility change:** response-required components receive their floor. Interior literal mentions do not trigger the shared gate. Explicit thinking/full exceptions may still grade. | Reopened-reasoning, literal-mention and legacy-parity tests. |
| Declared non-thinking runs | Flag-on required a closing tag and zeroed valid tagless answers. | **New opt-in:** `reward_reasoning_mode="none"` accepts a completed tagless response, reserving reasoning tags as control syntax. | Non-thinking/parser tests. |
| Prompt-prefilled closed thinking | Flag-on required another closing marker in generated text. | **Intentional boundary correction:** a rendered prompt ending in `</think>` establishes that generated text is already the delivered response. | Prompt-prefilled closed-thinking test. |
| Mixed response/process objectives | One global missing-final gate suppressed all components. | **New opt-in:** each declared exception can grade; response-required components still receive their floor. Truncation remains global. | Mixed-component/floor/truncation tests. |
| External custom verifiers | Could silently reapply a raw-boundary requirement to a selected suffix. | **Compatibility requirement:** final mode now rejects undeclared external input contracts before eligibility gating; legacy mode remains supported. | External-verifier declared/full-scope and missing-response tests. |
| Prepared registry validation / named-judge overrides | Preparation could assume answer defaults for a process factory; binding overrides could discard an explicit grading policy. | **Integration fixes:** preparation checks syntax without guessing factory defaults; named binding replacements preserve trusted policies. Effective inherited policies enter the prepared-data contract, rejecting stale caches that dropped them. | MaxLen/UpToMaxLen defaults, process/format binding preservation, explicit override precedence and stale-cache refusal tests. |
| Verifier/service failures, omissions, weights, floors | Existing exception, exclusion and target-dependent floor behavior. | Unchanged. New input provenance does not convert infrastructure failures to model mistakes. | Existing reward-failure/omission suites. |
| Inference records and feedback | Pinned recorder drops arbitrary metadata and its protocol identity omits global grading flags. | **New-contract provenance:** persist per-response selection; extend final-mode evidence keys with the effective global grading policy. Default-off keys, admission and rewards are unchanged. | Actual pinned-recorder persistence, feedback/key parity and contract-separation tests. |
| OPD, direct callers, non-MILES GRPO | Their own extraction contracts. | Unchanged; no global migration of these workflows is claimed. | OPD and direct-call tests. |

## Limits and remaining qualification

Literal thinking delimiters inside quotations, code, JSON strings or tool arguments are not distinguishable from structural tags using decoded text alone. Last-marker selection retains that existing limitation; a literal opener at the start of a selected response now triggers the documented reopened-reasoning gate. Interior openers are not interpreted by the shared eligibility check; native format parsers may impose their own syntax restrictions. Selected-input NeMo and long-context consumers no longer independently reinterpret opener text; historical direct callers retain their prior checks. Do not treat marker counts as a complete conversation parser. Multi-turn trajectories are retained for full-scope token consumers, not decomposed into per-turn thinking/response pairs.

Bad targets can still be hidden by an eligibility gate if that verifier has no independent target validation before scoring. Returned timeout semantics, permissive native checker rules, schema-only rewards and judge score parsing remain separate audit items. This refactor does not certify service deployment, gold-label correctness, external experimental modules or all native Reasoning Gym subtasks.

The pinned recorder's original protocol omission also affects historical/default-off evidence under different truncation settings. That existing upstream issue is not retrospectively repaired here; old records must not be assumed comparable merely because their old input keys match. New per-row grading identity supplements the base manifest, and runtime/verifier revision provenance is still required. The selector schema version must change if a later implementation changes its grading semantics.

H180 found mixed accuracy and termination changes and retained an IF normalization confound even after reward flow was restored. It does not establish that the cleaner objective improves stopping. These code changes introduce no format bonus, new training run, sampling change or production-default change. Saved-response replay and a controlled training comparison remain separate evidence from CPU regression tests; passing input-contract tests does not establish training quality.
