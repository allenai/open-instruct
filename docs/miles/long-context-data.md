# Long-context prepared datasets

MILES consumes prepared document reasoning, retrieval and transduction tasks
using versioned source adapters and reward contracts. Dataset selection,
inventory, hygiene, pool assignment, screening and release assembly belong to
[RL Data Workbench](https://github.com/allenai/rl-data-workbench). Open Instruct
provides the adapters, verifiers and training loader that Workbench calls.

## Ownership

| Component | Owner |
|---|---|
| Candidate revisions, explicit train/held-out split roles, row dispositions and context variants | Workbench |
| Decontamination, duplicate policy, quality filters, pool taxonomy and immutable releases | Workbench |
| Qualification orchestration, screen/trial protocols, analysis and evidence | Workbench |
| Source messages and verifier targets, reward implementations and runtime loading | Open Instruct |
| Mixture weights, sampler, prompt/response budgets and training | MILES run configuration |

The adapters `lc_docqa_v1` and `lc_golong_v1` preserve source messages and native
subtasks. Adapter identity is explicit metadata, including through `routed_v1`.
Adapters do not assign a pool. Workbench's labeling policy decides that; GoLong
math can share a pool with document tasks while retaining its math verifier.

## Consume a prepared release

Render a declared model/context variant in Workbench, attach its versioned pool
labels, and assemble the release. Workbench's `export-open-instruct` and
`bind-open-instruct` produce prepared JSONL, a namespaced verifier registry and
a run-local `data.toml`. Use that binding with the [MILES run configuration](grpo.md).
Preserve `origin_id`, `record_id`, `release_id`, `labeling_version` and the exported
`pool_id` fields. Configure sampling weights in the run; the dataset does not
prescribe them.

Choose prompt and response budgets explicitly and ensure their sum fits the
validated model/runtime context. A smaller prompt budget defines a different
population. Silently truncating a long-context task can destroy answerability.

Workbench owns the `rl_data_workbench.readiness` commands that inventory sources,
qualify exported bundles through the actual loader/rewards, and exercise bounded
GPU coverage. These commands require the selected Open Instruct runtime. Loader
and reward checks establish compatibility, not the correctness of every example
or a learning benefit. Historical screens, trials and storage locations live in
Workbench alongside their pinned population and scorer identities.

## Reward contracts

`LongContextVerifier` is local and deterministic. It reads the final region after
the last explicit `</think>` when present. Preserve
`core.reward_final_answer_only` and `core.reward_zero_truncated` settings across
comparisons. A valid truncated response receives zero under the truncation
policy; a verifier failure is unknown rather than an incorrect answer.

| Route | Contract and limitations |
|---|---|
| DocQA numeric | Whole numeric answer with absolute tolerance 1e-6 or relative tolerance 1e-4. |
| DocQA short answer | Case/whitespace-normalized whole-answer equality, with the published answer prefix and optional trailing period. Valid paraphrases can be false negatives. |
| Choice | A single extracted choice; contradictory explicit answer declarations receive zero. See the parsing limitations below. |
| GoLong LongBench QA | Case/whitespace-normalized whole-answer equality with answer-marker extraction; contradictory extra lines are rejected. |
| GoLong EM | Publisher's strict, case-sensitive whole-answer equality after thinking removal; no answer-marker extraction or punctuation normalization. |
| GoLong math | Existing isolated math verifier. |
| GoLong document sets | Set precision/recall F1; blank lines ignored, extra guesses reduce reward. |
| GoLong ranked retrieval | Linear-relevance NDCG at the reference length; blank lines ignored, repeated guesses keep their first occurrence, and duplicate references are rejected. |

Choice extraction is a format heuristic, not a general ambiguity detector. After
`The correct answer is (C)`, it extracts `C` while permitting trailing option text
and explanations. It does not infer whether later references to other options
are proposed answers or explanatory comparisons.

| Continuation after `The correct answer is (C)` | Behavior and limitation |
|---|---|
| `and (D)`, `or (D)`, `, (D)`, `or D.`, `or [D]`, or `and (d)` | Still extracts `C`; a hedged answer can receive credit. |
| A newline followed by `or (D)` | Still extracts `C`; a hedge can also span lines. |
| `The correct answer is (D)` on the same or a later line | Receives zero because the explicit answer declarations conflict. |
| `答案应是 (C)`, `Vitamin (D) deficiency`, or `(see paragraph (B))` | Retains `C`; repeated confirmation and explanatory references do not invalidate the extracted answer. |

Standalone answers without the explicit prefix use stricter whole-answer parsing:
`(C)` is accepted, while `(C) or (D)` is rejected. This format dependence is a
known limitation. Credit does not certify that the response asserts only one
answer, and unsupported formatting can still cause false negatives.

GoLong recall-only, partial-order, summary-proxy and table routes remain deferred.
Unknown task identifiers fail admission rather than selecting a scorer by guess.
These are versioned local contracts, not a claim of exact upstream scorer parity.
Fractional rewards are retained; full-credit pass@k and mixed numeric reward
probability are distinct screening measures.

The `lc_transduction_v1` adapter also supports NVIDIA Transduction. Its
positional line-accuracy contract uses independently recomputed references.
The pinned source is
[`90fa9816`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Long-Context-Transduction-v1/tree/90fa9816dd6e9a6bdb123a73fc1a2ed6408783e0),
under the publisher's declared OpenMDW 1.1 license.
