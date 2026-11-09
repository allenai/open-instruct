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
| Choice | A single extracted choice; contradictory extracted choices and an additional parenthesized capital letter on the same answer line receive zero. See the parsing limitations below. |
| GoLong LongBench QA | Case/whitespace-normalized whole-answer equality with answer-marker extraction; contradictory extra lines are rejected. |
| GoLong EM | Publisher's strict, case-sensitive whole-answer equality after thinking removal; no answer-marker extraction or punctuation normalization. |
| GoLong math | Existing isolated math verifier. |
| GoLong document sets | Set precision/recall F1; blank lines ignored, extra guesses reduce reward. |
| GoLong ranked retrieval | Linear-relevance NDCG at the reference length; blank lines ignored, repeated guesses keep their first occurrence, and duplicate references are rejected. |

Choice extraction is a format heuristic, not a general ambiguity detector. After
`The correct answer is (C)`, the same-line check rejects another `(A)`–`(Z)` token
regardless of its meaning. It permits option text and subsequent explanation
lines, with the following limitations (assuming the reference is `C`):

| Continuation after `The correct answer is (C)` | Behavior and limitation |
|---|---|
| `or D.`, `or D)`, `or [D]`, `or option D`, or `and (d)` | Still extracts `C`; these alternative formats evade the check. |
| A newline followed by `or (D)` | Still extracts `C`; an alternative on a later line can evade the check. A separate explicit `The correct answer is (D)` is still detected as contradictory. |
| `Vitamin (D) deficiency` or `(see paragraph (B))` | Receives zero even when this is legitimate option text or a reference, rather than another proposed answer. |
| A period and newline followed by `(D) is incorrect.` | Retains credit, allowing an explanation on the next line. |

The frequency of these missed hedges and false rejections has not been measured.
Account for both when interpreting screening rewards; the check does not certify
that every credited response asserts only one answer.

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
