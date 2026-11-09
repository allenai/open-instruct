# Long-context RLVR sources

DocQA-RL and GoLongRL provide document reasoning, retrieval and other long-context
training tasks. They are integrated into the Tagliere dataset preparation workflow
in [RL Data Workbench](https://github.com/allenai/rl-data-workbench), then consumed
by MILES alongside the other prepared sources. Open Instruct owns the source
adapters, reward contracts and runtime validation. Qualification establishes
loader and reward behavior; it does not establish a learning benefit.

## Preparation and runtime boundary

| Responsibility | Owner |
|---|---|
| Pinned candidate files, split roles, hygiene, cross-source duplicate assessment, pool assignments and immutable releases | RL Data Workbench |
| Source message/target adapters, deterministic verifiers, rendering and actual training-loader checks | Open Instruct |
| Mixture weights, sampler, prompt/response budgets, training and evaluation | The MILES run |

The Workbench candidate IDs are `LC-docqa` and `LC-golong`. Its
`configs/collections/tagliere-long-context-v2.toml` extends the existing Tagliere
source recipe; `configs/pools/tagliere-long-context-v2.toml` assigns both routes
to `long_context`, preserving source identity and native subtasks. GoLong math
rows use the existing math verifier while remaining in that primary pool.
Adapter identity is explicit metadata and must not be inferred from a subtask's
name. The Workbench export accepts an upstream pool label only when it agrees
with the pinned assignment, retaining the original label as provenance.

The current integration imports the existing 64K-qualified populations and their
hash-bound qualification receipts. Original snapshots retain unsupported and
longer rows; a different context variant requires separate preparation and
qualification. A source's historical clean receipt does not bypass the new
collection's hygiene policy. The new sources are checked with merged-main decon,
DocQA test inputs are protected from exact training overlap, and duplicate
assessment runs against the combined candidate pool. Existing Tagliere evidence
remains bound to its unchanged inputs. Exact-input duplicates are resolved by the
collection policy; exploratory fuzzy groups are not automatically removed.

| Source | Pinned release | Declared license | Raw train rows | Qualified 64K inputs before collection hygiene |
|---|---|---|---:|---:|
| DocQA-RL | [DocQA-RL-1.6K](https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/tree/dbee14eee17e9cf39badb7526a8af316445ca2d5) | Apache 2.0 | 1,591 | 1,589 |
| GoLongRL | [GoLongRL](https://huggingface.co/datasets/Kwai-Klear/GoLongRL/tree/34498d935a91b1f39c1e95f9dba6fcac90f04233) | CC BY 4.0 | 22,965 | 11,988 |

These counts describe the imported variants, not the final release membership.
The release manifest and attrition report give retained counts after collection
checks. The existing Tagliere development qualification level is preserved;
source qualification does not certify every example's content or all samplers.

## Use in a mix

Use Workbench's `export-open-instruct` and `bind-open-instruct` commands on the
versioned release to produce prepared JSONL, a namespaced verifier registry and
a run-local `data.toml`. Preserve the release's `origin_id`, `record_id`,
`release_id`, `labeling_version` and top-level `pool_id` fields. Choose sampling
weights in the training configuration; the dataset does not prescribe them.
Named-pool configurations use the consumer merged in
[PR #2187](https://github.com/allenai/open-instruct/pull/2187). This PR adds
the long-context adapters and rewards to that existing sampling implementation.

The 64K variant allows up to **65,536 rendered prompt tokens**. The completed
source trials reserved **16,384 additional response tokens**, giving an 81,920
context allowance. A run with a smaller input budget selects a different
population; do not silently truncate prompts to make them fit. The trial
checkpoint declared 65,536 and used no RoPE; its longer total allowance was an
explicit setting validated by the canary and completed trials, not a default
recommendation for arbitrary models.

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
| Choice | A single extracted choice; contradictory extracted choices receive zero. |
| GoLong QA / exact match | Whole-answer equality. Unlike upstream first-line-only grading, contradictory extra lines are rejected. |
| GoLong math | Existing isolated math verifier. |
| GoLong document sets | Set precision/recall F1; extra guesses reduce reward. |
| GoLong ranked retrieval | Linear-relevance NDCG at the reference length; duplicate guesses cannot earn repeated relevance, and duplicate references are rejected. |

GoLong recall-only, partial-order, summary-proxy and table routes remain deferred.
Unknown task identifiers fail admission rather than selecting a scorer by guess.
These are versioned local contracts, not a claim of exact upstream scorer parity.
Fractional rewards are retained; full-credit pass@k and mixed numeric reward
probability are distinct screening measures.

NVIDIA Transduction is also supported by this PR's adapter and verifier, but is
not added to the Tagliere recipe above. Its first screen had zero full-credit
success and little mixed reward. Its positional line-accuracy contract is a new
RL interpretation of an SFT source, with independently recomputed references.
The pinned source is
[`90fa9816`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Long-Context-Transduction-v1/tree/90fa9816dd6e9a6bdb123a73fc1a2ed6408783e0),
under the publisher's declared OpenMDW 1.1 license.

## Qualification and existing evidence

All 13,577 imported rows passed the actual training loader; 130
gold/wrong/near-miss/empty/truncated controls passed across 26 strata. Both sources
subsequently completed fixed-policy screens and 100-update training trials at
64K input / 16K response budgets. The earlier 16K populations and screens remain
separate variants and should not be substituted for these results.

| Source | Screen pass@1 | Screen mixed reward at 8 | RULER64K mean change from SFT after 100 updates |
|---|---:|---:|---:|
| DocQA | 17.45% | 54.23% | -0.32 percentage points |
| GoLong | 2.99% | 20.29% | +1.36 percentage points |

These are single short trials. Length, task composition, reward semantics and
training cost vary together; the results motivate mixture experiments rather
than establish causal rankings. The compact trial review and original receipt
paths are recorded in Workbench's `reviews/lc64k-screen-transfer-20261108.json`.

The prepared 64K release and receipts are archived under
`s3://robertb-beaker/rl-dataset-qualification/long-context-rlvr-64k-20261107-v1/`.
The historical runtime application revision is
`dd1d9d988eee7d812ed416d60a103b9d07501395`. New Tagliere artifacts preserve those
receipt identities and record their own hygiene and runtime compatibility checks.

To prepare a new variant from pinned local snapshots, use
`scripts.data.rl_readiness.prepare_long_context --source docqa --source golong`
with explicit tokenizer, prompt and reference limits. Retain per-row dispositions,
run the Workbench hygiene workflow, then qualify through
`scripts.data.rl_readiness.qualify_bundle`. Pending inventory must never be
promoted to a clean or qualified release without the corresponding checks.
