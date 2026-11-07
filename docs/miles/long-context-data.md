# Long-context RLVR sources

Short-context RL can improve math and instruction following while long-context
performance declines. This integration adds three open dataset families as named
MILES sources so we can test whether long-context training preserves RULER and
which kinds of tasks help. Qualification establishes loader and reward behavior;
it does not establish a learning benefit or reproduce a publisher's RL results.

| Source | Pinned Hugging Face release | Declared license | Supported contracts |
|---|---|---|---|
| `transduction` | [NVIDIA Transduction](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Long-Context-Transduction-v1/tree/90fa9816dd6e9a6bdb123a73fc1a2ed6408783e0) | OpenMDW 1.1 | All twelve arithmetic, UUID sorting, variable expansion and CSV variants |
| `docqa` | [DocQA-RL-1.6K](https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/tree/dbee14eee17e9cf39badb7526a8af316445ca2d5) | Apache 2.0 | Document multiple choice, numeric QA and short-answer QA; train split only |
| `golong` | [GoLongRL](https://huggingface.co/datasets/Kwai-Klear/GoLongRL/tree/34498d935a91b1f39c1e95f9dba6fcac90f04233) | CC BY 4.0 | Exact match, choice, math, document-set F1 and ranked retrieval |

Every prepared row has `metadata.pool_id="long_context"`, its source and subtask,
and the original record's revision, filename and row index. Source prompts remain
unchanged. These adapters do not alter existing dataset routes or sampling policy.
The existing [source catalog](first-pass-data.md) resolver turns qualified sources
into `[data.sources]` entries; the ordinary training preparation path still runs.

## Reward contracts

`LongContextVerifier` is a local, versioned deterministic verifier, with no judge
model or external tool calls. It examines the final region after the last explicit
`</think>`, when present. Set `core.reward_final_answer_only=true` when the run also
requires a closing thinking marker. Preserve the selected setting in comparisons.
Truncation receives zero through the existing `core.reward_zero_truncated` policy.

| Route | Reward and material differences from upstream |
|---|---|
| Transduction | Positional matching lines divided by the larger output/reference line count. Missing, extra, duplicate and reordered lines cannot earn full credit. This is a new RL contract for an SFT dataset, not an upstream reward. Every reference is independently recomputed from the input before admission. |
| DocQA numeric | Whole numeric answer with absolute tolerance 1e-6 or relative tolerance 1e-4. This is an explicit local protocol, not a claim of upstream scorer parity. |
| DocQA short answer | Case/whitespace-normalized whole-answer equality, allowing a trailing period and the published answer prefix. No substring credit; paraphrases and unlisted aliases may be false negatives. |
| Choice | A single extracted choice. Multiple contradictory extracted choices receive zero; arbitrary occurrences of the gold letter receive no credit. |
| GoLong short-answer QA | Normalized whole-answer equality, including textual labels. Upstream accepts the first line alone; this contract rejects additional contradictory lines. |
| GoLong EM/math | Whole-answer exact match, or the existing isolated math verifier, respectively. |
| GoLong document-set F1 | Set precision/recall F1, using upstream task assignments and normalization. Extra guesses reduce reward. |
| GoLong ranked retrieval | Linear-relevance NDCG at the number of reference documents. Duplicate guesses cannot earn repeated relevance; duplicate reference labels are rejected. |

GoLong's recall-only retrieval and partial-order families are deferred: the latter
can award full credit for a correctly ordered pair from a longer required order.
Its summary proxy and table QA routes are also deferred, pending a suitable reviewed
contract. Unknown task identifiers fail preparation rather than selecting a scorer
by guesswork. Dispositions distinguish unsupported contracts from context limits.

## Reproduce preparation and qualification

Download the three pinned snapshots and retain an `ACQUISITION.json` in each raw
source directory containing `repo`, `revision`, and a `files` list of relative
`path` and `sha256`. The inventory tool checks those hashes before processing.
It reads only training records and emits one disposition for every input row.

```bash
python -m scripts.data.rl_readiness.prepare_long_context \
  --raw-root /data/long-context/raw --output /data/long-context/inventory \
  --tokenizer /models/olmo-tokenizer --max-prompt-length 16384 \
  --max-reference-tokens 16384 --workers 4
```

This creates explicit 16K-prompt/16K-reference variants. Longer examples stay in
raw storage and receive budget dispositions; neither prompts nor gold answers are
truncated. The total model context must additionally accommodate generated tokens.
To qualify a longer variant, choose an explicit compatible budget and output
location and repeat preparation. Published dataset length labels are not used in
place of tokenization with the actual model tokenizer and chat template.

The inventory is **pending hygiene**, not a training release. Check its emitted
`questions/` against the evaluation references, retain the sensitivity reports,
and produce reviewed hygiene inputs and receipts. Shared boilerplate or document
passages require interpretation; they do not automatically establish question
contamination. Never turn the pending status into a clean receipt without checks.
Add each hashed receipt to the plan, then use the existing tools:

```bash
python -m open_instruct.miles.datasets.source_preparation \
  /data/long-context/reviewed-plan.json --output /data/long-context/prepared \
  --tokenizer /models/olmo-tokenizer
python -m scripts.data.rl_readiness.qualify_bundle \
  /data/long-context/prepared/catalog.json \
  --controls /data/long-context/inventory/controls.jsonl \
  --tokenizer /models/olmo-tokenizer --output /data/long-context/qualified \
  --max-prompt-length 16384
```

Review controls if hygiene removes their representative rows. Qualification requires
runtime gold, wrong, near-miss, empty and truncated controls for every retained
stratum, and runs the complete retained pool through the training loader. Any
preparation omissions must be explicitly reconciled with the qualification tool's
quarantine input. Keep the complete raw inventory alongside the qualified catalog;
a qualified context variant is not a claim that all upstream records are supported.

## Trial design

First exercise actual representatives in local MILES with the toy model and eight
responses per group. This validates generation, attribution, reward dispatch and
optimizer plumbing; random-model rewards do not measure dataset usefulness.
Then screen the starting policy for full correctness, reward variation, completion
length and truncation by source, subtask and input-length band. Fractional rewards
must be reported separately from binary pass@k.

Bounded learning trials use one node, eight responses per prompt, a common starting
checkpoint and 100 optimizer updates. Report generated input/output tokens, accepted
groups, throughput, wall time and pool repetition alongside endpoint changes.
Keep final and explicitly selected evaluation checkpoints, with bounded retention
for recovery checkpoints. Do not filter sources by whether the initial policy solved
their individual prompts.

The primary endpoint is the complete RULER 64K panel, using the same completion
format, data instances, context headroom and scoring for the starting checkpoint
and each trained checkpoint. Preserve task-level results and shorter-context RULER
results to localize changes. Add the ordinary math/IF/science endpoints to detect
regressions. A 16K training variant tests transfer to 64K; it is not a direct 64K
training experiment. Compare to a matched Dolci control before attributing preservation
to long-context data. Historical runs are contextual evidence if their starting
checkpoint, training budget or evaluation protocol differs.

Do not train on RULER evaluation instances. GoLong includes LongBench-derived tasks;
matching LongBench scores are not independent transfer evidence. DocQA's test split
is excluded from training, and shared source documents need to be considered when
interpreting any held-out DocQA evaluation.
