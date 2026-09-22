# Inference records

A training run's rollouts are its most expensive product, and the trainer keeps
only part of them. In the September 20 mixed 32K run, 30.6% of the 722M generated
tokens reached training. All-zero groups alone held an estimated 65% of generated
tokens. No per-prompt outcome was kept, so later runs could not avoid those
prompts, and the per-domain token split could not be computed.

Inference records keep the outcome of **every scored training group**, whether
the online filter passed it or not, and whether training later consumed it. They
live in a shared store that later runs and analyses read. Recording never changes
admission, filtering or training.

This covers phases 1 and 2 of the [inference records plan](plans/inference-records-20260921.md):
recording, and [summaries](#summarize-a-store). Prompt selection comes later. The
starter configurations leave recording off until a reliability qualification is
recorded.

Recording requires a fully async run (`[async] fully_async = true`), because it
lives in the completed-group buffer. `validate` rejects `records.enabled` in a
synchronous run.

## Enable recording

```toml
[records]
enabled = true
root = "/weka/oe-training-default/open-instruct-inference-records"
responses = "off"            # off, all, or sample
# response_sample_rate = 0.02  # required with, and only with, responses = "sample"
```

| Field | Meaning |
|---|---|
| `records.enabled` | Default false. When true, the completed buffer records every scored group. |
| `records.root` | Required when enabled. Use one shared absolute store for all runs. |
| `records.responses` | `off` (default) stores outcomes only. `all` also stores every response's text. `sample` stores the text of whole groups selected deterministically. |
| `records.response_sample_rate` | Fraction of groups, in (0, 1], whose text is stored with `sample`. |

The section maps to `core.records_root`, `core.records_responses` and
`core.records_response_sample_rate`; see the [configuration reference](configuration.md).
`plan` shows the resolved `records` section.

An outcome-only group record is about 1.5 KB, so a 13,000-group run adds about
20 MB. With `responses = "all"`, a run like the September 20 one would also
store several GB of text.

## Store layout

```text
<root>/<lineage>/<run name>-<run id>/manifest-<attempt>.json
<root>/<lineage>/<run name>-<run id>/records-<attempt>.jsonl
```

- **Lineage:** the first 16 hex characters of the SHA-256 of the starting
  checkpoint's **file inventory**, which workflow preparation records in
  `workflow-model.json`: source path, file sizes, modification times and JSON
  hashes.
  - Every run prepared from the same source shares a lineage directory.
  - It is not a digest of the weights; the manifest says `weights_hashed: false`.
  - Without the marker, the served directory's own inventory is used and
    `basis` says so.
- **Run ID:** the Beaker workload ID, or a random ID outside Beaker.
- **Attempt:** a random ID per process start. A resumed run writes a new manifest
  and record file beside the earlier ones.

The manifest records:

- `source`: `train`.
- The run: name, ID, attempt, `start_rollout_id`, loaded checkpoint and whether
  the run is fresh.
- The lineage.
- The `protocol` and its digest:
  - sampling settings and stop tokens;
  - `sglang_enable_deterministic_inference`;
  - hashes of the tokenizer, chat-template and generation-config files;
  - the verifier registry hash;
  - whether the zero-variance filter is on.
- The rollout seed, prompt data path and writer settings.

## Records

Each line is either a `group` row or a `disposition` row.

### Group rows

| Field | Contents |
|---|---|
| `observation_id` | Unique per scored group; links disposition rows. |
| `task_key`, `task_key_basis` | SHA-256 of the full rendered input, which includes every system and conversation turn, plus verifier targets. Exact under the recorded chat template. |
| `input_key` | `task_key`, the prompt-token hash and the protocol digest together. **Pool observations only within one `input_key`.** |
| `query_sha256` | Hash of the final user message only; a grouping hint, never an identity. |
| `prompt_token_sha256`, `prepared_sample_id`, `source_dataset`, `source_row` | Token-level and positional identity. Positional IDs change if a dataset is prepared again. |
| `verifiers` | Verifier names, which identify the domain. |
| `group_index`, `rollout_id`, `group_size`, `group_attempt` | Producer identity. `group_attempt` joins to `sibling_timing_*.jsonl` when timing observation is on. |
| `filter_decision`, `filter_reason` | `passed`, `filtered` (with the reason, for example `zero_std_0.0`) or `aborted`. Passing is not reaching training; see disposition rows. |
| `responses_included` | Whether response text is present for this group. |
| `responses[]` | Per-response fields, below. |

Per-response fields:

- `index`, `reward`, `reward_components` (name, score, weight).
- `validity`, described below.
- `response_tokens`, `status` and `truncated`.
- `policy_versions` and `policy_scope`, described below.
- `response`, the text, when included.

### Disposition rows

A group that passed the filter later leaves the completed queue as `consumed`
(given to training) or `expired` (past the policy-lag limit), with its
`staleness`. A passed group with no disposition row was still queued when the
run ended.

### Validity

Every expected verifier gets a state in `validity.components`:

| State | Meaning |
|---|---|
| `ok` | The verifier reported a normal outcome. |
| `completed` | The adapter saw the verifier return normally; that verifier reports no diagnostics of its own. |
| `timeout`, `rejected`, `service_error`, `judge_error` | The verifier or its service failed. The configured failure policy may still have assigned reward 0. |
| `unknown` | No status was recorded. |

`validity.valid` is `true` only when every component is `ok` or `completed`,
`false` when any component failed, and `null` when any is unknown. A null is not
evidence of success or failure.

### Policy scope

Weight versions count completed optimizer steps. A fresh run publishes the
starting checkpoint as version 0.

| `policy_scope` | Meaning |
|---|---|
| `start_checkpoint` | Generated entirely by version 0 of a fresh run. Interchangeable across runs of the same lineage and protocol. |
| `run_version` | Generated by one later version. Specific to this run's trajectory: version 10 of two arms are different policies. |
| `mixed` | Generated across versions, as refresh publication allows. A sample from no single fixed policy. |

## Coverage and guarantees

- Every group that reaches the completed buffer is recorded before the filter
  decides its fate, in barrier, engine-drain and refresh publication.
- Not recorded:
  - Groups the barrier and engine-drain homogeneity check returns to the producer
    before they reach the buffer. They are regenerated and recorded then.
  - Evaluation traffic.
- Rows go to a bounded background queue (4,096 rows). A full queue drops rows
  rather than blocking rollouts. The checkpoint lineage and protocol are read
  once, at startup.
  - If they are unavailable, or the store cannot be opened, the process logs
    one error and records nothing further.
  - Shutdown flushes the queue for at most 30 seconds.
- Metrics: `rollout/records/queued_total`, `written_total`, `dropped_total`,
  `failed_total` and `pending`.

## Summarize a store

```bash
python -m open_instruct.miles records summarize /weka/.../inference-records --output /tmp/records-summary
```

`STORE` is the records root or one lineage directory. The command reads only
complete JSON lines. It writes three files.

**`prompts.jsonl`** has one row per `input_key` and `policy_scope`:

- Group counts by outcome: `all_zero`, `constant`, `mixed` or `unscored`,
  classified from the rewards regardless of the filter's decision.
- Filter decisions, validity counts and truncated responses.
- `valid_reward`: count, mean, sample variance, min, max and an exact-value
  histogram over responses with `validity.valid = true` only. Fractional rewards
  keep their distribution.
- `unknown_validity_reward`: the same statistics for unknown-validity responses,
  kept separate and never pooled with valid evidence.
- `independent_units`:
  - for `start_checkpoint` rows, the number of distinct attempts, since they
    are independent draws from the same policy;
  - for other scopes, the number of distinct runs, since observations within
    one trajectory are correlated.

**`tokens.json`** gives, per domain (verifier names) and disposition, the groups,
responses, tokens, truncated responses and truncated tokens. Dispositions are:

- `consumed`: given to training.
- `expired`: past the policy-lag limit.
- `unused`: passed the filter but was still queued at shutdown.
- `filtered:<reason>`: dropped by the filter.
- `aborted`.

**`summary.json`** holds the input snapshot (every manifest and record file with
its size and SHA-256), the lineages, protocols, counts and warnings. Warnings
cover:

- incomplete final lines from a stopped writer;
- record files without a manifest;
- dispositions without a group row.

## Reading records

```python
import json
from collections import defaultdict
from pathlib import Path

rewards = defaultdict(list)
for path in Path("/weka/.../inference-records/<lineage>").glob("*/records-*.jsonl"):
    for row in map(json.loads, path.open()):
        if row["kind"] != "group":
            continue
        for response in row["responses"]:
            if response["validity"]["valid"] is True and response["policy_scope"] == "start_checkpoint":
                rewards[row["input_key"]].append(response["reward"])
```
