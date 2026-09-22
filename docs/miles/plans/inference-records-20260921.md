# Plan: inference records (2026-09-21)

**Status:** phase 1 (recording) implemented on `robertb/miles-inference-records`
after review, and exercised on a two-GPU fixture; see [light exercise](#light-exercise-2026-09-21).
Phases 2 (summaries) and 3 (selection) are implemented. Phases 4–5 are proposals. The user guide is
[inference records](../inference-records.md).

## Problem

Every training run already does the expensive inference, but keeps only the
groups the trainer consumes. In the September 20 mixed 32K run:

- 30.6% of 722M generated tokens reached training.
- All-zero groups held an estimated 65% of generated tokens (at least 51%).
- A run draws 13% of the 101,434-prompt basket and never revisits a prompt, so
  an in-run skip list would never fire.
- Arms started from the same checkpoint regenerate the same prompts, and nothing
  carries their outcomes forward.

The legacy open-instruct path (`--no_resampling_pass_rate`) excludes only
high-pass prompts, judges each from one visit, and never excludes all-zero
prompts. The MILES path recorded nothing per prompt.

## Goal

Record outcomes of all inference, with identity, validity and coverage settled
up front, because they cannot be reconstructed later. Exclusion policies can
evolve once the evidence exists. The records support:

1. Skip modes for later runs.
2. Curriculum: per-prompt pass-rate trajectories across updates.
3. Post-training evaluation: per-prompt comparisons between checkpoints.
4. Measurement: per-domain token accounting and paired per-prompt comparisons across arms.

## Decisions

| Question | Decision |
|---|---|
| Store | One shared store with an explicit configured root; per-run append-only shards |
| Response text | Off by default; `all` or a deterministic `sample` of groups |
| Selection | Opt-in, frozen at plan time into a materialized manifest |
| Formats | JSONL is the raw record; Parquet only as a derived summary |
| Default | On for substantial runs, text off, **after** a reliability qualification; starters stay off until then |

## Review findings adopted (2026-09-21)

1. **The generating policy is not "start checkpoint + update N".**
   - Two arms at the same update have diverged weights, and a response spanning
     updates 9 and 10 is a sample from neither.
   - Each response records its policy versions and a `policy_scope`:
     - `start_checkpoint`: a fresh run's version 0, the only scope that is
       interchangeable across runs.
     - `run_version`: specific to this run's trajectory.
     - `mixed`: generated across versions.
   - The manifest records run ID, attempt, `start_rollout_id` and the loaded
     checkpoint, which together give the resume lineage.
   - The store directory is the starting checkpoint's **lineage**: a digest of its
     file inventory (paths, sizes, modification times, JSON hashes). It is
     labelled `weights_hashed = false` and is not a weight digest.
   - A future `max_policy_distance` is an explicit approximation and must be
     labelled as such.
2. **The prompt key covers the complete task.**
   - `task_key` hashes the full rendered input (every system and conversation
     turn) plus verifier targets. It is exact under the template recorded in
     the protocol.
   - `input_key` adds the protocol digest: sampling settings, stop tokens,
     template files, verifier registry and filter.
   - Observations pool only within one `input_key`.
   - `query_sha256` is only a grouping hint.
   - A template-independent content key needs message hashes recorded at data
     preparation; that is deferred and additive.
3. **Every expected verifier has a validity state, including `unknown`.**
   - The reward adapter now records `completed` for verifiers that return normally
     without their own diagnostics.
   - A missing status is `unknown`. `valid` is true, false, or null (unknown), and
     readers must not treat null as valid.
4. **Passing the filter is not reaching training.**
   - Group rows record `filter_decision` (`passed`, `filtered`, `aborted`).
   - The completed buffer adds linked `disposition` rows (`consumed`, `expired`).
   - A passed group without a disposition was left unused at shutdown.
   - Every group gets an unconditional `observation_id`.
   - Outcome, disposition and truncation are separate fields.
5. **Recording cannot stall training.**
   - Rows go to a bounded queue (4,096) drained by a background thread.
   - A full queue drops and counts rows instead of blocking.
   - Checkpoint lineage and protocol are read once at startup.
   - Metrics: `queued_total`, `written_total`, `dropped_total`, `failed_total`,
     `pending`.
   - Shutdown flushes for at most 30 seconds.
   - Qualification covers a blocked store as well as normal throughput.
6. **Selection must be conservative, with correlation made explicit.** See phase 3.

## Light exercise (2026-09-21)

[Experiment 01M33JYBPXMRVNVJ6T9JSXT5S3](https://beaker.org/ex/01M33JYBPXMRVNVJ6T9JSXT5S3)
ran source `a0723ea9e` on image `01M33JY20G7573M3K0KT1T1RAW`.

- **Fixture:** the two-GPU synthetic-reward fixture from 2026-09-19: tiny
  model, refresh publication, fully async, 4 updates of 4 groups × 2
  responses, zero-variance filter on.
- **Recording settings:** response text sampled at 0.5 into a separate test store.
- **Outcome:** exit 0 after 6.5 minutes. A read-only audit job
  ([01M33KQT787R1SK4Q9A5Y3593K](https://beaker.org/ex/01M33KQT787R1SK4Q9A5Y3593K))
  reconciled the records with the run's own accounting:

| Check | Result |
|---|---|
| Rows | 1 manifest; 60 group rows and 17 disposition rows; unique observation IDs; no dispositions without a group or for a non-passed group |
| Filter decisions | 20 passed, 40 filtered (20 all-zero, 20 all-one). The 34 drops logged by the filter metric through the last update report are included; the rest were produced during the final drain. |
| Dispositions | 16 consumed = 4 updates × 4 groups; 1 expired, matching the run's `stale_groups_filtered = 1`; 3 passed groups left at shutdown |
| Consumption | 32 consumed responses, whose lengths match `rollout_flow.jsonl` exactly |
| Identity | 32 task keys = 32 training prompts, revisited across the run |
| Policy scope | 80 `start_checkpoint`, 16 `run_version`, 24 `mixed` responses |
| Validity | 120 `unknown`, as intended: the fixture's custom reward bypasses the verifier adapter |
| Text sampling | 28 of 60 groups carry text; presence always matches the flag |
| Writer | 0 dropped and 0 failed; at most 3 rows pending at any metric report |
| Truncation | All responses hit the 256-token cap, as expected from a tiny random model |

### Real verifier, summaries and selection

1. **Real GSM8K verifier.**
   [01M33QYAAN23QD3M1B16M5GRCN](https://beaker.org/ex/01M33QYAAN23QD3M1B16M5GRCN)
   ran source `250767ccd` with the filter off.
   - All 72 responses have validity `completed`, the adapter status for a
     verifier without its own diagnostics.
   - The 16 consumed groups match `rollout_flow.jsonl` exactly.
   - `records summarize` ran inside the image over the store's two runs
     ([audit](https://beaker.org/ex/01M33RGADPN5NQD0QQRRVGGGST)). Its token
     account sums exactly: 96 groups × 2 responses × 256 tokens.
2. **Selection tests and table.**
   [01M33SPAE7GEWAV2D37MGFV383](https://beaker.org/ex/01M33SPAE7GEWAV2D37MGFV383),
   on image `01M33SN7VP0DRQBC69RYZ3FC36` from source `a561a4024`:
   - 52 tests passed inside the pinned image, including the real MILES data
     source's skip, ledger and resume test.
   - `records select` built a deliberately relaxed mechanics table from the
     real-verifier run: 2 observations, 1 attempt, bound limit 1.0. It excluded
     all 24 all-zero prompts that had starting-checkpoint evidence.
3. **Selected run.**
   [01M33SRPJZSY5ZAX810RW0ECH4](https://beaker.org/ex/01M33SRPJZSY5ZAX810RW0ECH4)
   pinned that table.
   - It logged more than 101 skips as the tiny dataset cycled.
   - Its records hold 36 groups over only the 8 remaining prompts, with 0 groups
     for an excluded key
     ([audit](https://beaker.org/ex/01M33TA4FZM01FXEGAXCWVEPH8)).
   - The manifest carries the table digest, the protocol matches the table, and
     the consumed responses match `rollout_flow.jsonl`.

These exercise mechanics only. The production qualification for the starter
default is still owed: it needs a real verifier registry (validity states other
than unknown) and 32K throughput with recording on and off.

## Phases

1. **Recording** (implemented): group and disposition rows, validity,
   policy scope, lineage and protocol manifests, text modes, bounded writer.
   - Tests: CPU tests, including a blocked store, a full queue, a missing
     checkpoint, the system-instruction collision and missing diagnostics.
     Buffer tests run against pinned MILES.
   - Qualification: a light run with rows reconciled against the filter's drop
     counts and consumed batches.
2. **Summaries** (implemented): `records summarize` builds the per-prompt table per
   `input_key` and policy scope.
   - Reward distributions: count, mean, variance and value histogram for
     fractional rewards, not binary "successes".
   - Invalid and unknown observations are counted separately.
   - Evidence is counted per independent unit (run, attempt).
   - The per-domain token account: passed, consumed, expired, filtered by
     reward value, truncated.
3. **Selection, frozen and materialized** (implemented as `records select` plus a
   `[selection]` table pinned by SHA-256, applied as prompts stream in):
   - `records select` resolves the rule against the store into a table. The table
     lists the exact excluded and readmitted prompts, every store file it read
     with its size and SHA-256, and the resolved readmissions.
   - The run pins the table's SHA-256 in `[selection]`, so the plan fixes the
     table before launch.
   - The data source skips excluded prompts as they stream in; prepared data is
     unchanged, and resume applies the same table.
   - Defaults are conservative:
     - only `start_checkpoint` scope unless explicitly widened;
     - a minimum of 16 valid observations from at least two independent attempts;
     - exclusion only when an upper confidence bound on the pass rate is below a
       stated threshold;
     - a nonzero readmission fraction.
   - Same-seed arms count as separate attempts only when deterministic inference
     is off: the manifest records `sglang_enable_deterministic_inference`, since
     upstream then seeds sibling i with `rollout_seed + i`.
4. **Fixed-checkpoint screening and evaluation import**, in the same schema with
   `source = "screen"` or `"eval:<suite>"`. A screening job covers prompts that
   training never visited and gives clean checkpoint comparisons. Evaluation import
   maps olmo-eval instance IDs to task keys.
5. **Later:** live selection for prompt recycling, mixture rebalancing, curriculum
   weighting, per-prompt group sizes, and a template-independent content key.

## Pitfalls

- **Small samples:** at a 10% pass rate, 0 of 4 happens 66% of the time and 0 of
  16 still happens 19% of the time. Prefer down-weighting sparse evidence to
  excluding it.
- **Mixture shift:** skipping all-zero prompts favors easy domains; report kept
  fractions per domain.
- **Correlation:** observations from one policy trajectory are correlated across
  updates; only `start_checkpoint` observations are exchangeable across runs.
- **Recipe change:** runs compared with each other must share a selection manifest.
- **Storage:** about 1.5 KB per group without text; full text is several GB per run.
