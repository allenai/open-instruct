# First-pass data for MILES mixing experiments

This release turns the retained datasets from the early screening effort into
named training inputs, so initial mixing experiments can start without repeating
the preparation investigation. It provides **2,288,298 source rows across 48 named routes
from 22 dataset families**, including ToolRL and the supported routes of all five
Nemotron blends. Qualification checks integration and representative reward
behavior; downstream experiments still need to establish learning benefit.

## Start here

The versioned release destination is:

```text
/weka/oe-training-default/ai2-llm/data/robertb/miles-first-pass-20261106-v2
```

`catalog.json` is the entry point. `summary.json` contains per-source counts and
coverage; `sources/<name>/` contains prompts, verifier configuration, controls,
qualification receipts, hygiene evidence and quarantine records. `STAGED.json`
records verification on shared storage. The corrected release is **offline
qualified; shared staging is pending**, including all 48 routes and 2,288,298 rows.
Wait for a verified `STAGED.json` before using the paths and commands below.
`configs/<name>.toml` provides a ready
single-source `[data]` fragment
with absolute shared paths. The versioned S3 archive
below reproduces this release. It contains compressed source archives, an archive
manifest and `stage_release.py`; the staging script verifies archive hashes,
extracts the complete release and verifies the catalog's file hashes. Use `s5cmd`
for bulk transfer; Beaker is used only for the small staging helper and CPU job.

| Release identity | Value |
|---|---|
| S3 archive | `s3://robertb-beaker/rl-dataset-qualification/miles-first-pass-20261106-v2/` |
| Final catalog SHA256 | `c332b260d99a3c65f0e39262d9900024f08ce2261ebf8455c2fa58d7143e353a` |

To restore elsewhere, use credentials with read access to this prefix (the local
transfer uses Robert's `halo` AWS profile). Download into a local archive directory,
then choose a **new** shared-storage destination:

```bash
mkdir -p /tmp/miles-first-pass-archives
s5cmd cp 's3://robertb-beaker/rl-dataset-qualification/miles-first-pass-20261106-v2/*' \
  /tmp/miles-first-pass-archives/
python /tmp/miles-first-pass-archives/stage_release.py \
  /tmp/miles-first-pass-archives /weka/oe-training-default/your-new-release-directory
```

The script writes `STAGED.json` only after verifying the release and regenerates
the config fragments with the chosen destination's absolute paths. The Beaker
helper is a small copy of the same script and transfer utility for the original
staging job; it is not needed for this restore command. `v2` names the release
prefix, not S3 object versions. On November 6, 2026, the bucket had neither S3
versioning enabled nor a lifecycle expiration rule. Preserve the catalog and
archive hashes; the prefix is not protected against later overwrites or deletion.

Version 2 supersedes v1 after verifier review: it quarantines 221 structured-output
rows with unresolved schema references and recovers one ToolRL string-union row.
It preserves the existing 80-row ToolRL holdout and all other retained prompts.
All six structured pools and ToolRL were requalified; the final image passed
all 135 controls for those pools and replayed code controls are recorded separately
(all 115 passed). `REVIEW_FIXES.json`, `local-controls-v2.json` and
`code-controls-v2.json` preserve the scope and evidence. Version 1 remains
available for reproducing the original training smoke.

The release records runtime code
[`54ffddf8f68d09f4ad9eb1ec0fd2c75e60e9af20`](https://github.com/allenai/open-instruct/commit/54ffddf8f68d09f4ad9eb1ec0fd2c75e60e9af20)
in `CODE_REVISION` and `VALIDATION.json`. Use Beaker image
`01M6SZCSXG790WNDXV1J4121Z0`, built from that revision; **253 regression tests
passed inside this image**. It retains the previously tested MILES runtime and
private execution-client pins. A subsequent preparation-only guard rejects
non-string source values outside their declared union; use preparation code at
`0fcedabb85377e27d2db5750645763a28c558057` or later when rebuilding raw ToolRL data. It reproduced all 3,587 converted records
exactly and passed a 247-test focused suite, so the released data and training
runtime are unchanged. The earlier GPU training smoke used image
`01M6SHC8Z3BF1GW3G9R9AFNPZY` at `2c622531eb33302a3e337c7db179ea6b24d15379`;
those GPU results are baseline evidence, not a claim that v2 was run on GPUs.

Requalify when changing reward behavior. A rebuilt MILES application image needs
the dependencies in `scripts/data/rl_readiness/requirements.txt`. The readiness
Docker overlay installs these on an application image built from matching code;
it does not replace the pinned MILES runtime. Building that overlay requires
`--secret id=github_token,env=GH_TOKEN` with access to the private execution client.
BuildKit mounts the token only for installation; do not pass it as a build argument.
The archive's `README.md` and `VALIDATION.json` are offline snapshots. This document
and PR #2178 record the current shared-staging status and credential configuration.

These are explicit **16,384-token prompt variants**, rendered with the bundled
Olmo Pythonic tool template and tokenizer. Allow response tokens in addition to
that prompt budget, for example a 32K total context with a 16K response budget.
A different tokenizer/template needs preparation again; a smaller prompt limit
needs an explicit smaller-context variant. `long-context/` preserves longer
records separately, with hashes and counts, rather than truncating histories.
These **64,478** records are not qualified for this release's prompt limit.

Resolve a deliberately chosen mixture **offline**, then put the resulting
`[data]` fragment in the run configuration:

```bash
python -m open_instruct.miles.datasets.source_catalog \
  /weka/oe-training-default/ai2-llm/data/robertb/miles-first-pass-20261106-v2/catalog.json \
  --source c031=0.5 --source c029=0.3 --source toolrl=0.2 \
  --output data.toml
```

These weights illustrate the interface; they are not a recommended recipe.
ToolRL has a different reward scale from the other two sources; see below.
Training trusts the resulting configuration and performs ordinary preparation.
It does not rerun qualification. Preserve the release/catalog hash with the run.

The qualification tokenizer came from the local
`rl-dataset-qualification/checkpoints/rhea-preflight-v3/checkpoint/tokenizer`
snapshot. Compare content hashes rather than inferring compatibility from a
model name:

| Bundled file | SHA256 |
|---|---|
| `tokenizer/tokenizer.json` | `18e309ad7f9c60037eaf26aca401a96128187cad2bdfa01a508dca7526bf6300` |
| `tokenizer/chat_template.jinja` | `43b0c225dd327d4af450809bfb3abfbd828eb19d6546d3e9ae33782464874d6a` |

## Available populations

Counts are retained **source rows**, not globally unique questions. Standalone
sources and blend routes can overlap, and some upstream blends repeat examples.
Select sources deliberately rather than combining every row by default. Earlier
hygiene exclusions and protected holdouts remain excluded; this release does not
claim a new independent decontamination pass.

| Dataset | Source name or prefix | Rows |
|---|---|---:|
| Litmus v0.2 | `c010` | 41,638 |
| Llama-Nemotron IF | `c016` | 51,189 |
| Nemotron Citation Formatting | `c018` | 9,540 |
| Nemotron Free-Form Formatting | `c019` | 9,037 |
| Nemotron IF | `c021` | 44,538 |
| Calendar v2 | `c023` | 9,659 |
| Structured Outputs v2 | `c029` | 57,485 |
| Nemotron Knowledge MCQA | `c031` | 610,469 |
| ReasoningGym | `c036` | 14,525 |
| Nemotron Cascade Math | `c037` | 14,379 |
| Nemotron OpenMathReasoning | `c039` | 112,647 |
| Nemotron Math Stack Overflow | `c040` | 434,156 |
| Nemotron Nano blend | `c043_*` | 60,243 |
| Nemotron Cascade-2 blend | `c044_*` | 60,459 |
| Nemotron Lightning blend | `c045_*` | 55,901 |
| Nemotron Super blend | `c046_*` | 269,333 |
| Nemotron Ultra blend | `c047_*` | 163,720 |
| AceCode | `c076` | 87,149 |
| AceReason Math | `c091` | 49,538 |
| DeepMath-103K | `c094` | 100,985 |
| Dolci IF | `dolci-if` | 28,207 |
| ToolRL | `toolrl` | 3,501 |

Blend names have route suffixes: `_code`, `_if`, `_mcqa`, `_structured`,
`_calendar`, `_rg`, `_citation`, `_format`, and `_tools`, where present. For
example, `c046_tools` selects Super's offline next-action tool examples. The
catalog lists the exact available combinations; unsupported routes are absent.

## What qualification establishes

Every retained row went through the training loader. Reviewed gold, wrong,
near-miss, empty and truncated controls went through the training reward bridge.
The **5,200 runtime control checks** passed: the original 5,195 plus five controls
for the recovered ToolRL row. Each receipt binds the checked prompt
and verifier files to their content hashes.
Code controls exercised the live service with complete test suites.

IF uses explicit **checker-family coverage**: a selected real constraint is
projected for each control, while training retains complete original constraint
bundles. Every represented family is covered. This is not a solved witness for
every combination. Other sources use their recorded reward/task strata. These
checks do not independently prove every reference answer or code test correct,
and no source was selected by whether a policy happened to solve it.

ToolRL retains its historical reward scale: call targets range from -3 to 4;
format-only targets from 0 to 1. NeMo tool message targets reward nonempty prose,
not semantic answer quality. Tool-call rows preserve their history and declared
tools, but grade the next action offline; they do not execute an environment.

Code uses deployment version 9 at the pinned execution endpoint. Inject
`CODE_EXECUTION_TOKEN` from Beaker secret
`robertb_MILES_PR2178_CODE_EXECUTION_TOKEN` in workspace `ai2/olmo-instruct`;
credentials are absent from the artifact. The endpoint is configurable through
the preparation plan's `code_service.api_url`, recorded in each prepared
`verifiers.json` as `config.api_url`. This release uses
`https://1n7avfskve.execute-api.us-west-2.amazonaws.com` and pins the service source
revision. Changing service behavior requires requalification.
The service owner reports zero failures
across 38,056 requests and 542,262 outcomes, tested through concurrency 128, with
an AWS ceiling of 900. That is a shared service limit, not a per-run entitlement.

## Training validation

The original v1 training smoke checks that prepared examples can be sampled, generated,
graded and used in optimizer updates with the tested application image. It uses
an Olmo 3.5 MoE SFT checkpoint on four B300 GPUs: two trainer ranks and two
inference engines. The deterministic integration pool contains 16 rows per route
(768 total), selected without policy-success filtering. It is not a substitute
for the full release or a representative accuracy evaluation.

| Check | Observed result | Evidence |
|---|---|---|
| Original v1 staging | All 48 routes and 2,288,518 rows verified against the catalog hashes | [Staging run](https://beaker.org/ex/01M6SMNTM585MPTKQ7WQ6Z611S) |
| First non-code training leg | 43 configured routes; 12 updates on both trainer ranks; finite losses and gradient norms; zero OOMs and omitted records; committed checkpoint at 12 completed steps | [Training run](https://beaker.org/ex/01M6SKMECBC8MGHADXA179ZSM1) |
| Code training | Five configured routes using the authenticated service; four updates on both ranks; finite losses and gradient norms; zero OOMs and omitted records; committed final checkpoint | [Code run](https://beaker.org/ex/01M6SNFM7KHH9X4DB3PXPAK9BW) |
| Non-code checkpoint resume | Restored at step 12 and completed through step 24 on both ranks; finite losses and gradient norms; zero OOMs and omitted records | [Resume run](https://beaker.org/ex/01M6SQ4TPZJC62B5QGYTX250RX) |
| Recorded route consumption and checkpoint cursor hashes | All 48 routes consumed: 576 non-code and 96 code prompt groups, totaling 2,688 training responses; saved sampler cursor hashes match checkpoint manifests | [Final audit](https://beaker.org/ex/01M6SSK3PCGEVYC7H8AD044NEE) |

The final audit's result dataset, `01M6SSK3PRPEPCXCH41KNJD8QV`, contains
`noncode-audit.json` and `code-audit.json`, including per-route consumption,
reward diagnostics, checkpoint clocks and cursor hashes. Both ranks completed
every expected update without skipping an optimizer step. All recorded rewards
were finite; the omission metric was present and zero in every collection.
The audit exposed an existing throughput-reporting bug: background collection
time was compared with the driver's shorter final await. The separate
[reporting fix, PR #2198](https://github.com/allenai/open-instruct/pull/2198),
was applied only to analysis, with its exact revision and file hashes recorded
in the audit reports. Training used the original v1 application image identified above.
The v2 fixes were checked through preparation, qualification and reward-bridge
regressions; no additional GPU run is claimed.

The smoke uses 16K response and 32K total-context limits. It retains groups with
zero reward variance to exercise the training path; some code batches therefore
produce zero gradients. Long code responses frequently reach the response limit.
Inspect response lengths and within-group reward variance when choosing a mixing
experiment's generation budget. Passing this smoke establishes integration, not
learning benefit.

The recorded sample had no positive rewards for `c010` (Litmus),
`c044_structured`, `c047_structured`, `c046_calendar` or `c047_calendar`. These are valid
zero scores rather than service failures; offline positive controls passed.
Structured/calendar parse errors describe model responses that failed the
required format, and ToolRL's negative scores follow its native reward scale.
Across all recorded code responses (including unused prefetch), 514 of 660
Nemotron responses reached the limit, versus 2 of 124 AceCode responses.
Nemotron code may need more than 16K response tokens to avoid mostly truncated,
zero-reward samples. These small-pool observations call for inspecting examples
and generation budgets in the first mixing experiments, not population-level
accuracy conclusions.

Startup, compilation and checkpointing dominate these short runs. For example,
the code run spent 326 seconds in inference startup and 85 seconds saving its
checkpoint; its first optimizer phase took 358 seconds and its last took 37
seconds. Substantial asynchronous prefetched work remained unused at shutdown.
These timings are not a sustained-throughput benchmark.

## Deliberate exclusions and follow-ups

| Area | First-pass decision | Follow-up |
|---|---|---|
| Malformed source rows | Quarantine exact IDs with reasons; retain the rest | Investigate repairs separately, without changing this release |
| Long histories | Preserve outside the 16K-prompt variant | Qualify a longer-context variant when an experiment needs it |
| Math duplicate labels | Quarantine two DeepMath rows labeled `B` versus `2` for the same numbered-choice question, and two AceReason rows whose `X` / `\text{X}` labels the existing verifier grades inconsistently | Fix the existing MathVerifier normalization issue in a separate upstream change |
| ReasoningGym halting | Quarantine three Ultra rows: native 0.1.25 applies `bool()` to strings, awarding every nonempty response full credit | Correct and validate the upstream scorer before admitting this family |
| Judged and hybrid routes, including Nemotron Math v2 (C038), proof, GenRM, rubrics, safety and abstention | Not included in this release | Implement each native prompt/parsing/aggregation contract and calibrate its service; do not substitute a generic judge |
| Workplace, SWE execution, Lean, browser and other interactive routes | Not included; offline tool support does not claim their execution environments | Integrate actual environments and recovery checks |
| Alternative valid code constructions, arithmetic grammar and language detection | Preserve the already documented upstream behavior | Investigate deliberate grading improvements separately; do not delay initial experiments for them |
| Schema combinations | Native list-traversal behavior is preserved; all six structured pools were requalified after the fix. The audit records 38 rows with object branches under `allOf` and 16 under `oneOf` | Investigate upstream schema satisfiability in the cleaning PR. For example, C029 direct-generation row 1378 requires a pet `id` through `allOf` while its parent forbids that undeclared key. Integration qualification does not prove every schema satisfiable. |
| Sustained verifier outages | Individual ungraded prompt groups are omitted and replaced; W&B reports `rollout/verification/omitted_records` | Decide the systemic-failure stopping threshold; none is introduced here |

The first experiments should examine reward distributions, omitted-group counts,
context utilization and downstream evaluations alongside learning curves. The
purpose is to identify useful signal and concrete failure cases, while keeping
this release and its qualification evidence reproducible.
