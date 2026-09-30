# EMO support implementation plan

Status: full-pool implementation added September 30, 2026. CPU checks cover
configuration, export round-trips, replay weights/gradients and packed route
alignment. GPU serving, distributed training and lifecycle qualification remain
open. This document does not extend the current [support matrix](feature-parity.md).

## Selecting the implemented mode

For an EMO-bearing HF or native Core source, add these fields to a copy of a
maintained run configuration:

```toml
[model]
source = "/path/to/emo-checkpoint"
emo_routing_mode = "full_pool"

[trainer]
use_rollout_routing_replay = true
router_aux_loss_weight = 0.0
router_z_loss_weight = 0.0
```

Keep the rest of the copied configuration, including its topology and tokenizer
settings. Native inputs still need `format = "olmo_core"` and `hf_template`.
Preparation writes `emo_routing_mode = "full_pool"` and sets the evaluation pool
to all routed experts in the prepared copy. Original EMO settings are retained
in `emo_source_config`; source files and weights remain untouched. A missing
native evaluation pool can be resolved by this explicit selection. Other
malformed metadata is rejected.

Exports already carrying the full-pool mode can be loaded without another
selection. Ancestry-only SFT exports with absent/null EMO fields keep their
ordinary routing path and need no EMO option. Changing the option invalidates
prepared-model reuse and same-run recovery. The v1 trainer requires replay,
zero auxiliary coefficients and the default `router_aux_count_source="dispatch"`.
Fresh reference scoring uses full-pool selection without replay.

The synthetic final replay row still uses IDs `0..k-1`, which are valid in the
full pool. It is excluded from the policy loss. Existing auxiliary accounting
includes that row, but EMO v1 requires auxiliary coefficients to be zero.
No cache policy or async provenance semantics change: mixed-policy behavior
probabilities remain historical, while replay IDs describe the final rebuilt
forward. GPU qualification must cover those existing paths with active EMO
metadata before claiming production support.

V1 makes EMO checkpoints work correctly in MILES RL through full-pool
olmo-sglang inference and replay of selected token-level experts in Core.
It covers model loading, scoring, backward recomputation, publication,
evaluation and checkpoint recovery. Core already implements native EMO routing
and expert dispatch. We will connect that existing machinery to our RL path.

Every routed expert remains eligible during inference. For the 512-expert,
top-16 checkpoint, each token selects 16 experts from all 512. Replay fixes those
identities during policy scoring and training while recomputing mixing weights
from the current model. This trains EMO-pretrained weights; it does not continue
stochastic document-pool restriction during RL.

Restricted task/request pools, prompt-selected pools, adaptive pools, expert
pruning and a modularity-preserving auxiliary objective are outside v1.
No pool artifact, per-request selection algorithm or new pool-aware serving
state is required for this release.

## Implementation validation

The implementation branch has passed 113 workflow/configuration/recovery tests,
20 MILES adapter/packing tests, 66 focused Core tests (including a two-process
CPU export test), and 150 portable olmo-sglang tests. CUDA/FLA cases were skipped
where unavailable. Router replay checks cover both reentrant and non-reentrant
activation recomputation. Native config serialization and HF weight/config
round-trips preserve the full-pool mode and source settings.

Ruff, the changed application files' type checks, generated documentation inputs
and the documentation build pass. Full-repository type checking has five
pre-existing diagnostics in the OPD modules, reproduced on the base branch.
These CPU checks do not qualify actual SGLang capture, GPU graphs, KDA kernels,
EP2 policy training, publication or fresh-process RL recovery.

The runtime lock points to the implementation dependency commits. Until their
feature branches are published, materialize them from local Git caches with
`scripts/miles/prepare_runtime.py --cache olmo-core=... --cache miles=...
--cache olmo-sglang=... DESTINATION`; a remote-only image build cannot fetch
unpublished commits.

## Existing support and integration work

The inspection baseline is the Open Instruct [runtime lock](../../runtime/miles/runtime.lock.json):
Core `d6a08226f4a0ab53bed13fda680df969504fcb34`, MILES
`268f7118476dd889faeb5ce65b4f2becdae46d20`, and olmo-sglang
`b3a795f33e048dac401c89bb5823381146181ac6`. These identify inspected source,
not qualification of the proposed changes.

| Component | Existing machinery | V1 implementation |
|---|---|---|
| Core EMO router and dispatch | Native document-pool routing, single-device and synchronous expert-parallel dispatch | Reuse dispatch; add the RL full-pool and replay execution paths |
| Core HF factory | Architecture and weight import used by MILES; EMO fields currently rejected | Construct EMO checkpoints and separate source training metadata from resolved RL execution |
| Core replay | Standard router replay with current-score gradients; EMO currently rejected | Implement replay in the EMO router and update the supported-router check |
| olmo-sglang | Full-pool softmax top-k and route capture; config validation currently ignores `emo_*` fields | Add an explicit EMO execution guard first, then qualify loading, capture and numerics |
| MILES adapter | Sample alignment, packing, replay contexts and publication provenance | Connect and validate the EMO model through those existing paths |
| Export and evaluation | Full-pool inference is compatible with Core's current EMO HF export restriction | Serialize resolved full-pool behavior and verify fresh serving/evaluation reload |

The factory and replay rejections identify code we need to implement; they do
not mean Core lacks native EMO dispatch. No new expert dispatch kernel is planned.
Existing native EMO pretraining behavior must remain unchanged.

The serving guard is an immediate implementation item. Unlike the Core factory,
olmo-sglang does not currently reject active EMO metadata: its
`validate_olmo3_moe_config` checks gating and normalization, but never the EMO
fields. A restricted-pool config can therefore load and silently execute ordinary
top-k. Validate the requested execution semantics before model startup. Do not
blanket-reject every non-null EMO field: Core's valid full-pool exports can retain
training pool metadata while setting `emo_eval_document_expert_pool` to the
number of routed experts. See the
[serving validator](https://github.com/allenai/olmo-sglang/blob/b3a795f33e048dac401c89bb5823381146181ac6/src/olmo_sglang/config.py)
and [Core export check](https://github.com/allenai/OLMo-core/blob/d6a08226f4a0ab53bed13fda680df969504fcb34/src/olmo_core/nn/hf/config.py#L49).

The paired EMO/non-EMO SFT smoke used ordinary routing in both arms. It provides
bounded evidence for those weights, not coverage of EMO-bearing construction
and the new replay integration. See the
[recorded smoke](https://github.com/allenai/open-instruct/blob/813bd5988beb16be5b4d879ee3e2c49d8d859ee5/docs/miles/measurements/hero-rl-smoke-20260924.md).

## Execution contract

Resolve full-pool execution once for the RL run and use it consistently across
generation, policy scoring, training, reference scoring and evaluation. Preserve
source EMO metadata as provenance without letting training-mode pool sampling
silently change the resolved policy.

For inference and fresh scoring, compute router logits in FP32, softmax across
all routed experts, select token top-k, and apply the checkpoint's normalization
and scale. Preserve latent projections and shared experts. Reuse existing
ordinary-routing arithmetic, including its tie behavior.

For replayed scoring and backward, compute current router scores, use supplied
expert IDs instead of fresh selection, gather their weights and apply the same
normalization and scale. Bypass native document aggregation and pool-size
sampling. The override remains active through backward recomputation.

Preserve the gating-specific weight formula when sharing Core replay code.
For recorded IDs `I` and current router logits `z`, the existing formulas before
optional normalization/scaling are:

| Core gating function | Replayed mixing weights |
|---|---|
| `softmax` | `softmax(z, dim=-1).gather(-1, I)` across the full expert vocabulary |
| `topk_softmax` | `softmax(z.gather(-1, I), dim=-1)` across only the selected experts |

These are distinct operations before normalization. Core's shared replay code
must retain both branches; v1 serving remains limited to its supported softmax
profile. Test raw selected weights and gradients as well as the final normalized
output so normalization does not hide a changed formula. The existing branches
are in [router.py](https://github.com/allenai/OLMo-core/blob/d6a08226f4a0ab53bed13fda680df969504fcb34/src/olmo_core/nn/moe/v2/router.py#L690);
the custom-router rejection is in
[replay.py](https://github.com/allenai/OLMo-core/blob/d6a08226f4a0ab53bed13fda680df969504fcb34/src/olmo_core/nn/moe/v2/replay.py#L30).

The initial numerical profile is unquantized BF16 weights with FP32 router
computation, softmax gating and the normalization supported by the serving path.
Preserve and validate checkpoint settings; do not silently cast or reinterpret
an unsupported profile.

Record the resolved mode with existing run/checkpoint provenance. Keep
architecture, tokenizer/template, arithmetic and weight-version identities
available for comparisons and resume validation. V1 does not need a new
per-request routing artifact or transport protocol.

Native EMO's full-document aggregation can use later tokens to choose earlier
tokens' experts. It must not run in the v1 policy likelihood path. Fresh
reference and diagnostic scoring also use full-pool routing, so their behavior
does not depend on whether rollout replay is installed.

## Core implementation

Work primarily in `olmo_core/nn/moe/v2/olmo3.py`, `router.py`, `emo_router.py`,
`replay.py` and HF config/export. MILES selects the RL execution contract;
Core owns the router behavior.

1. Extend the HF factory to construct EMO checkpoints. Preserve architecture and
   source EMO settings, and represent the selected RL behavior separately.
   Cover both EMO-bearing checkpoints and ancestry-only SFT exports whose EMO
   fields are null.
2. Implement replay in `EmoRouterV2`, sharing score gathering, normalization and
   scaling where appropriate. Replace the blanket subclass rejection with an
   intentional supported-router interface; simply removing the guard would
   leave EMO ignoring the override.
3. Make replayed IDs take precedence over document selection in evaluation-mode
   scoring and training-mode forward/backward. Validate dtype, shape, bounds,
   distinct IDs and coverage before distributed work.
4. Preserve gradients through current mixing weights and expert computations.
   Retain replay through activation recomputation and restore previous state
   after success or failure, including nested contexts.
5. Provide full-pool fallback for reference scoring and fresh-route diagnostics.
   Do not rely on `train()` versus `eval()` to decide whether stochastic
   document routing is enabled.
6. Reuse existing dispatch and document-isolated attention/KDA execution.
   Replay/full-pool operation should not acquire an unnecessary dependency on
   EOS-derived EMO segments. Preserve native segment and RNG behavior for
   existing pretraining callers.

There is a concrete boundary mismatch to cover: MILES passes packed sample
`doc_lens`, while Core derives EMO routing segments independently from EOS token
IDs. With native document-pool selection active, a sample truncated without EOS
and its following sample share a routing segment despite distinct attention
boundaries. V1 must bypass document-pool selection and prove isolation with this
fixture. Supporting document pools later requires deriving routing boundaries
from authoritative sample lengths, not just EOS. See
[Core segment construction](https://github.com/allenai/OLMo-core/blob/d6a08226f4a0ab53bed13fda680df969504fcb34/src/olmo_core/nn/transformer/model.py#L614).

Retain the token/layer/top-k expert-ID payload. Under current weighting semantics
it contains the discrete choices needed for replay; prove this with an independent
reference. Replay is a fixed-route optimization surrogate, not fresh selection
under updated weights. Diagnose fresh-versus-replayed routing separately.

Start policy qualification with router auxiliary coefficients zero, matching the
maintained examples. Dispatch counts follow actual selected IDs. Keep explicit
restrictions on unqualified auxiliary modes; review current-score count handling
separately rather than adding a new balancing objective to v1.

## Serving implementation

Work in olmo-sglang `config.py`, `routing.py`, `models/olmo3_moe.py` and its
validation fixtures. Reuse ordinary full-pool execution. Change base SGLang or
the MILES serving adapter only if capture or lifecycle validation identifies a
concrete missing capability.

1. Accept supported EMO checkpoints and validate full-pool execution. Log the
   resolved mode and arithmetic profile. Implement the guard before the broader
   integration: ordinary absent/null fields and valid explicit full-pool EMO
   exports remain accepted; restricted evaluation pools, malformed fields and
   unresolved EMO execution fail with an actionable message. Missing evaluation
   pool settings must not silently imply full-pool behavior. An explicit v1
   full-pool selection can resolve source training metadata during preparation;
   preserve that provenance separately and validate the resulting serving config.
2. Verify logits, selected IDs, mixing weights, scaling and model outputs against
   the reference and Core. Retain automatic fused rounding where its existing
   architecture/settings contract applies.
3. Reuse the existing capture of actual top-k selections. The inspected SGLang
   path already calls `capture_routed_experts_if_allowed` on the selected IDs;
   this needs integration tests, not a second capture implementation. Preserve
   logical expert identities when dispatch later remaps to physical experts.
   Check prompt/decode positions and logical-layer alignment, including dense
   layers; do not independently rerun top-k solely to produce a capture.
4. Qualify full/chunked prefill, cached decode, mixed lengths, batch padding,
   slot reuse and decode CUDA graphs under the same full-pool policy.
5. Exercise existing KV/KDA cache and weight-update behavior: retraction,
   cancellation, rebuild and changed weights. Preserve graph storage contracts.
   Validate radix reuse separately before claiming it for the new integration.
   V1 has no request-specific pools and adds no EMO-specific blanket cache-disable
   requirement; preserve the existing workload's cache configuration and test it.

Qualify TP1/serving EP1 first and trainer EP1 then EP2. Preserve current trainer
TP/CP/PP limits. Higher serving TP/EP, quantization, speculation, prefill graphs
and other geometries need separate evidence; they are not v1 prerequisites.

## MILES and Open Instruct integration

Use `python -m open_instruct.miles` and the committed-image workflow.
MILES owns the adapter, transport, publication and recovery; Open Instruct owns
researcher configuration, preparation and validation.

Add only the configuration needed to select and report full-pool EMO execution.
Final field names should follow schema review. `plan` reports mode and topology;
preparation validates the checkpoint and runtime. Do not add a general
request-pool configuration system.

Require the implemented EMO replay path for the v1 policy-training recipe.
Reference scoring retains independent fresh routing, explicitly full-pool.
Check standalone scoring, training-forward scoring and scoring-skip paths.

Preserve expert IDs through filtering, retries, truncation, DP distribution and
packing. Reuse authoritative sample lengths and existing attention/KDA boundaries.
Retain each sample's synthetic final replay row, currently IDs `0..k-1` appended
in `core_utils/data.py`. Those IDs are valid in v1's full expert range. The row is
unscored for policy loss and cannot influence earlier routing. Document its
existing auxiliary-loss contribution rather than changing it incidentally.

For mixed-policy refresh, preserve the distinction between original sampling
probabilities/version spans and the final rebuilt replay table/version.
Do not describe final replay IDs as original sampling routes for every historical
token. Rebuild with new weights under the same full-pool contract and retain
original behavior probabilities. Check historical-prefix and latest-forward
scores separately. Start new async qualification with policy lag six.

Persist resolved execution mode in run state, native checkpoints and exports;
reject incompatible resumes. A fresh process must recover the same next-update
behavior under controlled inputs and RNG. Export full-pool inference settings
while retaining source EMO provenance separately. Verify native save/resume, HF
export, fresh serving reload and evaluator loading. Restricted-pool export is
outside v1.

## Validation and release gates

Use a small independent PyTorch reference for expected selections, weights,
outputs and gradients; it must not call the production selector for its answers.
Include checkpoints with active source EMO fields so tests cannot pass solely by
exercising ancestry-only exports. Declare tolerances before assessing candidates.

| Gate | Required evidence |
|---|---|
| Construction and arithmetic | Loader guard accepts ordinary and explicit full-pool exports, rejects restricted/ambiguous/malformed configs; EMO-bearing and ancestry-only imports; unchanged weights/architecture; full-pool equivalence; gating-specific weights/gradients, scales, ties, invalid IDs and native EMO regression |
| Core replay | Exact IDs in scoring/backward/recomputation; finite nonzero router/expert gradients; evaluation/training agreement; nested/error cleanup; no pool resampling |
| Causality and packing | Future changes cannot affect earlier scores with fixed earlier routes; fresh full-pool scoring is causal; packed samples remain isolated; EOS-free truncation and synthetic final rows are covered |
| Serving numerics | Controlled-reference logits/logprobs; actual route capture; full/chunked prefill, cached decode, mixed lengths, slot reuse and graph replay |
| Distributed training | EP1/EP2 alignment; packed/unpacked policy loss, gradient and optimizer-state comparisons; every rank completes; malformed inputs fail before unsafe collectives |
| Publication and recovery | Updated routers/experts match a fresh engine at the same weights; rebuild, retraction, cancellation, refresh and restart preserve mode/provenance |
| RL lifecycle | Tiny barrier run with nonzero policy gradients, then engine drain, actual cross-version refresh and save/resume; no lost or duplicated samples |
| Real checkpoint | Conversion audit and same-prefix numerical study, then real-reward RL, async and restart exercises on the selected topology |

Require exact IDs for supplied replay and matched deterministic reference
arithmetic. For fresh optimized selection, report near-tie margins, mismatch
rates and downstream effects. Track logprob mean, percentiles and maximum,
greedy choices, importance-ratio tails and clipping fractions. The historical
mean-only logprob guard is insufficient evidence by itself.

Use synthetic rewards for bounded mechanics and real rewards with held-out
evaluation and meaningful response budgets for the learning exercise. Compare
unchanged-weight EMO full-pool execution with the ordinary-routing reference.
Quality improvement over non-EMO is a research result, not an implementation
acceptance requirement. Restricted-pool comparisons are outside v1.

Measure warmed prefill/decode throughput, update time, memory, cache behavior and
expert load against the existing full-pool path. Do not promise memory savings
from EMO ancestry or replay.

## Delivery sequence

| Milestone | Deliverable | Completion condition |
|---|---|---|
| 1 | Serving guard, full-pool execution contract and independent fixtures | Silent EMO fallback prevented; agreed semantics represented in tests, including EMO-bearing configs |
| 2 | Connect existing Core EMO support to loading and rollout replay | Correct scoring/backward using existing dispatch, including recomputation, packing and EP2 |
| 3 | Complete and validate olmo-sglang EMO loading and full-pool inference | Eager/cached/chunked/graph numerics and actual capture pass |
| 4 | MILES adapter and Open Instruct configuration/preparation | Tiny barrier run exercises configuration through optimizer update |
| 5 | Publication, drain, refresh, export/evaluation and recovery | Cross-version and fresh-process lifecycle gates pass |
| 6 | Real-checkpoint qualification and performance report | Support matrix records exact images, configs and limitations |

Core and serving work can proceed independently after the contract and fixtures
are stable. Integrate both before lifecycle and real-model qualification. Pin
dependent revisions together in a new runtime image; change the binary base only
if runtime changes require it. Preserve ordinary-routing and native EMO tests.

Update support claims only as gates pass. Keep experiment configs in ignored
`runs/`, without adding a maintained example tier. Run repository linters,
focused tests and the dedicated MILES runtime suite for implementation changes.

Follow the [distributed launch contract](launching.md#distributed-scheduling-contract).
Each compute launch requires its own confirmation with cluster, GPU/node count,
priority, preemption behavior and timeout. GPU experiments default to high
priority. CPU WEKA audits are unallocated: try Saturn first, inspect scheduler
events and stop an unschedulable attempt before trying Jupiter. Agreement on
this plan does not authorize compute launches.

## Deferred work

Restricted execution selects token top-k from a smaller eligible subset.
Fixed task/request subsets, prompt-selected pools and adaptive prefix pools would
need additional inference semantics, metadata and cache/replay validation.
Physical pruning also needs storage and ID-mapping support. None is a v1
dependency.

For future restricted pools, use SGLang's existing cache namespace interface
before considering disabling reuse: `RadixKey` has `extra_key`, and request
`cache_salt` can feed that namespace. These are not two independent fields on
`RadixKey`. Compose pool identity with existing salt/LoRA identity rather than
overwriting it, and verify isolation in both KV and KDA cache paths and across
retraction/publication. A same-pool request should retain valid reuse; a
different-pool request must not share incompatible state. Verify these hooks
against the eventual runtime pin before implementation.

Restricted pools would also require the final synthetic replay row to choose
distinct in-pool IDs for each routed layer instead of unconditional `0..k-1`.
Native document-pool support must resolve the confirmed sample-length/EOS
boundary mismatch described above. These changes are unnecessary for v1's
full-pool replay contract.

Continuing native EMO's stochastic document-pool objective during RL is a
separate training-objective decision. Investigate whether full-pool RL degrades
useful modularity before adding that objective; keep any full-document
regularization separate from the causal policy likelihood.
