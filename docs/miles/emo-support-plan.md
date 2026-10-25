# EMO checkpoint ingestion

This change lets MILES load checkpoints with active EMO metadata and train their
weights with **full-pool token routing**. It does not continue EMO's stochastic
restriction of experts to a document-level pool. Prompt-selected pools are the
next implementation stage, described below; they are not enabled by this change.

This work is deferred. The retained implementation is consolidated on
`robertb/emo` in Open Instruct, OLMo-core, miles-olmo-core and olmo-sglang.
Consolidation preserves the pinned implementation and design notes; it does not
reopen the ingestion PR or qualify additional execution modes.

## Supported inputs and configuration

| Checkpoint | Selection | Result |
|---|---|---|
| HF checkpoint with active EMO metadata | Set `model.emo_routing_mode = "full_pool"` | Prepare a full-pool copy, preserving source settings |
| Native Core checkpoint with EMO router config | Set `format = "olmo_core"` and `emo_routing_mode = "full_pool"`; retain the usual tokenizer/template configuration | Resolve native EMO settings before HF conversion |
| Previously prepared full-pool export | Its saved mode can be reused | Preserve the original source provenance |
| EMO-descended SFT export with absent/null EMO metadata | Omit the EMO option | Use the existing ordinary MoE path |

Add the following to a maintained [run configuration](workflow.md):

```toml
[model]
source = "/path/to/emo-checkpoint"
emo_routing_mode = "full_pool"

[trainer]
use_rollout_routing_replay = true
router_aux_loss_weight = 0.0
router_z_loss_weight = 0.0
```

Keep the configuration's other model, tokenizer, data and topology settings.
The initial implementation requires the default
`router_aux_count_source = "executed"`, softmax gating and L1-normalized expert
weights. It rejects unsupported modes and malformed metadata rather than
silently selecting another policy. `full_pool` is the only accepted explicit
mode; there is no `off`, `native`, or `prompt_selected` value yet.

Run `python -m open_instruct.miles plan RUN.toml` to inspect the
configuration, and use the [pinned runtime workflow](launching.md) for preparation
and training. The EMO branch requires its own [runtime pins](../../runtime/miles/runtime.lock.json);
an older MILES application image does not contain the EMO extensions.

## What preparation and training do

Preparation copies configuration/tokenizer assets and links HF weight files,
without modifying the source. For native checkpoints it first resolves the saved
router settings in an owned config, then uses Core's checkpoint converter.
An unset native evaluation-pool size can be resolved by the explicit full-pool
selection; other malformed settings still fail validation.

The prepared config records `emo_routing_mode = "full_pool"`, sets
`emo_eval_document_expert_pool` to the number of routed experts, and retains the
original EMO metadata in `emo_source_config`. Existing provenance is preserved
on repeated preparation/export. Changing the requested mode invalidates prepared
model reuse and same-run recovery.

Every token selects its top-k experts from the entire expert set: for example,
16 out of 512. Source minimum/maximum document-pool settings remain metadata and
do not restrict training or inference. Latent projections, shared experts,
normalization, scaling and model weights retain their checkpoint semantics.

During policy scoring and training, Core replays the expert IDs captured by
serving while recomputing mixing weights from current router parameters. Router
and expert weights receive gradients, including through activation recomputation.
Fresh reference scoring uses full-pool routing. Neither path samples document
pools or depends on EOS-derived routing segments, so EOS-free packed samples do
not share an EMO selection boundary.

The MILES adapter rejects incompatible replay/auxiliary-loss settings before
model allocation. The serving adapter validates EMO metadata before model
startup. Source metadata is carried through native/HF configuration export.

For optional route/gradient diagnostics, use
`core.replay_diagnostics = true` and `core.router_diagnostics = true`; see
[router diagnostics](core.md#detailed-router-behavior-diagnostics). These are qualification
tools, not required for normal execution. The synthetic fixture and same-token
numerical tools live in `scripts/miles/emo_smoke.py` and
`scripts/miles/emo_numerics.py`.

## Validation and limits

The implementation coordinates Open Instruct with these dependency changes:

| Dependency | Pinned revision | Contribution |
|---|---|---|
| OLMo-core | `9ea9a37a576276b076e2d6d9b78ad36410ce5bfe` | HF factory/export metadata, full-pool EMO router and replay support |
| miles-olmo-core | `ac801956bcbeb4d829db9dc89b553754c87e3cc6` | Training constraints and router diagnostics |
| olmo-sglang | `7d44e6c5594266cdb559ab389bdc3fd67785a90b` | EMO serving metadata guard |

Each revision is published on its repository's
`robertb/emo` branch. The serving input-aliasing fix found
during earlier EMO validation is already in the upstream serving base.

CPU coverage includes actual tiny HF/native checkpoint preparation, weight and
tokenizer preservation, fresh Core reload, configuration constraints, serving
metadata, replay weights/gradients, packing and activation recomputation. The
native preparation test writes and reads a real Core distributed checkpoint;
it is not a mock of the converter. This does not qualify distributed RL recovery.

The October 23 rebase onto Open Instruct main `dff453f54` preserves the MILES
v0.1.2 runtime and rebases all three companion branches onto their current mains.
Validation passed 1,085 application CPU tests (one skipped), 23 focused EMO
runtime tests, 216 Core tests (13 GPU/optional-dependency skips), and 45 serving
configuration tests. Core coverage includes EMO replay, HF factory/import/export,
RoPE import, custom objectives, and a regression combining EMO with scaled RoPE
import; scaled-RoPE HF config export remains unsupported. The actual runtime
parser matches all 1,528 checked-in option definitions. Runtime tests used the
existing `open-instruct-miles-core-fd8b57d34` image with rebased sources mounted
read-only and no GPU devices. No image rebuild or new GPU run is claimed.

Historical validation on October 15, based on main `95a84bb57`, passed 880 application
tests, 538 runtime tests (59 skipped), 87 focused Core tests and 48 portable
serving tests. Repository lint, formatting, type checks, generated references and
the documentation build passed. Tests used mounted source revisions in the
cached binary runtime, without GPU devices. The earlier separate serving
model-import/aliasing test could not collect in that CPU-only runtime because its
SGLang import lacked `fused_moe.silu_and_mul`; it is not counted as passing.

The historical [October 2 EP1 barrier smoke](https://beaker.org/ex/01M3YWGRVDW84MDN266PHBG3SC)
used application `699a77556` and image `01M3YWCBRCADTBVZY25N98PG86`. It completed
four optimizer updates, 32 responses and HF export on two B300 GPUs. All 64
scoring/training microbatches replayed exactly, router/expert gradients and
sampled parameter changes were finite and nonzero, and all five serving-weight
publication checks passed. Mean train/rollout logprob gaps were below 0.00085;
the maximum token gap was 0.00483, within the unchanged mean-gap guard of 0.05.
This run predates the current dependency rebase and is not a fresh GPU result
for the PR branch or the repository's GPU pytest suite.

Real checkpoints with active EMO metadata, distributed EMO RL, asynchronous
refresh, fresh-process RL recovery and production throughput still need their
own qualification. The historical smoke used a tiny synthetic fixture and does
not demonstrate learning quality. Support for the underlying model architecture
and topology remains subject to the [support matrix](feature-parity.md).

[H037](https://github.com/allenai/open-instruct/issues/1927) trained an EMO-descended
SFT export through ordinary `MoERouterV2` routing with inactive EMO metadata. It
provides real-checkpoint evidence for the ordinary path, not active-metadata
EMO ingestion or restricted-pool RL.

## Follow-up: prompt-selected document pools

The agreed next strategy is `prompt_selected`: select a fixed-size pool
**separately for each routed layer from the entire initial prompt**, then keep
those pools fixed throughout the response. Tokens still choose top-k within the
selected pool. Context-adaptive reselection, manually supplied task pools and
native stochastic document-pool training remain outside this follow-up.

The proposed implementation is:

1. Require an explicit pool size `P`, with `top_k <= P <= num_experts`. Use the
   complete prompt after templating/truncation, including valid special tokens,
   excluding padding and generated responses. Use authoritative prompt lengths.
2. Probe the prompt with full-pool routing at a pinned weight version. At each
   routed layer, sum FP32 softmax router probabilities across prompt positions
   and select the highest-scoring `P` experts, breaking ties by logical expert ID.
3. Discard the probe's KV/KDA state and prefill again using the selected pools.
   Capture the actual restricted token routes used for generation. Measure the
   extra probe/prefill cost separately from decode and training throughput.
4. Carry pool IDs, prompt identity/length, selector version, pool size and
   selection weight version with each sample through retries, filtering,
   truncation, packing, distribution and recovery. Keep the pool fixed during
   prefix rebuilds and weight refreshes; new requests select their own pools.
5. Validate replay IDs against their layer pools; recompute current mixing
   weights through the existing replay path. Reference scoring and fresh policy
   diagnostics use the same eligible pools with their respective model scores.
   Select synthetic final replay-row IDs from each pool rather than `0..k-1`.
6. Namespace restricted KV/KDA cache entries by pool identity, composing it with
   existing salt/LoRA identity. Never share probe state with restricted execution,
   or restricted state across different pools. Bind pools per request through
   batched prefill/decode, padding, graph replay and slot reuse.
7. Export strategy/size/selector metadata and preserve in-flight pools on recovery.
   New evaluation requests select their own pools. Unsupported consumers must
   reject this metadata. Qualify barrier execution before enabling async.

The two-pass selector is a proposed engineering choice, not a claim of equivalence
to native EMO document-pool training. Selection is discrete, as is existing route
replay. Using the whole prompt is valid for response likelihood conditioned on
that prompt, but does not establish causal prompt-token likelihoods; prompt
tokens stay outside the RL objective.

Acceptance requires independent selector/numerical references, response-token
exclusion, deterministic ties, prompt-chunk invariance, invalid-input rejection,
all-expert-pool equivalence, mixed-pool batch/cache isolation, packed EOS-free
samples, recomputation and recovery tests, then a bounded nonzero-gradient RL
smoke and real-checkpoint qualification. Existing full-pool results cannot
substitute for prompt-selected evidence. All expert weights remain resident;
restricted eligibility alone does not imply model-memory savings.

### Retained implementation details for the follow-up

These constraints preserve the more detailed prompt-pool design from the earlier
full-pool checkout. They describe future implementation and qualification, not
additional behavior supported by the retained full-pool code.

- Use one explicit integer pool size across routed layers, with separately
  selected IDs per layer. Do not infer a random size from pretraining metadata.
  Reject a pool-size setting in full-pool mode, empty prompts, non-finite selector
  scores and invalid IDs. Multi-turn/tool-driven pool reselection stays deferred.
- Compute FP32 softmax scores over all experts, mask eligibility for token top-k
  selection, then gather the original selected scores and preserve checkpoint
  normalization/scaling. Keep existing token-routing tie behavior separate from
  the deterministic aggregate-score tie rule used by the pool selector.
- Preserve the shared replay implementation's gating-specific formulas:
  `softmax(z).gather(I)` for softmax gating and `softmax(z.gather(I))` for
  top-k-softmax gating. Test raw selected weights and gradients before optional
  normalization so it cannot hide an arithmetic change. The EMO serving profile
  remains limited to its supported softmax configuration.
- Condition reference scoring on the rollout's recorded pool, using reference
  model scores rather than selecting a new pool or falling back to all experts.
  Measure that distinction explicitly in KL comparisons.
- Capture the actual logical expert IDs used for execution, before physical
  dispatch remapping, with correct token and routed-layer alignment. Do not run
  a second top-k solely for diagnostics or keep one global pool on a layer.
- Verify the pinned serving runtime's `cache_salt`/`RadixKey.extra_key` hooks,
  including KDA state. Compose pool identity with existing salt/LoRA identity.
  Reject or explicitly disable a cache path if it cannot isolate different pools;
  test cancellation, retraction, retries, mixed-pool batches and slot reuse.
- During refresh, distinguish original behavior probabilities/version spans from
  the final rebuilt replay table/version. Rebuild under the original fixed pools
  and retain behavior probabilities; do not relabel rebuilt routes as the routes
  originally sampled for every historical token. Model export must not install
  one request's selected pool globally. Preserve in-flight pools in recovery.
- Qualify serving TP1/EP1 first and trainer EP1 followed by EP2, preserving current
  trainer TP/CP/PP limits. Check every rank, packed/unpacked gradients and optimizer
  state, and malformed-input failures before unsafe collectives. Other serving
  geometries, quantization, speculation and prefill graphs need separate evidence.
- Report near-tie margins, route mismatches, logprob percentiles/maxima, greedy
  choices, importance-ratio tails and clipping fractions; a mean-only logprob
  guard is insufficient. Use independent references and several pool sizes, with
  probe/restricted-prefill overhead separated from warmed decode and update time.
