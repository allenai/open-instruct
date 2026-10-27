# Temporary Olmo Hybrid port

The sibling `/home/mnoukhov/miles-olmo-core-hybrid` checkout owns native model
construction, strict HF/native interchange, streamed publication, and the SGLang
adapter. This checkout pins those runtime sources and adds the GSM8K recipe and
base-prompt dataset preparation. Both use branch `codex/hybrid-mainline`.

The adapter preserves the release FLA Gated DeltaNet implementation (including
negative eigenvalues) and offers SGLang RadixAttention with NoPE, QK norms, and its
attention gate. The qualified temporary recipe instead selects the complete
release HF/SDPA backbone under SGLang scheduling and sampling, because the faster
packed/RadixAttention path failed strict numerical checks at some scales. It is deliberately restricted to BF16, TP=1, PP=1, eager execution,
no prefix cache, no chunked prefill, and no speculation. GDN and full-attention KV states in the fallback are maintained
per request slot outside SGLang's memory planner; the preallocated SGLang KV pool
is unused. This is a correctness bridge;
it does not yet offer a packed GDN serving kernel or optimized state allocation.

## Current evidence

- Native 275m configuration deserializes in the actual MILES/Core runtime. A meta
  model matches all 192 HF checkpoint tensor names and shapes.
- Nineteen runtime CPU tests pass (interchange, native layout, reference alignment,
  and serving-worker setup); reference/old-policy/learner autocast and
  resume tests pass on the separate original Open Instruct branch.
- All five final-midtrain sizes PASSED the BF16 HF-backbone fallback on an
  interactive Rhea A6000: 160 functional requests and 40 dense probes total,
  before/after reload, with queued requests, slot reuse, and prompts through
  3071 tokens. Maximum selected logprob error <5e-4; greedy agreement 100%.
  [Results and provenance](../../experiment.md#completed-bf16-reference-backbone-qualification).
- End-to-end MILES learner updates, training-time publication, restart recovery,
  and GSM8K learning have not yet been qualified for this hybrid family.
- The original Saturn 1,000-update rerun remains on Open Instruct/Core/vLLM.

Validation: 148 companion CPU tests pass (run-file schema, runtime source preparation,
verifiers, and the hybrid recipe). Formatting, lint, compilation, and type checks
pass using the companion's existing environment. The newer release environment
exposed five type errors in unchanged older model files; no unrelated files were changed.

## Prepare and review a MILES run

From this checkout in an environment with the release plugin, tokenizer dependencies,
and Weka mounted, prepare data once. The data helper runs host-side and is not
copied into the training image:

```bash
python scripts/data/miles/prepare_hybrid_gsm8k.py \
  /weka/oe-adapt-default/michaeln/open-instruct-hybrid/data/gsm8k-base \
  --tokenizer /weka/oe-training-default/yashasbls/hybridish-local/hf/v1/275m/midtrain/step0038147
python -m open_instruct.miles plan configs/miles/benchmarks/hybrid_275m_gsm8k.toml
```

Preparation uses the full original GSM8K training split and platinum test split;
only prompts over 512 tokens are dropped, with counts and resolved dataset
revisions recorded. Prompts exactly follow `Question: ...\nAnswer:` with no chat
control tokens. Verifier rewards are binary correctness. Dataset preparation
refuses to overwrite an existing output directory.

The recipe uses one learner and one inference GPU, 4 prompts x 4 completions,
1,000 updates, lr 1e-6, no KL penalty, 512/2048 prompt/response limits, native
checkpoints/evaluation every 100 updates, and barrier weight publication. It
retains zero-variance groups so a weak base model cannot stall collection.
Beaker auto-resume uses the same durable output root; three retries and a 24h
attempt cap are configured. Actual recovery still needs a GPU restart check.

The explicit `core.model_config` points to the paired native release descriptor;
keep this setting because the workflow may copy HF artifacts to a prepared path.
For another scale, change both the HF source and native descriptor together.
`MILES_HYBRID_BRIDGE=1` and `MILES_HYBRID_REFERENCE_BACKBONE=1` must reach serving workers; the runtime selects the external
model package, forwards the backbone selection, and registers the custom HF config in spawned children.
The recipe explicitly keeps core.fp32_lm_head=false: model computation and logits
are BF16. FP32-head diagnostics did not fix all faster-path mismatches. HF cached
versus full-sequence logprobs also differ in BF16 (450m max KL 0.00240536), so a
learner/rollout importance-ratio check is still required before a long MILES run.

Build/publish on Beaker, following the root AGENTS instructions. The canonical
runtime Dockerfile installs the pinned release plugin without replacing the
binary Torch/SGLang toolchain. The Rhea qualification image is a source overlay
for serving tests; it is not a fully rebuilt image of this companion checkout.
Before launching this recipe, build this checkout, validate inputs, and run a short
native learner/publication/resume test. Use the standard `--miles` launcher with
an immutable validated image ID. No MILES training launch is claimed here.
