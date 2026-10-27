# Temporary Olmo Hybrid port

The sibling `/home/mnoukhov/miles-olmo-core-hybrid` checkout owns native model
construction, strict HF/native interchange, streamed publication, and the SGLang
adapter. This checkout pins those runtime sources and adds the GSM8K recipe and
base-prompt dataset preparation. Both use branch `codex/hybrid-mainline`.

The adapter preserves the release FLA Gated DeltaNet implementation (including
negative eigenvalues) and uses SGLang full attention with NoPE, QK norms, and its
attention gate. It is deliberately restricted to BF16, TP=1, PP=1, eager execution,
no prefix cache, no chunked prefill, and no speculation. GDN states are maintained
per request slot outside SGLang's memory planner. This is a correctness bridge;
it does not yet offer a packed GDN serving kernel or optimized state allocation.

## Current evidence

- Native 275m configuration deserializes in the actual MILES/Core runtime. A meta
  model matches all 192 HF checkpoint tensor names and shapes.
- Four interchange/native-layout CPU tests pass; reference/old-policy/learner autocast and
  resume tests pass on the separate original Open Instruct branch.
- Rhea GPU qualification is [submitted](https://beaker.org/ex/01M5ZA7ZXSFH1T18AQFMX8P9RQ)
  for final midtrain 275m, 450m, 810m, 1.4b, and 2.7b. Submission is not a pass.
- End-to-end MILES learner updates, training-time publication, restart recovery,
  and GSM8K learning have not yet been qualified for this hybrid family.
- The original Saturn 1,000-update rerun remains on Open Instruct/Core/vLLM.

Validation: 148 companion CPU tests pass (run-file schema, runtime source preparation,
verifiers, and the hybrid recipe). Formatting, lint, compilation, and type checks
pass using the companion's existing environment. The newer release environment
exposed five type errors in unchanged older model files; no unrelated files were changed.

## Prepare and review a MILES run

Inside the candidate runtime, prepare data once:

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
`MILES_HYBRID_BRIDGE=1` must reach serving workers; the runtime selects the external
model package and registers the custom HF config in spawned children.

Build/publish on Beaker, following the root AGENTS instructions. The canonical
runtime Dockerfile installs the pinned release plugin without replacing the
binary Torch/SGLang toolchain. The Rhea qualification image is a source overlay
for serving tests; it is not a fully rebuilt image of this companion checkout.
Before launching this recipe, build this checkout, validate inputs, and run a short
native learner/publication/resume test. Use the standard `--miles` launcher with
an immutable validated image ID. No MILES training launch is claimed here.
