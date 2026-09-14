# Dense Olmo 3 through MILES/Core

Dense Olmo 3 uses the isolated `standard_models.py` adapter: Core's
`TransformerTrainModule`, AdamW and FSDP for multiple trainer ranks. Rollout,
rewards, orchestration and publication remain on the shared MILES path. Set
expert parallelism to one and disable router replay; dense models have no MoE
router. This does not select deprecated `open_instruct/grpo.py` or its vLLM stack.

## Current qualification

The full 7B Think-DPO model completed four updates across an initial process and
a fresh two-GPU colocated FSDP resume, native checkpoints, final HF export and
fresh serving reload on B300. See [lifecycle evidence](measurements/colleague-20260913/README.md)
and the [retained audit](measurements/colleague-20260913/dense-lifecycle.json).
The earlier disaggregated two-trainer/one-engine smoke and independent audit
also passed. These exercised batches had zero policy advantages; they establish
lifecycle, not nonzero-gradient full-model learning. Tiny synthetic models do
exercise nonzero updates and exact next-step checkpoint continuation.

The adapter preserves per-layer YaRN: sliding layers use ordinary RoPE and full
attention uses the released scaling. It canonicalizes layer keys across native
checkpoint JSON and retains the released HF `rope_scaling`/`rope_theta` descriptor
on export. FSDP diagnostics use persistent parameter references, not temporary
full-parameter views released by backward.

Independent synthetic logits were generated under release-era Transformers 4.57.0;
the installed newer Transformers forward has different rotary semantics. The
[reference report](measurements/olmo3-yarn-20260912.json) records nine BF16 lengths,
bit-exact weight round trips and explicit numerical tolerances. No runtime-wide
Transformers downgrade is required.

## Checkpoint and prompt identity

The proposed starting point is the published
[Olmo-3-7B-Think-DPO](https://huggingface.co/allenai/Olmo-3-7B-Think-DPO), revision
`7b18bf927b430ff06376fdfa5610eb3b1b6a5c38`: after SFT and DPO, before RL. This is
not the earlier experimental MoE SFT checkpoint.

The downloaded config is retained in
[`tests/miles/fixtures/olmo3-think-dpo-config.json`](../../tests/miles/fixtures/olmo3-think-dpo-config.json).
It describes 32 dense blocks, hidden width 4096, intermediate width 11008,
32 query/KV heads, three sliding layers followed by one full-attention layer,
a 4096 sliding window, and YaRN factor 8 from an original 8192-token context.

The checkpoint's HF chat template differs from the original open-instruct RL
script's `olmo_thinker` template. In particular, the latter defaults to
`You are a helpful AI assistant.` The copied original template is
[`configs/miles/templates/olmo-thinker.jinja`](../../configs/miles/templates/olmo-thinker.jinja),
SHA256 `eba6e269f669706e5c788e370160dad953137f3fa7014fa69d03c7b3ad9f0e72`.
Use it in both comparison arms; recording only the checkpoint name is insufficient.

Download the pinned snapshot first, then stage it without modifying its source:

```bash
python scripts/miles/stage_olmo3.py /path/to/pinned-snapshot /path/to/oi-template
python -m open_instruct.miles plan configs/miles/qualification/olmo3-think-gsm8k.toml
```

Replace `YOUR_USERNAME` and the model path in a copied run config. The stager
copies tokenizer/config metadata, references weight files through symlinks, and
records source and template identities. Retain the original snapshot. It does
not download weights or authenticate their HF revision itself.


## Running a first dense baseline

Copy [the dense qualification TOML](../../configs/miles/qualification/olmo3-think-gsm8k.toml),
replace its checkpoint/output paths and inspect `plan`. Use the selected sharing
image through the [launch guide](launching.md). Preserve checkpoint revision,
chat template, held-out identities and reward settings when extending a run.
Choose enough training prompts to produce mixed-reward groups and inspect actual
advantages/gradients; all-correct groups do not establish learning.

Dense KV sizing differs sharply from the KDA MoE model. This 32-layer MHA model
uses approximately 0.5 MiB of BF16 KV per cached token before other memory costs.
A 131,072-token pool is about 64 GiB; do not copy a 524,288-token MoE pool into the
dense profile. The observed SGLang implementation does not apply hybrid sliding
window KV memory savings. Long-context and high-concurrency dense capacity still
need their own measurements.

The [200-update mixture proposal](../../configs/miles/proposals/olmo3-think-dolci-200.toml)
is a proposal with future asset paths, not a ready-to-submit baseline. The
original [released RL script](../../scripts/train/olmo3/7b_think_rl_no_pipeline.sh)
used DeepSpeed/vLLM, so matching architecture and recipe does not establish
identical backend semantics. The
[historical recipe comparison](measurements/implementation-history/olmo3-pre-rl-before-sharing-20260913.md#preparing-the-recipe-comparison)
records known differences. Check immutable data proportions, templates, truncation,
TIS/filtering and token reduction before calling a run a reproduction.
