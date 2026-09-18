# Router controls and standard-example qualification — September 18, 2026

## Runtime identity

The candidate runtime is Beaker image `01M2V3EGGYA2YFMYCAS1X6JSD7`
(`robertb/open-instruct-router-controls-dcc77a875`), application commit `dcc77a875`,
Docker `sha256:808e61ca7cb7ca5abee0bfd4f2579fff3a729eacb00ebfec5c94d79bdeb10037`.
It includes the router-control integration from `a753ca134`, Core revision
`ab64c30699d5c3de327830be6f4b2e2277a0edd3`, and reviewed standard examples.
Later guide-only edits do not change its application code. No runtime source
overlays are used.

The standard examples explicitly preserve pack grouping, token-weighted balancing
and z losses, and dispatched routing counts. Per-response objectives and current
routing counts are optional controls; neither changes the separate policy-loss
normalization. See [router objectives](../core.md#router-auxiliary-objectives).

## Scope and current evidence

| Configuration | Qualification scope | State / evidence |
|---|---|---|
| dev | Tiny full-attention MoE; one colocated GPU; four updates | [Completed](https://beaker.org/ex/01M2V3HEZC73FBC99VNYYQWP2V) |
| small | Same fixture; one trainer and one inference GPU; four updates | [Completed](https://beaker.org/ex/01M2V3HNVANJTHPSGTY5SQFAH6) |
| medium | Full policy; EP8, seven inference engines and one managed judge; two updates at the original 32K response budget | [In progress](https://beaker.org/ex/01M2V3WSK6NJRBBW6TVK5DG82B) |
| large | EP8/DP2, 32 inference engines and one judge, reserving 56 GPUs | [Cancelled at user request](https://beaker.org/ex/01M2V4N2NR58JD624QMN2C4AQM); outside qualification scope |
| Optional router controls | Three small variants: current counts; per-response grouping/averaging; their combination; replay and recomputation enabled | Corrected variants queued |

Dev and small completed their expected optimizer updates and initial/periodic
held-out evaluation, published updated weights, saved native checkpoints and
completed HF export. Their final workflow states contain rollout IDs 0–3.
The retention logs show the earlier checkpoint removed after the final commit.
A [read-only artifact audit](https://beaker.org/ex/01M2V4GAP7KMTWKWTE8P69Y7N9)
confirmed exactly one retained checkpoint, valid cursor checksums, readable export
shards and tokenizer, and sampled parameter changes for both.
Their tiny random model is a mechanics fixture: all sampled rewards were zero,
so policy gradients were zero. Router auxiliaries produced finite nonzero
router gradients and measured parameter changes. This is not learning evidence.

The three initial optional-control attempts failed before any optimizer update:
the qualification copies enabled packing but retained the small template's Torch
attention backend, which cannot mask packed documents. The maintained small
example itself does not pack and passed. Corrected variants use Flash Attention 4
with fresh output paths; this is a qualification-config correction, not a router
implementation change. Corrected runs:
[current counts](https://beaker.org/ex/01M2V4D69JAG89EJ8JH19RRRWH),
[per-response with dispatched counts](https://beaker.org/ex/01M2V4DCGE74Z6PEJSHK4F06WC),
and [per-response with current counts](https://beaker.org/ex/01M2V4DKHQA56Y0JZ1314F2CAN).
Scheduler events report the workspace group at its 160/160 GPU slot limit;
the corrected small variants remain pending, not qualified. Large was cancelled
at the user’s request and will not be launched again for this qualification.

The medium qualification copy shortens training from 200 to two updates,
evaluates/saves every update and uses eight held-out prompts per domain. It retains
the original training data, 32K generation budget, 34,816-token packing budget,
batch sizes and topology. This bounded run cannot establish sustained
throughput or recovery. Interrupted-run resume and fresh SGLang reload of exported
weights are separate checks; completion/export alone does not qualify them.

## Input and local checks

The tiny fixture is a two-layer, four-expert full-attention MoE with GPT-2
tokenization and pinned GSM8K preparation. Full-policy inputs and all judge shards
were checked on WEKA; correct/incorrect function and standard-I/O code canaries
returned the expected rewards. These checks establish input availability, not
mixed-workload training success.

The source config/docs/parser checks passed (100 tests). In the built image,
185 runtime-appropriate tests passed, six CUDA cases were skipped, and the
repository-link-only test was deselected because the application image omits a
legacy launcher referenced by the broader source documentation. That link check
passed in the complete source checkout. Lint, generated-doc checks and the docs
build passed; the docs build retained unrelated existing link warnings.

Exact submitted configs, template deltas, image provenance, launch receipts,
logs and downloaded results are retained in ignored
`runs/standard-examples-20260918/`. Run artifacts and weights remain under the
corresponding WEKA root. Keep the run/image identities with any reported result.
