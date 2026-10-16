# Atomic thinking tokens: SFT precision audit

The first experiment isolates optimizer precision. Keep token initialization fixed
until that comparison is complete. No model-level routing or quality results have
yet been collected for this fix.

## What changes

`TokenizerConfig.reserved_slot_tokens` promotes unused vocabulary slots to
`<think>` and `</think>`; it does not enlarge the embedding or output matrices.
`dataset_transformation.promote_tokens_into_reserved_slots` records the old token
decompositions. `model_utils.seed_embedding_rows` initializes the selected rows;
`olmo_core_utils.initialize_promoted_token_embeddings` handles Core SFT after
checkpoint loading. Resume skips initialization.

The historical experimental OLMoDDP path kept precise FP32 optimizer weights
separately from BF16 model weights. Its whole-matrix synchronization copied BF16
values back over both FP32 matrices after initializing two token rows. This also
rounded every unrelated vocabulary row. The new
`sync_promoted_token_optimizer_rows` copies only the selected rows and preserves
all other master values and optimizer moments. Standard optimizers without
separate masters require no additional synchronization.

Historical runs used open-instruct commit
`028b279adea84dce1e8d62fa2041884270854ffc` with OLMo-core commit
`89e7dcb739e2168f5f1022c49f35eec2167383b8`. See
[the original feature](https://github.com/allenai/open-instruct/pull/1911) and
[the ledger decision](https://github.com/allenai/olmo-post-training-ledger/issues/118#issuecomment-5963802040).
This branch is based directly on that historical open-instruct commit, preserving
its hero MoE V2/OLMoDDP dispatch, checkpoint configuration loading, runtime pin,
lockfile, Dockerfile and launch scripts. The initial main-based implementation is
retained on `robertb/atomic-think-sft-main-backup`; it is not the experiment branch.
The precision fix replaces the old whole-matrix synchronization in the actual
historical initialization path. There is no training-stack migration in this test.

The ledger's `bd68b1b2` launcher revision differs from the image's `028b279a` only
in launch scripts. Use the recorded launch arguments and immutable dependencies.
The training image must contain our patched source: launching the old image
unchanged would rerun the bug, regardless of the local checkout. Keep the runtime
fixed when building the patched image and record its source commit and image ID.

## Controlled comparison

Start every arm from the same **native checkpoint containing FP32 masters**.
An exported BF16 HF checkpoint has already lost the precision under investigation.
Keep hardware, runtime, tokenizer revision, data ordering, packing, batch size,
optimizer settings, seeds and training budget fixed. Record their exact versions.

| Arm | Tokenization | Optimizer master change at initialization |
| --- | --- | --- |
| A: control | Existing multi-token tags | None |
| B: rounding only | Same as A | Copy the full BF16 embedding and head back to FP32 masters |
| C: safe atomic | Atomic tags, existing mean-piece seed | Only the two promoted rows |
| D: historical atomic | Same as C, identical seed | Whole-matrix copy, as in the old implementation |

Compare A/B first. This tests the rounding hypothesis without changing tokenization
or sequence length. Compare C/D to confirm the effect within the atomic treatment.
Then compare A/C for the actual product change. Initialization alternatives are a
later experiment; changing them now would confound these comparisons.

1. Before training, compare every parameter, master tensor and optimizer moment.
   In C, require bitwise equality outside promoted rows; verify tokenizer round-trip,
   ordinary text, chat prefixes, decode behavior, and checkpoint save/reload.
2. In A/B, the initial BF16 forward weights should be identical. Merely rounding
   masters to their existing BF16 copies does not predict immediate rerouting.
   Repeat A/A probes to measure numerical nondeterminism before interpreting A/B.
3. Apply identical saved gradients to A/B for one optimizer step to isolate the
   precision effect. Check BF16 weights, ordinary-token logits, loss and natural
   routes. Then run normal matched SFT updates and probe at steps 0, 1, 2 and 10.
   Save RNG and mutable router state around probes so measurements do not alter
   the training trajectory. Include load-balancing biases and expert counters.
4. Use a fixed serialized probe batch, including document boundaries and a padding
   mask. Capture actual selected expert IDs for every layer, with no route replay.
   Report changed expert sets, retained experts, changes in ordering alone, and
   per-expert traffic. Measure selection margins using the router's actual
   selection scores, including any bias; raw logits can be the wrong scores.
5. If this short comparison shows a material effect, run paired SFT seeds and the
   deployment-matched forced-thinking evaluation. Report accuracy, termination,
   cap-hit rate, response length and ordinary capabilities with paired uncertainty.
   Routing divergence alone does not establish the cause of a quality regression.

Token positions in atomic and multi-token sequences differ. Do not compare their
route arrays position-by-position. Use ordinary text with identical tokenization
for direct route comparisons, and separately align thinking examples by textual
boundaries. Do not infer dead experts from a small probe with zero observed traffic.

## Route comparison artifact

`scripts/tokenizers/compare_sft_routes.py` compares captures; it does not launch or
instrument a trainer. Each artifact saved with `torch.save` contains:

- `input_ids`: integer token IDs for one probe forward.
- `doc_lens`: identical packed document lengths/boundaries.
- `valid_mask`: boolean tensor shaped like `input_ids`, selecting nonpadding tokens.
- `routes`: mapping of layer names to integer expert IDs shaped
  `(*input_ids.shape, top_k)`.

In Core MoE V2, the router forward's second return value is the selected expert IDs.
Capture that value from each routed-expert router, detached and copied to CPU.
Capture one forward only, excluding activation-checkpoint backward recomputation.
For distributed probes, save and compare each rank's aligned local inputs and
routes; aggregate counts, not unweighted rank percentages. If the runtime flattens
or partitions tokens, explicitly align the captured IDs with the local input view.

```bash
python scripts/tokenizers/compare_sft_routes.py control-step1.pt rounding-step1.pt
```

The comparator rejects mismatched inputs, missing layers and invalid routes.
Tests cover order changes versus expert membership changes and padding exclusion.
Separate CPU tests exercise row preservation with real DTensors on three ranks,
including flattened master shards whose boundaries differ from embedding shards.

## Later: initialization selected by model behavior

Constituent-token averaging is a heuristic, not an equivalence between a vector
and a token sequence. Hugging Face's
[default vocabulary expansion](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings)
uses the existing vocabulary's mean/covariance, a different policy. Its rationale
concerns limiting output-distribution disturbance, not teaching boundary semantics.

[Poolside's report, section 4.1.1](https://poolside.ai/assets/laguna/laguna-m1-xs2-technical-report.pdf)
describes unstable random special-token initialization in its larger MoE, followed
by constituent averaging and a 100-step phase training embeddings and the head
with the backbone frozen. That result does not separate averaging from warmup.

A follow-up can sample candidate rows and evaluate them in a frozen model on real
opening and closing contexts. Score activation scale, expert traffic and routing
stability on held-out contexts. Balanced traffic across examples is different from
nearly tied scores per example; the latter could increase routing sensitivity.
Compare against the model's normal traffic as well as uniform traffic. A separate
objective must teach the output rows when to emit the boundaries. A short phase
training only the new rows is an experimental option, not yet implemented here.

## Registered launch

The experiment is H077, preregistered in
[ledger PR 318](https://github.com/allenai/olmo-post-training-ledger/pull/318).
`scripts/train/atomic_think/` contains its frozen probe manifest, runtime overlay,
allocated-urgent Beaker spec and bounded gate/panel runner. The overlay derives
from the actual H030 image rather than rebuilding historical dependencies.

The gate runs four uninterrupted updates without probes, four with probes, and
resumes the reference at step two for two more updates. Exact batch, LR, model
and optimizer-state hashes must agree at steps three and four; otherwise stop
and investigate reproducibility before the 512-update panel. Checkpoint saving
is synchronous. Every scientific arm saves step 512 permanently, step 256 as
a recovery checkpoint, and the tokenizer alongside its full configuration.

The ordinary probe consists of 128 held-out MATH-500 problem statements, with
identical token IDs across arms. This limits the routing conclusion to this
mathematical-text sample; it does not establish broad quality preservation.
Another 64 reference-solution contexts include both tags and token offsets.
