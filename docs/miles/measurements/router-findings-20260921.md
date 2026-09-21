# Router investigation: findings worth keeping

**Removing both router auxiliary losses improved sampled accuracy in the completed
three-seed toy study. Strong balancing changed routing substantially more, but we
did not prove that those routing changes caused the accuracy loss. The large
Core–Megatron routing difference was traced to padding entering Megatron's
auxiliary objective, not to Core's replay-count source.**

This September 21 consolidation supersedes tentative explanations in the
September 17–19 research notes. It records completed measurements, including
results that contradicted our hypotheses. It does not launch new experiments or
change training defaults. [The numerical evidence snapshot](router-findings-20260921/evidence.json)
preserves per-seed scores, routing summaries, component probes, and SHA-256 hashes
of the original local source artifacts.

## What was actually tested

The coefficient sweep used one SFT-initialized Olmo MoE checkpoint: 19 router
layers, 512 experts per layer, top-16 selection, and 18.514 billion total parameter
coordinates. Training was **41 synthetic filter/sort/count/sum prompts**, not
GSM8K or the later mixed-workload campaign. Each run trained for 60 updates with
32 responses per update, an EP2 trainer and one serving GPU, 4,096 generated-token
cap, 6,144-token packs, LR 1e-6, Adam betas .9/.95 and epsilon 1e-8. Core used
pack-local, token-weighted auxiliary losses, replay dispatch counts, and
mixed-policy refresh with lag 2.

Five coefficient settings used training seeds 17, 18 and 19. Evaluation used the
same image (`01M2SERAKCCJA2A4TCR0X505N9`), 1,024 held-out task prompts, four samples
per prompt, temperature 1, and engine seed 17. The primary comparison was declared
as F minus A at the 4K cap. The 8K evaluation was a separate generation pass.
This holdout had already been inspected: it is exploratory evidence, not a new
confirmatory population.

All 15 endpoints completed. The saved-answer audit verified 122,880 responses
across both caps against the official score records. Failed partial attempts
were excluded; export and storage incidents required documented retries.
[Final answer audit](https://beaker.org/ex/01M2XHH5N8BRT0PEW4Z7TX82CW).

## 1. Aux-off improved this task across all three seeds

| Arm | Balancing coefficient | z coefficient | Mean sampled accuracy, 4K |
|---|---:|---:|---:|
| A | .01 | 1e-5 | 70.70% |
| C | .003 | 1e-5 | 73.97% |
| B | .001 | 1e-5 | 74.11% |
| D | 0 | 1e-5 | 74.89% |
| F | 0 | 0 | 74.53% |

F minus A was **+3.08, +3.10 and +5.30 percentage points**, averaging **+3.82 pp**.
The paired training-seed t interval was **+0.66 to +6.99 pp**, using two degrees
of freedom and an approximately normal seed-difference assumption. Three seeds
cannot establish that assumption reliably. The secondary 8K contrast was +3.47 pp.
Thousands of sampled responses do not substitute for independent training runs.

The supported conclusion is an aux-off benefit in this model/task/horizon.
Intermediate coefficients did not identify an optimal positive coefficient or
a reliable accuracy dose curve. D minus F, which isolates adding z=1e-5 when
balancing is absent, was +0.37 pp at 4K with an interval of −4.26 to +4.99 pp:
**inconclusive, not proof that z-loss has no effect**.

## 2. Balancing has a clear routing dose response; aux-off does not freeze routes

These are identical-token, teacher-forced HF forwards on 32 fixed task responses,
not comparisons between newly generated texts. Each layer's expert counts are
pooled over the panel before computing its load coefficient of variation (CV:
standard deviation divided by mean); the table averages layers and seeds equally.
Assignment turnover counts selected expert IDs lost from the original top-16 set,
divided by 16. It is not the fraction of tokens with any changed expert.

| Arm | Initial task load CV | Update-60 task load CV | Selected assignments changed |
|---|---:|---:|---:|
| A: .01 balance | .4568 | .3956 | 9.26% |
| C: .003 balance | .4568 | .4328 | 4.88% |
| B: .001 balance | .4568 | .4491 | 3.79% |
| D: balance off, z retained | .4568 | .4574 | 3.50% |
| F: both off | .4568 | .4572 | 3.50% |

Strong balancing reduced task load CV by about 13%; aux-off barely changed it.
The task panel had one unobserved expert/layer pair before training and still one
at the endpoint in every arm/seed. The general-text panel observed every expert
in every layer. This is no observed collapse on these panels over 60 updates;
an expert unobserved on a finite panel is not necessarily globally inactive.

General text was 256 fixed public English WildChat conversations, first
user/assistant exchange, capped at 2,048 tokens per document. Its pooled load CV
stayed near .336, yet strong balancing changed **4.56%** of assignments versus
**2.30%** with aux-off. Stable pooled loads can conceal substantial reassignment.
Neither these forwards nor small likelihood changes establish general capability
harm or loss of useful pretraining specialization.

A separate gate-share probe recomputed top-16 choices with zero ID mismatches.
On task-response tokens, the starting top expert carried **14.53%** of selected
gate mass and rank 16 carried **4.31%**. In seed 17, experts evicted at the strong-
balancing endpoint carried **7.40%** of the original selected mass per token/layer,
versus **2.58%** for aux-off. Most evictions were low-ranked, but high-ranked
evictions were much more frequent under strong balancing. Gate mass measures
router weighting, not actual expert-output importance or causal task damage.
[Gate-share probe](https://beaker.org/ex/01M2XSCV43MNQVQF2DPK67WD0D).

The fixed-hidden-state probe also limits a router-matrix-only explanation. For
Core seed 17, substituting trained router weights while holding their inputs fixed
changed about **1.52%** of assignments, versus **9.48%** in the full endpoint
forward. Upstream representation changes matter substantially. This counterfactual
is not an additive causal decomposition of the two effects.

## 3. Padding was a demonstrated live-objective bug in Megatron

The historical hybrid recomputation path and Olmo MoE wrapper dropped the padding
mask before the router. Masking policy loss did not mask the independently
attached auxiliary losses. Padding therefore affected both the auxiliary token
population and the histogram controlling balancing pressure on real tokens.

A same-checkpoint, same-response, same-replay endpoint probe compared the native
padded path with and without a correct router mask. It contained 46,874 real and
36,150 padded positions (43.5% padding), with no optimizer update:

| Router gradient | Padding included | Correctly masked |
|---|---:|---:|
| Policy norm | .010565 | .010565 |
| Balancing norm | .052853 | .012570 |
| z-loss norm | .00020446 | .00004193 |

The balancing norm changed by **4.20×**, and its padded-versus-masked cosine was
**.0163**: this was chiefly a different direction, not just a coefficient scale.
The saved raw first moment aligned with the live padded balancing gradient at
cosine **.9382**. The masked gradient matched the unpadded reference at .999972.
This explains why the earlier unpadded probes matched across backends yet failed
to explain live training.

The propagation correction passed
[11 focused GPU tests](https://beaker.org/ex/01M2T3SF4V3W4MKF1M5RNKE6BM), then
qualification and a [matched 60-update run](https://beaker.org/ex/01M2T6NWD9WG5C2ABNHRNQPPWS).
On the frozen task panel, seed-17 endpoint CV was:

| Backend | Load CV |
|---|---:|
| Core | .39502 |
| Original Megatron | .44790 |
| Mask-corrected Megatron | .39716 |

The correction closed **95.96% of that matched-seed CV gap**, with intermediate
milestones also tracking Core. The bug had **suppressed real-token flattening**;
correcting it increased flattening. It did not demonstrate a quality advantage:
corrected-minus-original common sampled accuracy was +1.05 pp at 4K and −1.12 pp
at 8K. The −6.05 pp native-greedy result was decoder-specific and from one seed.

This establishes the historical mechanism, not the patch status of every current
Megatron image. Runtime fixes still require pinned-source and execution checks.

## 4. Fresh counts did not explain the flattening gap

Three Core runs replacing replay-derived balancing counts with current-router
top-k counts ended at mean CV **.396075**, versus **.395627** for ordinary Core.
The single-response, packing-disabled Core arm also followed Core's routing curve
(one seed; it changes both physical execution and local loss grouping).

On a captured update-40 batch, fresh/replay balancing-gradient cosine was
**.999916**, with a 1.31% relative vector difference. Top-16 set disagreement was
about 18.8%, but pooled histogram total variation was about .19%, and weighted
per-pack histogram variation about .52%. Many token-level disagreements can
cancel in the aggregate counts that determine balancing pressure.

Fresh-count sampled accuracy effects remained uncertain. These results reject
count source as the main explanation of this measured routing gap; they do not
establish universal equivalence. The earlier assertion that replay balancing has
“no fixed point” was too strong. Route-table age also need not equal the age of
the policy that originally generated the response.

## 5. Export and optimizer checks corrected several tempting explanations

- **Export conversion was not the cause at the checked endpoints.** Canonical
  native-BF16/HF comparisons found zero unequal elements across all
  18,514,193,152 coordinates in each historical Megatron aux-on/off endpoint.
  Separate master-cast checks and Core audits corroborated storage parity.
  Tensor parity does not establish identical forward kernels.
- **Aux-off backends had similar displacement norms, not identical updates.**
  Their native FP32 router displacement cosine was about .086. Similar norms
  cannot prove identical stochastic trajectories or interchangeable trainers.
- **Z cancellation in Core was not supported.** Raw common-across-expert saved
  first moments aligned with the endpoint common z gradients in both Core and
  live-padded Megatron (cosines .99775 and .99926). Adam's coordinatewise
  normalization can create or rotate a common update component; that component
  is not a direct measure of z-loss's cumulative contribution.
- **BF16 rounding mattered, but did not explain away the routing gap.** Replacing
  endpoint router working weights with FP32 masters changed routes without
  closing the gap. That inference counterfactual is not a test of training with
  FP32 router storage. Adam also has no universal one-learning-rate bound on
  each coordinate's step.
- **Norm agreement at initialization was insufficient.** Matching unpadded
  auxiliary objectives produced close native gradients, but missed live padding.
  A useful equivalence test must reproduce token masks, packing, recomputation,
  backward scaling, and parameter layouts actually used during training.

## What to carry forward

Aux-off is an evidence-backed choice for this short toy workload, with routing
and load monitoring still warranted at larger scale. A balancing coefficient is
an optimization choice, not a universally harmless systems setting. Measure
assignment turnover as well as load CV, and compare learning with matched
evaluation and multiple training seeds.

The study **did not establish** that flattening mediates the accuracy loss, that
z-loss is inert, that useful specialization was destroyed, that aux-off is optimal
for pretraining or broad RL, or that expert-aware scheduling improves wall-clock
performance. Those claims need their own interventions and measurements.

## Retired branches and reproducibility

Production router controls were consolidated separately in `3e961ab1a`; see
[router auxiliary objectives](../core.md#router-auxiliary-objectives). The four
research branches below are retired after preserving their full Git histories:

| Branch | Archived tip | Material |
|---|---|---|
| `codex/core-router-followups-20260917` | `a09410e7dad1822f19cf0f2327a6793ee3489dae` | Gradient, optimizer, export and padding forensics |
| `codex/moe-common-image-20260917` | `abb46f430007886efed0cb5dda8ba01681cbee9b` | Shared-runtime seed campaign and evaluator |
| `codex/router-fresh-counts-20260918` | `d17d3e6aa1dd07bd0e064d3727ac6983f240a195` | Current-count and single-response interventions |
| `codex/core-balancing-sweep-20260918` | `62edce327a838c82e4411ff9765744d64b7239f1` | Coefficient sweep, analysis and reports |

The local archive is `runs/router-research-archive-20260921/`: its README records
bundle verification, artifact backups, and restoration commands. It is ignored
by Git, so it is not an off-machine backup. Existing `runs/` results and remote
experiment artifacts are retained. The committed evidence snapshot above keeps
the principal numerical results available without those worktrees. Original
reports remain in `runs/core-balancing-sweep-20260918/`,
`runs/router-forensics-20260917/`, `runs/router-fresh-counts-20260918/`, and
`runs/gate-share-probe-20260919/`; dated pending statements in them may be stale.
