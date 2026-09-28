# FP32 output head: matched update-300 comparison

September 28, 2026. The [update-300 evaluation](https://beaker.org/ex/01M3MAVQ03VB614CNSEC2PYDQ7)
completed successfully (exit 0); corrected scoring and independent local rescoring
are complete. Training remains stopped. No additional training or evaluation was
launched for this analysis.

Runs: [control](https://beaker.org/ex/01M3FX32BWWTQMGJQWKDYD04GF),
[FP32 treatment](https://beaker.org/ex/01M3G3RPVQAYNP9JFCXR4M23J2),
[update-300 evaluation](https://beaker.org/ex/01M3MAVQ03VB614CNSEC2PYDQ7).
The submitted treatment changed only `trainer.fp32_lm_head=true` and run/output
identities; see [configuration differences](config-diff.json). The workload used
64 groups × eight samples, a 16K response cap, EP4 training, 20 TP1 rollout engines
and six optimizer updates of allowed policy lag. Evaluation used temperature 0.8,
top-p 0.95 and the same 16K cap. Exact source/image identities, scoring hashes,
evaluation settings and completion statuses are in [provenance](provenance.json).

This is the longer follow-up to the
[32-update FP32-head screen](../trainable-fp32-head-math-20260926.md).

## Finding and recommendation

Both arms continued learning at the same measured pass@1 pace from 200 to 300.
FP32 remains slightly behind, with no statistically clear pass@1 difference.
The practical recommendation is to keep BF16 as the default and retain the FP32
head as an option. This run demonstrates no quality benefit sufficient to justify
changing the default. It also does not establish a real learning penalty from
FP32, or that BF16 numerical noise is beneficial. Stopping here was reasonable.

## Matched result

512 identical questions, four responses each, 2,048 responses per arm. The corrected
primary score requires a correct, completed final answer. Same panel, prompts,
labels, response seeds, generation settings, immutable evaluator source and serving
flags were verified. Both evaluations use the common default serving precision;
the treatment enables FP32 during training and rollout, not in this final evaluator.

| Metric | Control | FP32 head | FP32 minus control; paired 95% interval |
|---|---:|---:|---:|
| Completed-correct pass@1 |34.86%|34.03%|−0.83pp [−2.39,+0.73]|
| Pass@4 |51.56%|49.80%|−1.76pp [−5.08,+1.56]|
| Raw answer correctness |35.30%|34.52%|−0.78pp|
| Token-cap rate |22.17%|24.17%|+2.00pp [0.00,+4.00]|
| Mean response tokens |9,916|10,008|+92 [−62,+248]|

The cap-rate gap is suggestive but borderline; its interval touches zero. The
length gap is smaller than at 200 and its interval includes zero. These are soft
behavioral differences, not evidence of an established broad quality penalty.

| Update | Control pass@1 | FP32 pass@1 | FP32 minus control |
|---|---:|---:|---:|
|100|30.13%|28.13%|−2.00pp|
|200|32.67%|31.84%|−0.83pp|
|300|34.86%|34.03%|−0.83pp|

From 200 to 300, both arms gained exactly 2.20pp (45 additional correct completed
responses). The gap did not widen. From each arm's independent startup evaluation,
control gained 7.86pp and FP32 gained 7.52pp. The difference in gains is −0.34pp,
95% interval [−2.64,+1.90]. There is no evidence here of materially different
learning progress. Consecutive checkpoints are correlated observations of the
same two runs, not three independent trials.

## Completion and source breakdown

Control has 714 correct responses among 1,594 completed responses (44.79%);
FP32 has 697 among 1,553 (44.88%). FP32 has 41 fewer completed responses and 17 fewer
correct responses. This is consistent with a completion-related difference,
but the completed subsets differ; it is not a causal decomposition or proof
that underlying reasoning is equal.

At the matched response-slot level, 547 responses are correct in both arms,
167 only in control, and 150 only in FP32. Of those one-sided successes, 40 control
wins face an FP32 cap and 33 FP32 wins face a control cap. These are descriptive
pairings; independently generated continuations are not counterfactual traces.

| Source | Questions | Control pass@1 | FP32 pass@1 |
|---|---:|---:|---:|
|Dolci math|389|17.42%|16.65%|
|GSM8K cleaned|123|90.04%|89.02%|

Both sources have a small negative point estimate for FP32. The mixed-panel result
does not conceal a large positive reversal on either source.

## What the precision result means

Earlier fixed-weight diagnostics found better training/inference numerical
agreement with the FP32 head. That is a different outcome from learning quality.
Changing head arithmetic also changes sampled training responses, rewards, group
advantages, probability ratios and subsequent updates. Less numerical disagreement
therefore has no guaranteed monotonic effect on this finite-run development score.

This result is best described as **no demonstrated quality dividend**, with a
small practical preference for BF16 in this configuration. It is not evidence that
precision work is generally unhelpful, that the GRPO implementation is wrong, or
that adding numerical noise would help. A cheap future causal diagnostic would
hold weights and token sequences fixed and compare ratios, clipping and gradients
with the option on/off. No such follow-up was launched here, and another long
training continuation is not justified by these results alone.

## Recovery provenance and limits

The original update 300 HF write stalled around 07:56 UTC and remained unmarked.
After preemption, Beaker resumed from native 275 and replayed updates 276–300.
Saved driver timing shows rollout 275 training at 10:25 UTC, rollout 299 at 11:51 UTC,
and native 300 committed at 11:53:48 UTC. The earlier description that Beaker resumed
from 300 was incorrect and has been corrected in the state notes.

The evaluated export was reconstructed from this committed native 300 checkpoint,
not certified from the interrupted earlier HF file. Native clock, all four scheduler
epochs, dataset cursor checksum and shard bounds passed verification. All 23,441
serialized tensors exactly matched the reconstructed values; repeated artifact
checks were stable. The original and recovered exports differ in 433,862,909 of
12,496,190,080 values (~3.47%), maximum absolute difference 0.000244140625. They
belong to different pre/post-replay trajectories, so the mismatch is not evidence
of a conversion rounding bug or file corruption. This recovered export is the
checkpoint that the completed comparison actually measures.

There is only one training realization per arm, with preemption/resume effects.
The confidence intervals resample 512 questions, retaining all four responses and
all checkpoints/arms together: 20,000 resamples, seed 20260927. They characterize
question/evaluation uncertainty, not training-seed variability. The metrics are
not multiplicity-adjusted. Small gaps should not be treated as causal proof.

The requested stop-at-300 guard failed to handle the incomplete original export.
Training continued after replay to at least update 362 before manual shutdown.
All three treatment replicas exited at 15:43 UTC; the evaluation still measures
native update 300. The treatment must not be resumed without a new request.

## Verification and artifacts

- [Comparison data](comparison.json): all means, paired intervals, source and completion breakdowns, and the raw collection SHA256.
- [Run provenance](provenance.json): source/image pins, task settings, original and corrected evaluation manifests, shared-storage locations and terminal statuses.
- [Configuration differences](config-diff.json): treatment versus control at submission.
- [Replay evidence](recovery-replay-evidence.json) and [export comparison](recovered-difference.json): saved timing and the CPU-only audit of the two update-300 attempts.

Both arms retain their immutable generations and panels on WEKA. Under
`/weka/oe-training-default/robertb/open-instruct/runs/`, the control root is
`hero-math-learning-20260926` and treatment root is
`hero-math-learning-fp32-head-20260926`. Relative to each root:

- Original evaluation: `evaluation/results/update-00000300-47ae9fc9104a8071/`.
- Corrected rows, metrics and hashes: `evaluation/corrections/development-gold-v2/update-00000300-47ae9fc9104a8071/`.
- Corrected W&B series: `eval/development_gold_v2/*`, published using each evaluation receipt's run identity.

The local analysis bundle is `runs/hero-fp32-head-20260926/`: `analyze_update300.py`,
`analysis-update300.log`, and `raw-matched-development-300.json.gz`. That folder is
Git-ignored; this archive contains the result and provenance independently of it.
The raw collection hash in `comparison.json` identifies the exact local input.

Independent rescore used olmo-eval revision
`941ef57b41121ae2887e10bc29a620e83928e689` and the corrected development scorer.
All 4,096 scores/completion flags agree with the automatic corrected scorer;
generation hashes, question/response identities and both correction-scorer hashes
match. The analysis also reproduces all earlier headline means. Seventeen focused
scorer/reconciliation tests passed; Ruff passed on the fetch, wait, analysis and
read-only export-audit scripts. Archive documentation validation is recorded in
the commit that adds this report.
