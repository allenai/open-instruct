# Design follow-ups (not yet implemented)

Captured from discussion after the first Qwen length-shaping run revealed the
reward-hacking failure mode. None of these is implemented yet; we'll revisit
once intermediate-step evals land and we have the per-step trajectory.

## Conceptual issue we identified

In the current implementation, length shaping rewrites `r` to `r_shaped` and
then GRPO computes advantage as `(r_shaped − μ(r_shaped)) / σ(r_shaped)`. Both
the numerator and the denominator now mix correctness and conciseness, and the
per-prompt baseline μ no longer cleanly measures problem difficulty. A
longer-correct response and an incorrect response end up with identical
advantage (both have numerator `0 − μ_shaped`). That's not a numerical bug — it
matches what the shaping math prescribes — but it conflates two signals the
advantage was designed to keep separate.

## Options under consideration

### A. Unshaped baseline, shaped numerator
```
advantage_i = (r_shaped_i − μ(r_raw)) / σ(r_raw)
```
Baseline statistics come from the raw verifier. Cheap change; keeps difficulty
normalization clean. Doesn't fully solve the longer-correct == incorrect
collision because the numerator is still 0 for shaped-to-zero longer-corrects.

### B. Raw advantage + additive length penalty (recommended)
```
advantage_correctness_i = (r_raw_i − μ(r_raw)) / σ(r_raw)         # standard GRPO
length_term_i           = correct_i × −(L_i − L_min) / NORM        # 0 for incorrect
advantage_i             = advantage_correctness_i + λ_shape × length_term_i
```
Length is an additive *advantage* adjustment, not a reward rewrite. Longer-
correct keeps the positive correctness advantage and pays a length tax, but
stays above incorrect in advantage space. Tunable λ_shape decouples
"correctness pressure" from "length pressure."

### C. Subtractive length penalty on reward
```
r_shaped_i = r_raw_i − correct_i × λ × (L_i − L_min) / NORM
```
Same intent as B but folded back into reward, so standard GRPO applies. Keeps
r_shaped strictly between 0 and r_raw rather than zeroing out — longer-correct
retains *some* positive correctness signal in advantage space.

### D. GFPO-style: filter, don't shape
Generate 2k samples per prompt, drop the longest-by-tokens half of the correct
set, then run standard GRPO on what remains. Costs 2× generation but doesn't
touch the reward function at all. The proposal references this as a comparison
baseline.

### E. Bigger denominator in the linear decay
Currently the relative excess is `(L − L_min) / L_min`, which makes the
"2× as long → zero" point dependent on `L_min`. If `L_min` is tiny (one-token
answer), `α = 1.0` zeroes out anything ≥ 2 tokens. A bigger, less group-
dependent denominator — e.g. `L_min + epsilon`, `max_response_length`, or a
running median — would make the decay much gentler and avoid runaway collapse
when `L_min` happens to be very small in a group. Easy to add as a CLI knob.

## Reporting bug to fix alongside

Our integration reassigns `scores = scores_per_prompt.reshape(-1)` *after*
shaping, and the downstream metrics block uses that post-shaping `scores`.
So `unsolved_batch_size_ratio`, `val/sequence_lengths_solved`, and
`val/sequence_lengths_unsolved` are computed against shaped scores, not raw.
A longer-correct response that got zeroed by shaping appears as "unsolved"
in those metrics, which makes it hard to tell whether the *true* solve rate is
collapsing.

**Fix**: compute the solved/unsolved metrics from the pre-shaping
`scores_per_prompt` snapshot (which we already capture for
`val/scores_pre_shaping`). Keep the post-shaping versions too if useful, but
under a clearly different name (e.g. `val/sequence_lengths_solved_after_shape`).

## Order to try, when we revisit

1. Ship the reporting bug fix (cheap; makes the next run interpretable).
2. Add **E** (bigger denominator) as a CLI knob — least invasive change to test
   "is α just way too aggressive because the denominator is bad?"
3. Add **B** as a `--length_reward_combine=additive_advantage` mode and re-run
   linear α=1.0 to compare against the current multiplicative-reward result.
4. Consider **D** if **B** still shows collapse — sidesteps the
   reward-shape question entirely.

We can keep the current `multiplicative_reward` mode available so existing runs
remain reproducible and we can do apples-to-apples comparisons across modes.
