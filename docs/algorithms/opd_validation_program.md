# OPD validation program: infra, methodology, and paper replication

Living tracker for the on-policy distillation (OPD) validation work that follows the
Qwen3.5 math sync reruns ([qwen35_math_opd_sync_rerun.md](qwen35_math_opd_sync_rerun.md)).
It records what we decided, why, what each step is, and where it stands. Updated as work
lands; the **Log** at the bottom is append-only and dated (UTC).

Owner: Kevin Farhat. Branches: Open Instruct work on `codex/qwen35-math-opd`; Miles work
on `robertb/miles-qwen35-opd` (worktree `open-instruct-miles`).

## Goal

Establish that (1) our OPD infrastructure is correct, (2) our methodology can reproduce a
published OPD result on math, and (3) decide how Open Instruct and Miles divide the OPD work
going forward. Then use that validated base to test one method improvement (EOPD).

## Decisions so far

| # | Decision | Why |
|---|----------|-----|
| D1 | Replicate the **plain-OPD baseline** of the EOPD paper ([arXiv 2603.07079](https://arxiv.org/abs/2603.07079)) first, then its EOPD result. | Their OPD baseline is our exact algorithm (sampled-token reverse-KL advantage under PPO clipping). Everything is public: Qwen3-Base students, Qwen3-8B teacher, MATH / DAPO-Math-14k, all hyperparameters in Appendix A, a verl fork on GitHub, 4xA100 compute. MATH500 Avg@8 over 4000 samples is a low-variance matching target. The Thinking Machines recipe (Qwen3-8B-Base from a 32B teacher, 70 on AIME24) needs a 400k-example SFT stage and a 32B teacher, so it is a later, larger validation. |
| D2 | **Do not migrate wholesale to Miles.** Miles is the research vehicle for OPD (replication + EOPD); Open Instruct stays for production training and gets its OPD pipeline hardened. Results that matter run in both until they disagree. | Two independent implementations agreed on 18/18 checkpoints, so the Open Instruct math is right; its flaws are pipeline-level and fixable. Miles already has a separate SGLang teacher with top-k scoring, minibatching (`global_batch_size`), and a runtime contract/audit, which EOPD needs. Miles' costs: only a Qwen3.5-2B model profile in the wrapper, pure-OPD wrapper pins top-k to 0, one person's branch on a young framework, GPU-efficiency not clearly better (136 vs 83 GPU-min per 4B step, with under-provisioned inference). |
| D3 | Evaluation for the replication and EOPD must be **sampled** (T=1.0, **top-p 0.8**, 8192 tokens, 8 samples, Avg@8 and Pass@8), matching the paper's Qwen2.5-Math harness run (Sec. 5.1 and App. C). Corrected 2026-09-18: Table 9's top-p 1.0 (Qwen) is the *training* rollout setting; evaluation uses 0.8 for every model. Arm configs fixed in Miles commit (eval_top_p 0.8). | Our greedy pass@1 eval cannot see the diversity effect the paper claims. |
| D4 | Gate EOPD on a **teacher-entropy diagnostic** before writing training code. | If our teachers rarely exceed the entropy threshold, EOPD reduces to OPD and there is nothing to test. |
| D5 | Miles replication and EOPD work lands **directly on `robertb/miles-qwen35-opd`** (Kevin, 2026-09-18: Robert cleared the branch for us, treat it as ours). | No separate branch or review gate for the wrapper changes in step 3 and 6. |

## Known flaws in the Open Instruct OPD path (audit 2026-09-17)

From reading `grpo_fast.py` / `grpo_utils.py` on `codex/qwen35-math-opd`:

1. **Async dataflow** (confirmed, caused the canonical 2B collapse): with `async_steps>1` and
   in-flight updates the trainer consumes the first-finished responses, so batches are
   length-sorted and the long tail is dropped as stale (16-18k of 25.6k episodes). Pure-OPD
   advantages are per-token and not group-centered, so batch composition steers the update.
   Sync mode is a 2-4x-cost workaround, not a fix.
2. **Mixed numerics in the advantage**: student side is the bf16 vLLM rollout logprob, teacher
   side is the fp32 trainer logprob. Late-run signal (2B mean reverse KL 0.0005) is far below
   typical engine mismatch (order 0.01/token). `debug/vllm_local_reverse_kl` measures the gap.
   Unbiased in expectation, but variance and any systematic bf16 bias are unmeasured.
3. **Silent NaN sentinel**: `mask_logprobs` maps NaN vLLM logprobs to `INVALID_LOGPROB=1.0`;
   inside the response mask this becomes a large positive advantage with no error or metric.
4. **Teacher tempered by the sampling temperature** in `compute_logprobs_tiled`. No-op at T=1.0;
   undocumented coupling otherwise.
5. **No end-to-end alignment test**: unit tests cover the fold arithmetic on synthetic tensors;
   nothing checks the `[:, 1:]` alignment of vLLM logprobs, teacher logprobs and response mask
   on real data. Miles has this in its runtime contract.
6. **Unbounded per-token advantage** (`opd_adv_clip` off, DPPO has no ratio clip). Matches the
   Thinking Machines recipe; shared with Miles; unguarded.

## Program

| Step | What | Why | Status |
|-----:|------|-----|--------|
| 1 | Close out the Qwen3.5 4B Miles v5 replication: eval hf-89, hf-99; manual `opd_audit` on the 4B root; final docs pass. | Completes the 2-framework cross-check (2B done 10/10, 4B 8/8 so far). | **Complete 2026-09-18.** 4B job finished 100 rollouts (exit 1 in the image's stale audit, like the 2B); manual `opd_audit` `01M2S39293NTSRJQGYWB2Y183R` passed (100 updates, max advantage error 0.0, `audit.json` written). hf-89 greedy `01M2RYZ320TBZWY8C89W92D84E` = MATH-500 90.0 / DAPO 78.0 / AIME 46.7 / BRUMO 63.3 vs sync step 90 90.0 / 77.0 / 43.3 / 56.7; hf-99 greedy `01M2S3ETXCA0KKM6KY1WD1KN6R` = 90.0 / 76.0 / 46.7 / 60.0 vs sync step 100 90.0 / 76.0 / 53.3 / 60.0 (`01M2NMYJWQYXY043YJGXEG7GWF`). Both frameworks agree at every matched step (2B 10/10, 4B 10/10 exports in band); rows in `qwen35_math_opd_sync_rerun.md` and Kevin's results doc. |
| 2a | Measure the vLLM-vs-trainer logprob gap against the OPD signal on the sync runs (`debug/vllm_local_reverse_kl` vs `objective/opd_reverse_kl`, W&B `rg1a2gel`, `zz3q6skv`, `8nlm6azd`). | Decides whether flaw 2 matters. | **Done 2026-09-17** (Beaker `01M2RXC2NK173XAYCHEZRYDARY`). Sync: gap 1e-4 to 2e-4 at every step; 4B signal 0.07 (gap negligible), 2B signal decays 0.033 -> 0.0005 so the gap is ~30% of the signal only in the last 30 steps. DPPO mask keeps 99.997% of tokens, ratio 1.0000. Flaw 2 downgraded to minor. Async: gap 0.01-0.036 for 2B (100-300x sync), mask keeps only 92-96% of tokens; 4B async gap ~1e-3, mask 99.0-99.9%. Quantifies flaw 1. |
| 2b | Open Instruct hardening: real-dump OPD audit (advantage identity + alignment), NaN-inside-mask becomes an error, document the temperature coupling. | Flaws 3, 5, 4. | **Done.** Real-dump audit `01M2S0BG4NP4878N18VASKNY6P` over the Qwen3-0.6B ← Qwen3-1.7B gsm8k smoke (`01M2RZ8YYFT927SM1EDC7646ST`, 16 steps, `--save_traces`): steps 1/2/8/16, 127 records, 65k response tokens, `advantage == kl_coef·(teacher − rollout)` with max error 0.0, no non-finite logprobs inside the mask, teacher signal up to 32 nats per token (no adv clip in that smoke). The `[:, 1:]` alignment of rollout logprobs, teacher logprobs, mask and advantages holds on real data. Remaining under 2: decision 2d (async fence). |
| 2c | One 2B sync arm with `--use_vllm_logprobs false` (trainer-side student logprobs). | Direct test of flaw 2. Needs 4 nodes for ~4h. | **Proposed skip** after 2a: the gap is 1e-4 against a 0.07 signal at 4B and only matters for the tail of the 2B run. Kevin to confirm. |
| 2d | Fix the async path for OPD: batch by prompt set, not by completion order (design in the 2026-09-18 04:50Z Log entry). | Flaw 1. Kevin (2026-09-18): the fence is not the answer, `async_steps=4` should work without the length bias. | **Implemented 2026-09-18** as `--fixed_prompt_batches` (commit `25e39f4db`: `data_loader.py`, `data_types.py`, `vllm_utils.py`, `grpo_fast.py` warning, docs, changelog, launcher `qwen35_math_opd_fixed_batches_rerun.sh`). Unit tests on Beaker `01M2SE7GZZM6TGZD07T5KDATA4` (5 passed at `b6f315bba`). **GPU verification approved by Kevin 2026-09-18 07:14Z ("go").** Patch dataset `01M2SNQYARKRCZPZ6AN8CEJPGX` (`qwen35-math-code-patch-ee0d4845e`); one-node smoke `01M2SNTT1XKJZW2B7B2QZK1VCX` **passed 07:26Z** (4 steps at async_steps 2; `results_parked_for_later_batches` 27/0/16/0, `stale_results_dropped` 0 every step, `opd_reverse_kl` 0.04). Full 4-node run: first launch `01M2SPKXVMR0QT3FCZEYE5AKXD` (07:28Z) died at `wandb.init` after 40 s because the 71-character exp name exceeded W&B's 64-character tag cap; name shortened (commit `1e51d4567`) and relaunched as `01M2SPXTBP04X0HD7FPDTKN3WZ` at 07:33Z (canonical 2B async-4 recipe + `--fixed_prompt_batches true --save_freq 10`, run name `qwen35_2b_opd_fixed_batches_async4_lr1e6_100step_4node_20260918_003302`, W&B project `opd`). **Finished 09:50Z, 100 steps, exit 0, and it collapsed like the canonical async run**: eval score (DAPO holdout + AIME25 + BRUMO, pass@1) 0.37 → 0.04 → 0.15 → 0.34 → 0.09 → 0.03 at steps 0/20/40/60/80/100, stop rate 0.67 → 0.02 by step 30 and 0.00 at step 100 (all responses at the 16384 cap); canonical async `01M1WQ2DRFJF2C1019HMZ02318` went 0.37 → 0.06 → 0.47 → 0.47 → 0.53 → 0.01; the sync rerun held DAPO 47.7 at step 100. Composition was fixed as designed (`results_parked_for_later_batches` 375 at step 1 then 0-8, `stale_results_dropped` 0 throughout), so batch composition alone does not explain the async collapse on this setup. The run also exposed a pipeline-depth bug: rollouts were 9 weight versions stale (canonical 4-5) because the refill was keyed off the data actor, which runs `async_steps` ahead of the trainer; fixed in `efc449cf7` (refill on trainer consumption; staleness bound = `async_steps`+1). A clean isolation needs a rerun with the fix (~2h15m on 4 nodes); awaiting Kevin. |
| 3 | Miles replication prep: Qwen3-1.7B-Base, Qwen3-4B-Base, Qwen3-8B model profiles; expose minibatching (4 optimizer steps per rollout) and 1 sample/prompt in the OPD TOML; cosine LR; sampled Avg@8/Pass@8 eval matching the Qwen2.5-Math harness. | Paper setting: B=128, mini 32, LR 3e-6 cosine, 4096 response, T=1.0, 3 epochs MATH / 2 epochs DAPO-Math-14k. | **Mostly done 2026-09-18** on `robertb/miles-qwen35-opd`: Qwen3 model map (upstream `qwen3-1.7B/4B/8B` profiles), `training.optimizer_steps_per_rollout`, `optimizer.lr_decay_style/lr_warmup_iters/min_lr`, `inference.top_p/eval_top_p/eval_max_response_length`, `scripts/miles/prepare_eopd_math_prompts.py`, specs `configs/miles/opd/eopd-opd-qwen3-{4b-base-dapo14k,1.7b-base-math}.toml`. Data rendered on Weka (`miles-opd/data/eopd-math-v1`, Beaker `01M2RYRHTNKRMPKNVJXRT7SHTX`): math_train 7496 (4 MATH rows lack a boxed answer), dapo_math_14k 14109 (7 overlap eval sets), math_500 500, aime24 30, aime25 30, amc23 40, minerva 272, olympiadbench 674; non-thinking Qwen3 template, paper suffix. First prepare job `01M2RZHTYE1Y6H0DB9JR0NKX2W` was rejected: Qwen3-Base stops on `<|endoftext|>`, Qwen3-8B on `<|im_end|>` (vocab identical). Added `model.align_eos_with_teacher` (Miles commit d2256c371): the learner adopts the teacher's eos, keeps its own as a second stop id. Without it a base learner would run on after `<|im_end|>` until `<|endoftext|>` or the length cap, which is the likeliest way to silently miss the paper's numbers. Prepare succeeded on attempt 4 (`01M2S0HMZQ7W7VG9W06K4V5J0B`, assets `Qwen3-1.7B-Base-…-eos`, `Qwen3-8B-…`); tiny 4-GPU smoke `01M2S0PHCCAKCY6B4WHWYN257J` was rejected (prepare had been run in the training root; the documented flow gives preparation its own `output.root`), relaunched as `01M2S132HYR6PX5Z1GNH2VPE3N` into root v4, which failed in the HF-to-Megatron conversion (Megatron padded the 151936 vocabulary to 152064 under TP2; mbridge scatters the HF embedding unpadded). Fixed with repository profiles `qwen3-1.7B` / `qwen3-4B` pinning `--padded-vocab-size 151936` (Miles commit abaff5d6a); image `01M2S1VTS5N846S0Z9155QXFNH` built and the smoke relaunched as `01M2S1W1XKQ93S4XXEXWCG0G5K` into root v5 (Miles commit 7b6ccfa39: 3 rollouts × 8 prompts, 4 optimizer steps per rollout, cosine LR, aime24 Avg@4 eval). That smoke converted the learner fine (padded-vocab fix confirmed) and ran the aime24 pre-eval and the first rollout, then died in Miles's `zero_std` rollout metric: with one sample per prompt every group is zero-variance and the metric rounds `sample.reward`, which our reward hook had set to the teacher's scoring payload (a dict). The hook now returns the numeric task reward 0.0 and parks the payload in `sample.metadata` until `post_process` (Miles commit f78a618a1); smoke #4 (root v6) then failed the in-image runtime tests: the new EOPD hook test compared float32-rounded top-k log-probs against exact literals (that test file cannot be collected on the Mac, so the GPU job is its first run). Test fixed (Miles c098412e0); smoke #5 (root v7) then ran the whole chain and exposed one more Qwen3.5 assumption in the audit's export check (single-file base, no `A_log`), fixed in Miles 50dab1a1d. **Smoke #6 `01M2S4MN9628M5C6900CRV3E4K` (root v8, image `01M2S4ME34M3JHMMGT5S23TMXM`) passed end to end on 2026-09-18 02:27Z**: runtime tests, teacher preflight, aime24 pre/post eval, 3 rollouts × 8 prompts, 12 optimizer steps (4 per rollout, cosine LR 2.95e-6 → decaying), audit (advantage identity, finite nonzero grads, 273 changed tensors, full-model scope), hf-2 export reload. The Qwen3 tiny path is ready for the arms. Pass@8: the paper's headline is Avg@8, which Miles in-run eval reports (8 samples, T 1.0, top-p 1.0); a Pass@8 post-hoc path (Open Instruct `--eval_pass_at_k 8` over the rendered sets converted to `messages`/`ground_truth` JSONL) is scoped but deferred until an arm needs it. |
| 4 | Run the OPD baseline: arm 2 (Qwen3-4B-Base from Qwen3-8B on DAPO-Math-14k), then arm 1 (Qwen3-1.7B-Base on MATH). Target: MATH500 Avg@8 within ~1 point of 78.8 / 67.8. | Infra validation against a public number. | Ready to launch 2026-09-18: the Qwen3 Miles path (tiny OPD smoke #6) and the EOPD path (tiny EOPD smoke) both pass end to end; remaining prep is the arm-2 TOML (Qwen3-4B-Base ← Qwen3-8B, DAPO-Math-14k prompts already at `miles-opd/data/eopd-math-v1/`, paper LR/rollout schedule) and the Avg@8 eval config. **Approved by Kevin 2026-09-18 (option 2: arm 2 OPD then arm 2 EOPD, one seed each).** Arm 2 OPD attempt 1 `01M2SD1X7WTE4KCKAA4XR7YZQ8` (04:40Z, image `01M2S4ME34M3JHMMGT5S23TMXM`, root `-v1`) **failed 05:02Z in the step-0 MATH500 eval: one math-verifier check exceeded 45 s and the timeout was fatal.** Fixed in Miles `0058e0028` (timeout scores 0); attempt 2 `01M2SEQX9NMEZ6K13RJ586FZJ0` launched 06:05Z (image `01M2SEQPBZRNP1HBWC72N71JCQ`, Miles `0058e0028`, root `-v2`). **Attempt 2 diagnosis (17:15Z-17:48Z): eos mismatch.** With `align_eos_with_teacher` the Qwen3-Base learner keeps `<|endoftext|>` (151643) as a stop id and ends most answers with it, while the chat teacher only ends answers with `<|im_end|>` (151645); the teacher's log-prob of the terminal `<|endoftext|>` averages -21.4 nats (`teacher-scores.jsonl`, 1500 records), so pure OPD punishes stopping. Train truncation went 0.11 (rollout 1) → 1.00 by rollout 20 and stayed there; MATH500 Avg@8 0.244 (base) → 0.744 (rollout 44) → 0.764 (rollout 88) with every eval response at the 8192 cap (the boxed answer appears early, then the model rambles). Fix in Miles `f92685054`: the launcher exports `OI_OPD_TEACHER_EOS_REMAP=151643:151645` from the prepared learner's `generation_config.json` and the rollout hook scores a *terminal* learner stop id as the teacher's eos (sample keeps its real tokens; `eos_remapped` recorded). Hook tests on the Miles image `01M2TT0R0TAR5PGXJ1PG42NYEP` (10 passed; the one failure `test_native_dataset_preserves_rendered_text_with_processor` predates this change, upstream `Dataset` in the image differs). Attempt 2 is left running to completion as the "eos-bug" reference (rollout 131/220 at 17:48Z, ~6.5 min/rollout, ETA ~2026-09-19 03:30Z). **EOPD arm on hold**: it would inherit the bug; attempt 3 (OPD with the fix, new image) and the EOPD arm need Kevin's OK. **Kevin (21:47Z): focus on the Miles replication, make it exactly the paper before relaunching.** Paper-fidelity audit against the authors' verl code done (Log 21:47Z); Miles `1f083baed` matches every controllable setting (trainer-side student log-prob, token-mean loss, verl AdamW 0.01 / 0.9 / 0.999, verl training suffix on all 14,116 DAPO prompts in `eopd-math-v2-train`, eos remap). Attempt 3 `01M2V7VG4H8Y3V1SYTFSX728KT` launched 21:48Z (image `01M2V7V946ZN9N9STPK0H04YWM`, Miles `1f083baed`, root `-v3`). |
| 5 | Teacher-entropy diagnostic on Qwen3-8B and on our verifier-DPPO teachers: histogram, % tokens with H>0.8, top-16 mass, % student tokens outside teacher top-16 (paper Fig. 3 / Fig. 9). | D4 gate; calibrates tau for our teachers. | In progress: `open_instruct/teacher_entropy.py` + `scripts/eopd/teacher_entropy_diagnostic.py` (OI commit 52de60681; exact entropy, top-16 mass, EOPD's renormalized top-k proxy and its gate agreement, sampled-token teacher rank). First run `01M2RZXXTT0GBCQSJCFKYAA76C`: Qwen3-8B on 256 Qwen3-4B-Base DAPO rollouts (T 1.0, 4096 tokens), output under `deletable_checkpoint/kevinfarhat/eopd/teacher_entropy/`. Second run `01M2S08RST44V0RFTGRFRSS72Y` **done**: verifier-DPPO 9B teacher on 256 Qwen3.5-4B rollouts over our math prompts (796k tokens): mean H 0.24, median 0.003, 11.4 % of tokens above tau 0.8, top-16 mass 0.9994, 0.21 % of student tokens outside the teacher's top-16 (1.6 % among high-entropy tokens), proxy gate agreement 99.87 % (false-negative 1.1 %), 54 % of rollouts hit the 4096-token cap. First run `01M2RZXXTT0GBCQSJCFKYAA76C` **done** (paper pair): mean H 1.26, 31 % above tau, top-16 mass 0.91, 12.4 % of student tokens outside the teacher's top-16, proxy gate agreement 98 %. Both runs recorded under "Step 5 results"; step 5 measurement complete, tau 0.8 retained for the replication. |
| 6 | Implement EOPD in Miles (teacher top-k via SGLang `top_logprobs_num`, top-k-renormalized entropy proxy, student log-probs gathered at teacher indices under TP, gated FKL term, audit + metrics). A/B vs the matched baseline, 2 seeds. Target: +1.8 Avg@8 / +5 Pass@8 at 4B. | Methodology validation. | **Coded and GPU-validated 2026-09-18** (Miles commits f78a618a1 / 50dab1a1d, off by default; see "Step 6 design"): `eopd_math` (renormalised top-k, entropy-proxy gate, FKL, TP-sharded student log-probs at the teacher ids), `eopd_loss.policy_loss` (upstream policy loss + `alpha * gate * FKL`), hooks requesting `top_logprobs_num = k` and storing the top-k in `train_metadata`, `[distillation] eopd/eopd_alpha/eopd_tau/eopd_top_k`, runtime patch (`metadata` in the train-step keys), `audit.json` `eopd` block, `configs/miles/opd/eopd-eopd-qwen3-tiny.toml`. Local tests: 65 passed incl. a two-rank gloo check that the sharded student log-probs and their gradients match the dense computation. EOPD tiny smoke `01M2S4MQ9MKXWSQFZSEWX5C9CC` (Qwen3-1.7B-Base ← Qwen3-8B, k=16, τ=0.8, α=1) passed end to end: 15 runtime tests, top-k in the trainer dumps' `metadata`, `audit.json["eopd"]` gate fraction 0.30 / 0.33 / 0.16 per rollout with proxy entropy 0.59 / 0.71 / 0.34 and top-k mass 0.95 / 0.91 / 0.97, all 12 optimizer steps log finite `train/eopd_fkl_loss` ≤ `train/eopd_fkl` (e.g. step 0: 0.83 vs 1.81, gate 0.41), advantage identity error 0.0, export reloads. Next: the A/B arms (step 4) once compute is approved. |
| 7 | Port the winner to the Qwen3.5 verifier-teacher setting; consider the Thinking Machines 8B-from-32B AIME24 run as the large-scale validation. | | Not started. |

### Paper targets (EOPD Table 2, OPD column, Avg@8 / Pass@8)

| Student (data) | MATH500 | AMC23 | Minerva | OlympiadBench | AIME24 | AIME25 |
|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base (MATH) | 67.76 / 84.80 | 39.06 / 70.00 | 29.83 / 47.06 | 30.09 / 51.56 | 8.33 / 20.00 | 6.25 / 16.67 |
| Qwen3-4B-Base (DAPO-Math-14k) | 78.81 / 90.80 | 57.33 / 80.00 | 40.08 / 54.00 | 42.10 / 58.80 | 18.33 / 26.67 | 12.08 / 30.00 |

EOPD deltas over OPD at 4B: +1.80 Avg@8, +5.05 Pass@8 averaged over the six benchmarks.
Paper hyperparameters: tau=0.8, alpha=1.0, k=16 (their README launch config says 32).

## Open questions for Kevin / Robert

- Compute and cluster for steps 2c and 4-6 (one 8-GPU node per Miles run; 4 nodes for the
  Open Instruct 2B ablation).
- Step 2d: fix or fence the async path.

## Where we are

Updated 2026-09-18 21:47Z (details in the latest Log entries).

- Done: step 1 (Qwen3.5 4B Miles replication closed out, both frameworks agree at every matched
  step), step 2b (Open Instruct OPD hardening), step 3 (Qwen3 Miles path: Qwen3-1.7B-Base ←
  Qwen3-8B tiny smoke passes end to end with minibatching, cosine LR, aime24 eval, audit and
  export reload), step 5 (teacher-entropy diagnostic), step 6 code (EOPD in Miles, GPU-validated
  by the tiny EOPD smoke; the A/B itself is part of step 4's compute).
- Step 4 arm 2 OPD baseline (Qwen3-4B-Base ← Qwen3-8B on DAPO-Math-14k): attempt 1
  `01M2SD1X7WTE4KCKAA4XR7YZQ8` failed at 05:02Z (verifier timeout fatal; fixed in Miles
  `0058e0028`). Attempt 2 `01M2SEQX9NMEZ6K13RJ586FZJ0` (root `-v2`, image
  `01M2SEQPBZRNP1HBWC72N71JCQ`) is at rollout 143/220 at 21:07Z, uninterrupted since 05:12Z,
  ETA ~2026-09-19 07:00Z (7.4 min/rollout including evals). **It has an eos-mismatch bug**: the learner stops with
  `<|endoftext|>`, the chat teacher never does (log-prob ≈ -21 nats there), so OPD punishes
  stopping; training truncation hit 100% by rollout 20 and every eval response sits at the
  8192 cap. MATH500 Avg@8 is still 0.764 at rollout 88 (paper 0.788) because the boxed answer
  comes early. Fixed in Miles `f92685054` (terminal learner eos scored as the teacher's eos;
  hook tests `01M2TT0R0TAR5PGXJ1PG42NYEP`). Attempt 2 runs to completion as the buggy
  reference; **the EOPD arm is on hold** (it would inherit the bug) and is NOT auto-launched.
  Kevin (21:47Z): the Miles replication is the focus; Open Instruct only if needed. Every
  controllable setting now matches the authors' verl code (Miles `1f083baed`, see the 21:47Z Log
  entry) and **attempt 3 `01M2V7VG4H8Y3V1SYTFSX728KT` launched 21:48Z** (image `01M2V7V946ZN9N9STPK0H04YWM`, root `-v3`).
  Watch early: `rollout/truncated` must stay well below 1.0 and `teacher-scores.jsonl`
  records must show `eos_remapped: true`. The EOPD arm follows on the same image once the
  baseline lands near 0.788; it now runs with trainer-side student log-probs too, so it wants a
  tiny EOPD smoke on that path first (never exercised on GPUs).
- Beaker's log endpoint returned nothing for the arm 2 job from ~17:40Z (`beaker job logs`,
  even with `--tail`), while `training.log` on Weka is fresh; progress is read from Weka
  (`scratchpad/weka-check/spec_read_arm2_v2_progress.yaml`).
- Step 2d (async fix, "fix not fence" per Kevin): `--fixed_prompt_batches` (commits
  `25e39f4db`, `b6f315bba`, depth fix `efc449cf7`). **GPU verification result (negative):** the
  2B async-4 fixed-batches run `01M2SPXTBP04X0HD7FPDTKN3WZ` collapsed like the canonical async
  run (eval 0.37 → 0.03, stop rate → 0.00) while composition behaved as designed (no drops,
  parking as expected). So composition alone is not the async collapse; staleness (in-flight
  updates + `async_steps` lag) is the leading suspect. The run was also 9 versions stale
  instead of 4-5 because of a depth bug in my refill logic, now fixed (`efc449cf7`, unit tests
  `01M2TQXK2Z2204PHY71YAC2XJS`). Decision for Kevin: rerun with the fix (~2h15m, 4 nodes) to
  isolate composition from staleness, or accept "async OPD needs on-policy data" and move on
  with the sync path (`synchronous_rollouts`) for OPD arms.
- Overnight plan (Kevin asleep from 07:15Z; Claude monitors on a 30-min cadence): keep arm 2
  alive (relaunch on the same image `01M2SEQPBZRNP1HBWC72N71JCQ` if a window dies without
  auto-resume); when it finishes collect the final MATH500 Avg@8 and record it. No EOPD launch,
  no fixed-batches rerun and no attempt 3 without Kevin (new compute). Push notifications only
  for unfixable failures or milestones (one sent ~17:40Z with both findings).
- Skipped: step 2c (Kevin, 2026-09-18).
- Later: step 7 (port EOPD to Open Instruct once the Miles A/B says it is worth it).
- Deferred: Pass@8 eval for the tiny config (only needed when an arm reports it).


## Step 6 design: EOPD on top of Miles's native OPD path

Upstream Miles at the pinned revision (`dbbab156`) already has the plumbing we need, so EOPD
is an extension of its OPD path, not a parallel implementation:

- **Teacher top-k (rollout side).** `miles/rollout/on_policy_distillation.py::_score_payload`
  already sends `top_logprobs_num = k` and reads `input_top_logprobs` per position, aligned to
  `logprob_start_len = 0`. Our hook (`open_instruct.miles.opd_hooks.reward`) keeps the
  sampled-token score it fetches today and additionally requests `top_logprobs_num = k` (k = 16),
  storing `teacher_topk_ids [R, k]` and `teacher_topk_logprobs [R, k]` on the sample. The Miles
  patch already adds `teacher_log_probs` to the rollout→train conversion key list
  (`train_data_conversion.py`); the two new keys go in the same list.
- **Gate (rollout side).** `open_instruct.teacher_entropy.token_statistics` gives the paper's
  proxy: entropy of the renormalized top-k teacher distribution. Gate `g_t = 1[H_proxy,t > tau]`
  (tau = 0.8) is computed once per token when the teacher scores arrive and stored as
  `opd_fkl_gate [R]`, so the trainer never re-derives it. Step 5's diagnostic tells us how well
  the proxy tracks the exact entropy for our teachers (`proxy_gate_agreement`).
- **Loss (Megatron side).** Miles accepts `--loss-type custom_loss
  --custom-loss-function-path <module.fn>`. `open_instruct.miles.eopd_loss.policy_loss` wraps
  upstream `policy_loss_function` (unchanged sampled-token OPD advantage, PPO clip) and adds
  `alpha * sum_t g_t * FKL_t / num_tokens`, with `FKL_t = sum_{j in topk_t} q_t(j) *
  (log q_t(j) - log p_S,t(j))`, `q_t` the renormalized teacher top-k. Student log-probs at the k
  teacher indices under tensor parallelism: `log p_S(j) = logit_j - logsumexp_V(logits)`; the
  logsumexp follows `_VocabParallelEntropy` (max all-reduce, sum-exp all-reduce) and the
  gathered `logit_j` comes from the owning vocab shard with a SUM all-reduce of zeros elsewhere.
  Both reductions follow Megatron's vocab-parallel convention (`_VocabParallelCrossEntropy`,
  `_VocabParallelEntropy`): the loss is replicated on every TP rank, so the SUM all-reduce is an
  identity in backward (not `torch.distributed.nn.functional.all_reduce`, whose backward would
  sum the identical gradients again and scale them by TP). Chunked over tokens like
  `calculate_log_probs_and_entropy`; only `[tokens, k]` extra activations. Verified on CPU with a
  two-rank gloo group against the dense log-softmax (values and gradients).
- **Wrapper.** `[distillation] eopd = false, eopd_alpha = 1.0, eopd_tau = 0.8, eopd_top_k = 16`
  in `opd_config`; when `eopd = true` the runtime emits the custom loss path and the extra
  request fields; validation requires `use_rollout_logprobs = true` and `top_p = 1.0`.
- **Audit and metrics.** Dump `teacher_topk_*`, `opd_fkl_gate` and the per-token FKL through the
  existing contract records; `opd_audit` re-derives the gate from the dumped top-k logprobs and
  checks `FKL_t` on a sample of positions. Metrics per rollout: gated fraction, mean proxy
  entropy, mean FKL, mean top-k mass, plus the existing reverse KL.
- **Carrying the top-k data to the loss (checked against upstream `dbbab156`).** `Sample` has no
  top-k fields, but `sample.train_metadata` already travels to the trainer as the `metadata` list
  (`train_data_conversion.py`, packaged per DP shard by our runtime patch). The Megatron train step
  requests a fixed key list from the micro-batch iterator (`megatron_utils/model.py::train_one_step`:
  tokens, loss_masks, log_probs, advantages, rollout_log_probs, opd_reverse_kl, ...), so the only
  runtime change is adding `metadata` to that list in `runtime/miles/patches/miles.patch`; the
  custom loss then reads `teacher_topk_ids`, `teacher_topk_logprobs` and `opd_fkl_gate` from each
  sample's metadata dict and tensorises them on device (micro-batch size is 1). The debug dump
  (`debug/train_data/<rollout>_0.pt`) carries `metadata` too, so `opd_audit` can re-derive the
  gate and check the FKL term offline.
- **Image.** The wrapper code is `open_instruct/miles`, but the one-line key-list change lives in
  the runtime patch, so the image must be rebuilt through `runtime/miles/Dockerfile`'s
  `runtime-base` stage (which re-applies the patch); still a local build, no new base image.
- **Paper check (resolved 2026-09-18).** Eq. 9-10 of arXiv 2603.07079: `L_EOPD = L_OPD +
  alpha * 1[H_t > tau] * L_FKL`, a hard gate, with `L_FKL = sum_{x in S_k^t} q~(x) log(q~(x) /
  p_theta(x))` summed over the **teacher's** top-k `S_k^t`, the teacher renormalised over that set
  and the student left as its full-vocabulary probability at those indices. That is exactly the
  design above. The paper writes `H_t` of the unrestricted teacher distribution; an SGLang teacher
  only returns top-k log-probs, so our gate uses the renormalised top-k entropy, which step 5
  shows agrees with the exact gate on 99.9 % of tokens for our teacher (false-negative 1.1 %).
  Paper Fig. 9: 15-20 % of tokens exceed tau 0.8 once training stabilises; our pre-training
  measurement on the Qwen3.5 pair is 11.4 %.

## Step 5 results: teacher entropy over student rollouts

Per-token teacher statistics over 256 student rollouts (temperature 1.0, top-p 1.0, 4096-token
cap), scored by `scripts/eopd/teacher_entropy_diagnostic.py`; outputs under
`deletable_checkpoint/kevinfarhat/eopd/teacher_entropy/`.

| Pair (teacher on student, prompts) | Job | Tokens | Mean H | Median H | % H > 0.8 | Top-16 mass | % sampled outside top-16 (all / H>0.8) | Proxy gate agreement / FN | % rollouts truncated |
|---|---|---|---|---|---|---|---|---|---|
| Verifier-DPPO 9B on Qwen3.5-4B, qwen35-math-v1 train | `01M2S08RST44V0RFTGRFRSS72Y` | 796,108 | 0.243 | 0.003 | 11.4 | 0.9994 | 0.21 / 1.6 | 99.87 % / 1.1 % | 53.9 |
| Qwen3-8B on Qwen3-4B-Base, DAPO-Math-14k (paper pair) | `01M2RZXXTT0GBCQSJCFKYAA76C` | 422,257 | 1.263 | 0.034 | 31.2 | 0.909 | 12.4 / 37.8 | 97.97 % / 6.5 % | 27.0 |

Reading of the paper pair (Qwen3-8B teacher, Qwen3-4B-Base student, before any training): the
teacher is far less certain about the base student's tokens than our verifier teacher is about
Qwen3.5-4B's. Mean entropy is 1.26 nats, 31 % of tokens exceed tau 0.8 (the paper reports 15-20 %
once training stabilises, so the gated fraction should fall as the student moves toward the
teacher), the teacher's top-16 holds only 0.91 of its mass (0.71 on high-entropy tokens), and 12 %
of the student's sampled tokens fall outside the teacher's top-16 (38 % among high-entropy
tokens). Consequences for step 6: the top-16 forward KL discards a tenth of the teacher's mass on
average and nearly a third on the very tokens it is meant to teach, so the renormalisation in
Eq. 10 matters and the term should be reported alongside its mass coverage; the top-k entropy
proxy is noticeably biased low (mean 0.61 vs 1.26, Pearson 0.90) yet still agrees with the exact
gate on 98 % of tokens with a 6.5 % false-negative rate, which is acceptable for a replication
but argues for logging the exact fraction when the trainer can compute it. 27 % of rollouts hit
the 4096-token cap.

Reading of the Qwen3.5 pair: the teacher is confidently peaked on almost nine tokens in ten
(median entropy near zero), so an entropy gate at tau 0.8 selects about 11 % of positions, the same
order as the paper's "high-entropy minority". Top-16 mass is essentially complete even on
high-entropy tokens (0.995), so a top-16 forward KL loses almost no mass and the renormalized
top-k proxy reproduces the exact gate on 99.9 % of tokens. Student tokens almost never fall
outside the teacher's top-16 (0.2 %), which bounds how much the sampled-token reverse KL can
miss relative to a full-vocabulary term. The 54 % truncation rate at 4096 tokens is a property of
the untuned Qwen3.5-4B student at temperature 1.0 and is why the sync campaign caps responses
lower; it does not affect the per-token statistics above.

## Log

- **2026-09-17 21:10Z** Miles 4B hf-79 evaluated: 79.0/56.7/50.0/90.0 vs sync step 80
  78.0/50.0/60.0/90.0 (DAPO/AIME/BRUMO/MATH-500). 4B 8/8, 2B 10/10 in band.
- **2026-09-17 ~22:30Z** Audit of the Open Instruct OPD path (flaws 1-6 above). Deep dive on
  EOPD (arXiv 2603.07079); decisions D1-D4 recorded. This tracker created.
- **2026-09-17 23:20Z** Step 2a done. Sync runs: `debug/vllm_local_reverse_kl` 1e-4 to 2e-4 at
  every step vs `objective/opd_reverse_kl` 0.033 -> 0.0005 (2B) and 0.07 flat (4B); DPPO mask
  kept 99.997%. Async canonical runs: 2B gap 0.01-0.036 and mask kept 92-96%; 4B gap ~1e-3, mask
  kept 99.0-99.9%. Conclusion: mixed numerics are a minor issue; staleness under async is the
  real one. Added `validate_opd_logprobs` (NaN/inf/positive logprob inside the response mask now
  raises) with tests; documented the temperature coupling and the async warning in
  `on_policy_distillation.md`. Proposed skipping step 2c.
- **2026-09-18 00:20Z** Kevin: treat `robertb/miles-qwen35-opd` as our branch (D5). Step 3
  starts there. 4B Miles job R8CZMP still running (hf-89 export expected ~00:30Z).
- **2026-09-18 00:45Z** Step 2b: trace dumps now carry teacher logprobs + advantages;
  `opd_trace_audit.py` (identity + alignment + finiteness, mirrors Miles `opd_audit`) with 9
  tests; Beaker real-dump job building (Qwen3-0.6B <- 1.7B smoke, 16 steps). Step 3: Miles
  wrapper knobs, Qwen3 models, EOPD data script and two run specs committed (bc5965b0a and
  follow-up). Paper check via arXiv HTML: teacher non-thinking (`<think></think>` prefix),
  suffix "Please reason step by step, and put your final answer within \boxed{}", Qwen
  top-p 1.0 for training and eval (D3 corrected), AdamW, 4xA100.
- **2026-09-18 00:40Z** hf-89 export present; eval launched. Local Docker image build fails
  (CUDA extension build, out of memory), so Open Instruct GPU jobs run the campaign image
  `01M01EPKXMXR4502S1HNJVYN0M` with the pushed branch cloned in-job (`CODE_REF`). Real-dump
  audit job launched. EOPD prompts rendered on Weka. Tiny Qwen3 Miles smoke spec committed.
- **2026-09-18 00:55Z** First real-dump audit job failed in 42 s: the image's editable install
  shadowed the cloned branch, so `grpo_fast.py` (branch) met a stale `StreamingDataLoaderConfig`
  (image). Launcher now overlays the clone onto `/stage/open_instruct` (the same trick the
  post-hoc eval script uses); relaunched as `01M2RZ8YYFT927SM1EDC7646ST`.
  Miles: `MILES_EXISTING_IMAGE` cannot carry the new wrapper knobs because the job runs
  `python -m open_instruct.miles train` from the image's own `/opt/core-rl` copy, so the tiny
  Qwen3 smoke needs a fresh overlay image (`runtime/miles/Dockerfile` only copies source and
  pip-installs wheels onto the pinned base, which builds locally). Building it and launching the
  Saturn CPU `prepare` phase for Qwen3-1.7B-Base / Qwen3-8B assets first
  (`name="eopd-opd-qwen3-tiny-prepare"`), then the 4-GPU smoke on the same image.
- **2026-09-18 00:45Z** hf-89 greedy on band with sync step 90 (rows appended to the sync-rerun
  doc). Pinned Miles base image had disappeared from local Docker; re-pulled from Beaker (docker
  id matches `runtime.lock.json`), built overlay image `01M2RZHM7E5Z91M10H9SJBMF85` from
  37f2bd8c5, launched the Saturn prepare job `01M2RZHTYE1Y6H0DB9JR0NKX2W`. Also added
  `MILES_CODE_OVERLAY=1` to the Miles launcher (in-job fetch of the committed HEAD over the
  exercised image) as the fallback when the image cannot be rebuilt; documented in
  `docs/miles/opd.md`. Real-dump audit relaunch `01M2RZ8YYFT927SM1EDC7646ST` still running.
- **2026-09-18 00:50Z** Prepare job rejected on the eos mismatch (see step 3 row); fixed with
  `model.align_eos_with_teacher`, enabled in all three EOPD specs. Step 5 tooling landed and its
  first job launched (Qwen3-8B teacher on Qwen3-4B-Base rollouts). Real-dump audit at step 15/16.
- **2026-09-18 00:58Z** Real-dump audit, first pass: training ran 16 steps and dumped every
  trace; the in-job audit command failed only because mason appends its own flags to the end of
  the command line (launcher fixed). Re-running the audit on CPU over the saved traces failed
  every record on shape alone: I had dumped `advantages` in the full query_response frame
  (`[B, T+1]`) while the logprobs and mask are `[:, 1:]`-shifted; the trainer consumes
  `advantages[:, 1:]`, so this was a dump inconsistency, not a training defect. Fixed the dump
  (OI 70dd732e1) and made the audit slice full-frame dumps the same way; CPU audit rerun
  `01M2S0BG4NP4878N18VASKNY6P`. Step 5 second run launched: verifier-DPPO 9B teacher on
  Qwen3.5-4B rollouts (`01M2S08RST44V0RFTGRFRSS72Y`).
- **2026-09-18 01:02Z** Real-dump audit passed (step 2b done): max advantage error 0.0 over 65k
  response tokens across steps 1/2/8/16; alignment verified on real dumps. Miles prepare attempt 3
  failed on the tiny spec's own context limit (a MATH train prompt of 1681 tokens > 1536); tiny spec now uses
  4096 context and root v3, prepare attempt 4 launched. Upstream Miles at the pinned revision
  already ships top-k OPD options (`--opd-log-prob-top-k`, `--opd-top-k-strategy`,
  `--opd-reward-weight-mode`, `--opd-topk-per-position`, in `miles/rollout/on_policy_distillation.py`
  and `loss_hub/opd.py`); step 6 should build on that path rather than a parallel one. The wrapper
  currently pins `log_prob_top_k = 0`.
- **2026-09-18 01:05Z** Miles prepare passed on attempt 4 (data only; model assets were staged by
  the earlier attempts). Tiny Qwen3 smoke launched: `01M2S0PHCCAKCY6B4WHWYN257J`. It exercises the
  Qwen3 Megatron profile, the 4-steps-per-rollout schedule, cosine LR, eos alignment and Avg@k eval
  end to end before any paper arm spends GPU time.
- **2026-09-18 01:12Z** Tiny smoke `01M2S0PHCCAKCY6B4WHWYN257J` exited 2 in 8 s: "Run configuration
  changed; choose a new output.root". Cause: the prepare phase had been launched into the training
  root (v3) and left a completed `workflow.json` there; `docs/miles/opd.md` runs preparation in its
  own root. Relaunched into root v4 as `01M2S132HYR6PX5Z1GNH2VPE3N` (Miles commit b3c895243, same
  image; assets are already staged). Step 5: verifier-DPPO teacher diagnostic finished, results in
  "Step 5 results"; launcher fixed so an empty revision is omitted (OI c5ab6d759). Qwen3-8B
  diagnostic still scoring.
- **2026-09-18 01:15Z** Log clock corrected: the six entries above from 00:45Z on had been stamped
  up to two hours ahead of UTC; times now follow the Beaker job timestamps.
- **2026-09-18 01:20Z** Step 5 complete: the Qwen3-8B / Qwen3-4B-Base diagnostic finished (results
  table). The paper pair is a much harder distillation target than our verifier pair: 31 % vs 11 %
  gated tokens, top-16 mass 0.91 vs 0.999, 12 % vs 0.2 % of student tokens outside the teacher's
  top-16. Smoke relaunch `01M2S132HYR6PX5Z1GNH2VPE3N` exited 1 inside the learner's HF-to-Megatron
  conversion (Qwen3-1.7B-Base, TP2); reading the converter log next.
- **2026-09-18 01:27Z** Smoke conversion failure diagnosed from `conversion.log`: `ProcessGroupNCCL::scatter:
  invalid tensor size (expected (76032, 2048), got (75968, 2048))`. Megatron rounds the vocabulary
  up to a multiple of 128 x TP (152064 for Qwen3 under TP2) while mbridge 0.15.1 splits the HF
  embedding's 151936 rows as they are. Miles honours `--padded-vocab-size`, so the Qwen3 learner
  profiles now pin it to 151936 (a multiple of 128; checkpoint and HF export keep the HF shape).
  Qwen3.5 never hit this because 248320 is a multiple of 256. Image `01M2S1VTS5N846S0Z9155QXFNH` built from
  abaff5d6a; smoke relaunched as `01M2S1W1XKQ93S4XXEXWCG0G5K` (root v5). Step 6 data path settled (see design): teacher top-k travels in
  `train_metadata`, one key added to the Megatron train-step batch list in the runtime patch.
- **2026-09-18 01:46Z** Smoke `01M2S1W1XKQ93S4XXEXWCG0G5K` exited 1 at 01:37Z after 10 min: conversion
  passed (padded-vocab fix confirmed), runtime tests 10 passed, teacher up, aime24 pre-eval ran
  (Avg@4 0.0 for the untrained 1.7B base, as expected), first rollout of 8 collected, then
  `miles/ray/rollout/metrics.py::_compute_zero_std_metrics` raised `type dict doesn't define
  __round__`: with 1 sample per prompt every group is zero-variance and Miles rounds
  `sample.reward`, which our hook set to the teacher's payload dict. The Qwen3.5 runs never hit it
  because two distinct payloads per group are never "equal". Fix (Miles f78a618a1): the reward
  hook returns 0.0 and keeps the payload in `sample.metadata`; `post_process` derives
  `teacher_log_probs` from it. Same commit lands the step 6 EOPD code (off by default) and the
  runtime-patch key-list change, so the image is rebuilt from f78a618a1 and smoke #4 launches
  into root v6 as `01M2S32Q4NE37G0N4TAZDGA4EH` (image `01M2S31VDMXVBAG5DY65119A7Q`). 4B job `01M2NJB63SB0E636G6JPJ63KZ2` still running; hf-99 not yet seen.
- **2026-09-18 01:57Z** 4B Miles replication (`01M2NJB63SB0E636G6JPJ63KZ2`) finished training: job
  exited 1 at 01:47Z in the image's stale audit (`KeyError: 'log_probs'`, the pre-fix audit code),
  as with the 2B run. Manual audit `01M2S39293NTSRJQGYWB2Y183R` on the run root with current
  code: 100 optimizer steps / 100 updates, student key `rollout_log_probs`, max advantage error
  0.0, min signal 10.37, hf-99 export complete (370 changed tensors vs hf-89, 312 frozen-base
  tensors, language-only scope); `audit.json` written. hf-99 greedy post-hoc eval launched as
  `01M2S3ETXCA0KKM6KY1WD1KN6R`; rows land in the rerun doc when it finishes.
- **2026-09-18 01:57Z** Smoke #4 `01M2S32Q4NE37G0N4TAZDGA4EH` exited 1 at 01:50Z before training:
  in-image runtime tests 13 passed, 1 failed (`test_eopd_post_process_stores_the_top_k_for_training`
  asserted exact equality on float32-rounded log-probs; `tests/miles/test_opd_hooks.py` imports
  Miles and cannot be collected on the Mac, so the GPU job was its first run). Test fixed (Miles
  c098412e0, tolerance compare); smoke #5 launched into root v7 as `01M2S3K7CNWARHB1VEPHTYXCFW` (image `01M2S3K0X8R43DJTJFEFEQBY87`). The
  `Sample` dataclass in the image has both `metadata` and `train_metadata`, as the EOPD path assumes.
- **2026-09-18 02:06Z** Step 1 complete. hf-99 greedy eval `01M2S3ETXCA0KKM6KY1WD1KN6R` (exit 0,
  02:03Z): MATH-500 90.0 / DAPO 76.0 / AIME25 46.7 / BRUMO 60.0 vs sync step 100 90.0 / 76.0 /
  53.3 / 60.0; the AIME gap is two questions of 30. All ten 4B Miles exports sit in the Open
  Instruct sync band, matching the 2B result. Rows and closing prose added to
  `docs/algorithms/qwen35_math_opd_sync_rerun.md` and `docs/experiments/qwen35_math_opd_results.md`.
  Smoke #5 `01M2S3K7CNWARHB1VEPHTYXCFW` running since 01:57Z (past the runtime-test point of smoke #4).
- **2026-09-18 02:20Z** Smoke #5 `01M2S3K7CNWARHB1VEPHTYXCFW` ran the whole training chain: runtime
  tests passed, teacher up, aime24 pre-eval, 3 rollouts × 8 prompts with the numeric reward
  (`zero_std` metric no longer trips), 12 policy-loss calls (4 optimizer steps per rollout),
  24 teacher-score records, aime24 post-eval, hf-2 export with `.complete`. It exited 1 at
  02:08Z inside the audit's `complete_export`: the Qwen3-1.7B base is a single
  `model.safetensors` (no index) and has no `A_log` tensors, so the Qwen3.5-shaped checks
  (`model.safetensors.index.json`, FP32 `A_log`, `model.language_model.*` frozen extras)
  did not apply. Fixed in Miles 50dab1a1d: weight maps come from the index or the shards, the
  `A_log` and frozen-extras rules apply only to multimodal bases, and a plain LM must export
  every base tensor (`training_scope` "full model"); in-image audit tests 7 passed. Image
  rebuilt as `01M2S4ME34M3JHMMGT5S23TMXM` (Miles 66379207a); smoke #6 (OPD tiny, root v8) is `01M2S4MN9628M5C6900CRV3E4K` and the first EOPD tiny smoke (root v1, `configs/miles/opd/eopd-eopd-qwen3-tiny.toml`) is `01M2S4MQ9MKXWSQFZSEWX5C9CC`, both launched 02:13Z on that image.
- **2026-09-18 02:33Z** Smoke #6 `01M2S4MN9628M5C6900CRV3E4K` exit 0 at 02:27Z (12.5 min): `workflow.json`
  status complete; `audit.json` has 12 optimizer steps with finite nonzero grad norms and a cosine
  LR (2.95e-6 at step 0, 1.89e-6 by step 4), advantage identity satisfied on every rollout, export
  273 changed tensors / 0 frozen extras / scope "full model"; `export-reload.json` shows the hf-2
  export generating in a fresh SGLang. `result.json` status "trained". Reverse KL swings between
  minibatches (11.8 / 1.06 / 14.6 / 0.94 / 0.57 over steps 0-4): 2 prompts per minibatch from an
  untrained 1.7B base, expected at this scale. Step 3 (Miles replication prep) is functionally
  complete; only the Pass@8 eval (deferred) remains. EOPD tiny `01M2S4MQ9MKXWSQFZSEWX5C9CC`
  started 02:19Z, still running.
- **2026-09-18 02:40Z** EOPD tiny smoke `01M2S4MQ9MKXWSQFZSEWX5C9CC` exit 0 at 02:32Z (12.8 min), the
  first GPU run of the step 6 code. `workflow.json` complete, `result.json` trained, in-image
  runtime tests 15 passed (incl. the EOPD hook and audit tests). The trainer dumps carry
  `metadata` = `[{eopd_topk_ids, eopd_topk_logprobs}]` per sample, so the `train_metadata` →
  runtime-patch key-list path works. `audit.json["eopd"]`: gate fraction 0.30 / 0.33 / 0.16,
  proxy-entropy mean 0.59 / 0.71 / 0.34, top-k mass 0.95 / 0.91 / 0.97 over rollouts 0-2 (k=16,
  τ=0.8); advantage identity error 0.0 on every rollout; export 267 changed tensors, reload OK.
  Per-step `train/eopd_*` metrics are finite on all 12 steps, gate fraction 0.02-0.71 per
  minibatch, `eopd_fkl_loss` always below the ungated `eopd_fkl` (step 0: 0.83 vs 1.81; step 9:
  1.14 vs 1.39 with gate 0.71), and `train/loss` = OPD loss + α·gated FKL. Teacher top-k mass
  0.83-1.0 says the k=16 proxy sees most of the teacher distribution on these tokens, matching the
  step 5 diagnostic. Nit for the arms: the cosine schedule reaches LR 0.0 exactly at the last
  optimizer step (step 11 of 12), so the final step is a no-op; set a nonzero min-LR ratio or
  accept it. **Steps 3 and 6 are done; step 4 (baseline arms) is the next launch and needs
  Kevin's compute sign-off (see "Open questions for Kevin / Robert").**
- **2026-09-18 03:05Z** Re-read the paper's setup for Kevin's sign-off. Confirmed from the text:
  students Qwen3-0.6B-Base / 1.7B-Base / 4B-Base; teacher Qwen3-8B with thinking disabled (the
  post-trained Qwen3-8B, non-thinking template with an empty `<think>` block, which is what our
  prepared assets use); MATH for the 0.6B/1.7B students, DAPO-Math-14k for the 4B; batch 128 × 1
  sample, mini-batch 32 (4 gradient steps per iteration), LR 3e-6 cosine, AdamW, 4096-token
  responses, training temperature 1.0 / top-p 1.0 (Qwen), 3 epochs MATH / 2 epochs DAPO, k=16,
  4×A100-80GB, ~47.8 s per training step of which ~37.7 s is student generation. Evaluation:
  Qwen2.5-Math pipeline, zero-shot, 8 samples, T=1.0, **top-p 0.8**, 8192 tokens, Avg@8 and
  Pass@8 on MATH500 / AIME24 / AIME25 / AMC23 / Minerva / OlympiadBench. Our arm configs had
  eval top-p 1.0; fixed to 0.8 (Miles branch), D3 corrected. Kevin: skip step 2c (agreed), copy the
  paper's schedule exactly for the arms (min LR 0 stays).
- **2026-09-18 04:50Z** Kevin approved option 2. Arm 2 OPD baseline launched as
  `01M2SD1X7WTE4KCKAA4XR7YZQ8` (first attempt failed locally: the Docker daemon no longer had the
  `olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks` base tag; relaunched on the smoke image
  `01M2S4ME34M3JHMMGT5S23TMXM`, whose code is identical to the branch head, the only later
  commit being the eval top-p config change that travels with the launch spec; the base is being
  re-pulled from Beaker image `01M24E7MSDGN2QFW1T8Z31BCKS` for future builds). Step 2c skipped.
  Step 2d decided: fix, not fence. Design ("batch by prompt set"): prompts are enqueued in
  batches of `num_unique_prompts_rollout` tagged with a batch index; `async_steps` batches are in
  flight; training step `s` consumes batch `s` only when *all* of its results have arrived, while
  results for later batches are parked in per-batch buffers instead of being consumed early; no
  result is ever dropped for age; staleness (up to `async_steps` weight versions, and a response
  can still straddle an in-flight update) is corrected by the existing importance ratio between
  the stored rollout log-probs and the current policy (DPPO/PPO mask), which is what Miles does
  with its one-step-off pipeline. Batch composition is then a uniform random prompt set at every
  step, so the length ordering that steered pure-OPD updates disappears, and the generator stays
  busy because later batches keep generating while the trainer waits for the slowest response of
  the current one. Zero-std filtering and active sampling stay incompatible with this mode (they
  refill a batch from whatever finishes next, which is the bias again); pure OPD needs neither.
  Touch points: `data_loader.add_prompt_to_generator` (batch tag on `PromptRequest`, carried on
  `GenerationResult`), `accumulate_inference_batches` (per-batch buffering instead of first-N),
  `DataPreparationActor` (enqueue by batch, refill one batch per consumed batch), config flag
  `batch_by_prompt_set` (default on for pure OPD), unit tests with fake queues. Verification: the
  2B arm at `async_steps=4` should now track the sync curve where the canonical async run collapsed.
- **2026-09-18 05:35Z** Step 2d implemented in Open Instruct as `--fixed_prompt_batches`
  (commit `25e39f4db`; design as in the 04:50Z entry). Unit tests cannot run locally (vLLM is not
  importable on macOS), so they ran in Beaker CPU jobs on the campaign image
  (`01M2SDPBBNYVR73RDGPC28XY6D`, `01M2SDZECD67TK8EFS0K0HJNX3`): the four new tests (batch
  parking, finishing a batch from parked plus queued results, prompt tagging, config
  validation), the neighbouring accumulate tests and the ten `vllm_utils` tests pass. One
  pre-existing test, `test_accumulate_inference_batches_drops_stale_model_steps`, fails at the
  parent commit too: it expects one replenished prompt after a stale drop, but the accumulator
  has replenished one prompt per consumed result since that test was added (Hamish Ivison's
  "Drop stale async rollout results"), so three are queued. Expectation corrected to three.
  Arm 2 OPD `01M2SD1X7WTE4KCKAA4XR7YZQ8` started 04:46Z: in-image runtime tests 15/15, native
  preflight and conversion done, teacher server up by 04:52Z. Next: verification run for 2d
  (`qwen35_math_opd_fixed_batches_rerun.sh`, smoke on one node then the 4-node 100-step run at
  `async_steps=4`; needs a fresh code-patch dataset from this checkout and Kevin's compute OK).
- **2026-09-18 05:50Z** Arm 2 attempt 1 `01M2SD1X7WTE4KCKAA4XR7YZQ8` failed at 05:02Z, 16 minutes
  in: runtime tests, preflight, conversion and the teacher server were fine, and the step-0
  MATH500 eval (4000 samples, 8192 tokens, T=1.0, top-p 0.8) was 650/4000 through when one
  symbolic-math check hit the reward pool's 45 s timeout. The `TimeoutError` propagated out of
  `RolloutManager.eval` and ended training (`training.log`, root `-v1`). Cause: a base model at
  T=1.0 with an 8192-token budget emits responses that stall the LaTeX/sympy grader; Open
  Instruct's own verifiers score such a check 0. Fix in Miles `0058e0028`: `rewards.score`
  catches the timeout, scores the verifier 0 and records `timed_out` in the diagnostics (other
  verifier failures still raise); tests `tests/miles/test_rewards_timeout.py`. Attempt 2 launches
  on a rebuilt image (base image restored locally) with root `eopd-opd-qwen3-4b-base-dapo14k-v2`;
  `-v1` stays on Weka untouched. Step 2d unit tests re-ran at `b6f315bba`
  (`01M2SE7GZZM6TGZD07T5KDATA4`): 5 passed, including the corrected stale-drop test.
- **2026-09-18 06:05Z** Arm 2 attempt 2 launched: `01M2SEQX9NMEZ6K13RJ586FZJ0`, image
  `01M2SEQPBZRNP1HBWC72N71JCQ` built from Miles `0058e0028` (verifier timeout scores 0), root
  `eopd-opd-qwen3-4b-base-dapo14k-v2`. The build itself needed the local base image re-tagged
  by ID first: Docker Desktop kept listing `olmo-miles:gate-…` but could not resolve it by name.
- **2026-09-18 07:10Z** Arm 2 attempt 2 progress check (CPU read of root `-v2`,
  `01M2SNFRFB94FGF0GVVZ1F5R5Y`): rollout 19/220 done, checkpoint `iter_0000019` written, PPO
  clipfrac ~0.002-0.005, `opd_reverse_kl` ~0.07, ESS ~0.998, LR on the cosine schedule (2.94e-6).
  Step-0 eval completed without a verifier timeout ending the run. Only errors in the log are
  SGLang "request disconnected" aborts at 05:52Z from the eval's own cancellation, harmless.
  Projected finish ~2026-09-19 06:00Z; auto-resume across 8h windows.
- **2026-09-18 07:20Z** Kevin approved the 2d GPU verification ("go") and went to bed with
  "monitor and make sure these runs keep going". Built the code-patch dataset from `ee0d4845e`
  (`01M2SNQYARKRCZPZ6AN8CEJPGX`, 7 files incl. `data_types.py`, `vllm_utils.py`) and launched the
  one-node smoke `01M2SNTT1XKJZW2B7B2QZK1VCX` (`RUN_MODE=smoke`: async_steps 2,
  `--fixed_prompt_batches true`, 256 episodes = 4 batches of 32 prompts × 2). Success criterion:
  4 training steps, `results_parked_for_later_batches` logged, no drops, evals run. Then the full
  run. Arm 2 attempt 2 unchanged (rollout 19+). Overnight monitor cadence: every 30 min.
- **2026-09-18 07:30Z** Fixed-batches smoke `01M2SNTT1XKJZW2B7B2QZK1VCX` passed (exit 0 in 12 min):
  4 training steps at `async_steps=2`, `results_parked_for_later_batches` 27 / 0 / 16 / 0 (early
  results of the next batch parked, then consumed by their own step), `stale_results_dropped` 0
  throughout, `model_step_min` 1 (in-flight updates active), `opd_reverse_kl` 0.04 with
  `opd_teacher_logprob` about -0.48 per token. The parking path works on GPUs with the real vLLM
  actors. Launched the full run `01M2SPKXVMR0QT3FCZEYE5AKXD` (4 nodes on jupiter, 100 steps at
  async_steps 4, `--fixed_prompt_batches true --save_freq 10`, everything else the canonical
  recipe). Expected ~5h; first eval at step 0, then every 20 steps.
- **2026-09-18 07:35Z** Full fixed-batches run `01M2SPKXVMR0QT3FCZEYE5AKXD` exited 1 after 40 s:
  `wandb.init` rejected the tags because the exp name
  `qwen35_2b_opd_from_verifier_2b_fixed_batches_async4_lr1e6_100step_4node` is 71 characters
  (W&B caps tags at 64; grpo_fast adds the exp name as a tag). The smoke name was 57 characters,
  which is why the smoke passed. Shortened to `qwen35_2b_opd_fixed_batches_async4_lr1e6_100step_4node`
  (54; commit `1e51d4567`, launcher only, so the patch dataset is unchanged) and relaunched as
  `01M2SPXTBP04X0HD7FPDTKN3WZ`. No GPU time lost beyond the 40 s.
- **2026-09-18 17:15Z** Fixed-batches verification `01M2SPXTBP04X0HD7FPDTKN3WZ` finished at 09:50Z
  (100 steps in 2h14m, exit 0; the monitor session was interrupted and picked this up at 13:11Z).
  Result: collapse. Eval pass@1 (DAPO holdout / AIME25 / BRUMO aggregate) 0.37, 0.04, 0.15,
  0.34, 0.09, 0.03 at steps 0-100; training stop rate 0.67 at step 10, 0.46 at 20, 0.02 at 30-40,
  a partial recovery to 0.67 at step 60, then 0.00 at 100 with every response at the 16384 cap.
  Canonical async (`01M1WQ2DRFJF2C1019HMZ02318`): 0.37, 0.06, 0.47, 0.47, 0.53, 0.01 with the
  same oscillation; sync rerun: DAPO 47.7 at step 100. Composition metrics were as designed
  (375 parked at step 1, then 0-8 per step; 0 drops). Staleness was not: `model_step` lagged
  the data step by exactly 9 from step 11 on (steps 1-10 all at model step 1), against 4-5 in
  the canonical run. Cause: the data actor may run `async_steps` steps ahead of the trainer
  (`_last_consumed_step` throttle), and I refilled batch `s+async_steps` when the *actor*
  consumed `s`, so ~2×`async_steps` batches were in flight; the canonical path hides the same
  depth by dropping old results. Fix `efc449cf7`: `fixed_batches_due()` queues batch
  `t+async_steps` once the trainer has consumed `t`, checked on every result received while a
  batch assembles; unit test added; docs/changelog amended. Conclusion so far: on the 2B
  verifier-teacher setup, uniform batch composition at ~9-version staleness still collapses, and
  the canonical run collapses at 4-5, so staleness with in-flight updates is the prime suspect
  rather than length ordering. A rerun with the depth fix isolates the two; not launched (new
  compute). Arm 2 Miles baseline: rollout 125/220 at 17:07Z, uninterrupted since 05:12Z; MATH500
  Avg@8 0.244 (base) → 0.744 (rollout 44) → 0.764 (rollout 88) vs paper 0.788 at the end;
  eval responses report length 8192 = cap with `truncated_ratio` 1.0 and `repetition_frac`
  0.3-0.6 from rollout 44 on, under investigation (train-side stats job `01M2TQXKPGFNRQP8AV357SNAZR`).

### 2026-09-18 17:48Z

- Arm 2 attempt 2 diagnosed (Weka reads `01M2TQXKPGFNRQP8AV357SNAZR`, `spec_miles_src_eos`):
  train-side `rollout/truncated` 0.11 at rollout 1 → 1.00 at rollout 20 and after; response
  length mean at the 4096 cap; `rollout/opd_reverse_kl` stays positive. Cause: the learner ends
  answers with `<|endoftext|>` (151643) while the chat teacher's answers end with `<|im_end|>`
  (151645); `align_eos_with_teacher` only adds `<|im_end|>` as an extra stop id. The teacher's
  log-prob of the terminal `<|endoftext|>` is -21.4 nats on average (1500 `teacher-scores.jsonl`
  records) against ≈ -0.3 for ordinary tokens, so with `adv = kl_coef·(log π_T − log μ)` the stop
  token carries a huge negative advantage and the student learns never to stop. MATH500 Avg@8
  0.244 → 0.744 (rollout 44) → 0.764 (rollout 88) survives because the boxed answer is emitted
  before the rambling; eval `truncated_ratio` 1.0, `repetition_frac` 0.3-0.6.
- Fix (Miles `f92685054` on `robertb/miles-qwen35-opd`): `opd_prepare.teacher_eos_remap` reads
  the prepared learner's `generation_config.json` (`eos_token_id=[teacher_eos, learner_eos, ...]`)
  and the launcher exports `OI_OPD_TEACHER_EOS_REMAP="151643:151645"` when
  `align_eos_with_teacher` is set; `opd_hooks.tokens_for_teacher` replaces only a *terminal*
  learner stop id in the tokens sent to the teacher (plain `reward_func` path via a copied
  sample, EOPD `_score_payload` path), the sample keeps its real tokens, the ID check runs against
  the scored tokens, and the score record gains `eos_remapped`. Unit tests: 4 new hook tests +
  `teacher_eos_remap` test; local `open_instruct/test_miles_opd.py` 47 passed; hook tests in the
  Miles image `01M2TT0R0TAR5PGXJ1PG42NYEP` 10 passed, 1 pre-existing failure
  (`test_native_dataset_preserves_rendered_text_with_processor`, from `c531badd9`; the image's
  upstream `Dataset` differs). `docs/miles/opd.md` amended.
- Plan change: the EOPD arm is not launched when attempt 2 finishes (it would inherit the bug).
  Attempt 3 (OPD + eos fix, rebuilt image) then EOPD is Kevin's call; attempt 2 is kept running
  as the buggy reference (rollout 131/220 at 17:48Z). Beaker's log endpoint went blank for the
  arm 2 job around 17:40Z; monitoring continues from Weka `training.log`.

### 2026-09-18 21:13Z

- Arm 2 status check (Weka read `01M2V5RVM5Z53DTTV6GA06QZ03`): rollout 143/220 at 21:07Z, job
  still running, no errors. Third mid-run eval at rollout 131 (20:23Z): MATH500 Avg@8 0.7575
  (0.744 → 0.764 → 0.7575; paper 0.788), all eval responses still at the 8192 cap. Pace since
  rollout 87 is 7.4 min/rollout including evals, so the run should finish ~2026-09-19 07:00Z.


### 2026-09-18 21:37Z

- Paper-fidelity audit of the arm 2 Miles recipe against the paper (App. A / C, Sec. 5.1) and
  the authors' code ([github.com/WLS04/EOPD](https://github.com/WLS04/EOPD), a verl fork;
  files saved under `scratchpad/eopd_src/`). **Their trainer does the same eos remap we just
  added**: `OnPolicyDistillTrainer._replace_eos_token_for_teacher` swaps the first
  `<|endoftext|>` (151643) in each response for `<|im_end|>` (151645) on a copy of the batch
  sent to the teacher, "because student model uses <|endoftext|> while teacher expects
  <|im_end|>". This confirms the attempt 2 diagnosis and that Miles `f92685054` is the paper's
  behaviour. Their launch script (`on_policy_it.sh`) is not in the repo, so hyperparameters
  beyond Table 9 are verl defaults.
- Matches: models, non-thinking teacher and template (`<think>\n\n</think>`), DAPO-Math-14k,
  B=128 x 1 sample, mini-batch 32 (4 steps), LR 3e-6 cosine, no warmup, AdamW, 4096 train
  response, T=1.0 top-p 1.0, 2 epochs (220 rollouts), PPO clip 0.2, advantage
  `teacher − old` (kl_coef 1), no entropy bonus / KL-to-ref, eval T=1.0 top-p 0.8 8192 Avg@8 with
  the App. C suffix "Please reason step by step, and put your final answer within \boxed{}."
- Deviations: (1) student side of the advantage: verl recomputes `old_log_prob` in the trainer
  (`calculate_log_probs: False`); our arm uses `use_rollout_logprobs = true` (SGLang). Miles
  supports `false`; our EOPD wrapper currently requires `true`. (2) Loss aggregation: verl
  default `token-mean`; Miles default is per-response mean (`calculate_per_token_loss` off).
  (3) Optimizer details hard-coded in our launcher: weight decay 0.0, betas 0.9/0.98 vs verl
  defaults 0.01, 0.9/0.999. (4) Training prompt suffix: their preprocessing uses "Let's think
  step by step and output the final answer within \boxed{}."; we train with the App. C wording.
  (5) 14109 vs 14116 prompts (7 eval-overlap rows removed). (6) Teacher scored by an SGLang
  server (bf16 logprobs) vs an in-trainer verl ref worker; eval by our SGLang loop + verifiers
  vs the Qwen2.5-Math harness. (7) 8xH100 one node vs 4xA100. None of these is the truncation
  bug; (1)-(4) are cheap to flip before attempt 3 if Kevin wants an exact copy.

### 2026-09-18 21:47Z

- Kevin: focus on the Miles replication ("its not necessary to use open instruct unless we want
  or need to") and make it exactly the paper before launching attempt 3. Changes (Miles
  `1f083baed`, pushed): `distillation.use_rollout_logprobs = false` in both Qwen3 arm specs
  (verl recomputes the pre-update student log-prob in the trainer); new
  `training.loss_aggregation = "token"` → `--calculate-per-token-loss` (verl `token-mean`;
  default `response` keeps Miles' per-response mean); new `optimizer.weight_decay /
  adam_beta1 / adam_beta2` = 0.01 / 0.9 / 0.999 in the arm specs (verl AdamW defaults; wrapper
  defaults stay 0.0 / 0.9 / 0.98 so the finished Qwen3.5 runs are unchanged); training prompts
  re-rendered with the authors' preprocessing suffix "Let's think step by step and output the
  final answer within \boxed{}." into Weka `miles-opd/data/eopd-math-v2-train` (Beaker
  `01M2V7J17FESTS1DEBFHM41NC8`: 14,116 DAPO prompts, all kept as in the paper; 7,496 MATH),
  evaluation sets unchanged (`eopd-math-v1`, App. C suffix). The EOPD-requires-rollout-log-probs
  check was dropped (the loss does not depend on the student side). Flags confirmed in the Miles
  image (`01M2V7PHJA38CATP26FQ777VHW`). Tests: `open_instruct/test_miles_opd.py` 49 passed.
- Arm 2 attempt 3 `01M2V7VG4H8Y3V1SYTFSX728KT` submitted 21:48Z (image `01M2V7V946ZN9N9STPK0H04YWM`; the first build attempt hit the base-image tag flake, re-tagged `fe34fb1fef49`). Image build + launch started 21:47Z from `1f083baed`, root
  `eopd-opd-qwen3-4b-base-dapo14k-v3` (one holmes node, 8h windows with auto-resume, ~24h).
  Attempt 2 (`01M2SEQX9NMEZ6K13RJ586FZJ0`, rollout ~150/220) keeps running as the buggy
  reference until Kevin says otherwise.

