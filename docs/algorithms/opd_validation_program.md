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
| D3 | Evaluation for the replication and EOPD must be **sampled** (T=1.0, top-p 0.8, 8192 tokens, Avg@8 and Pass@8), matching the Qwen2.5-Math harness the paper uses. | Our greedy pass@1 eval cannot see the diversity effect the paper claims. |
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
| 1 | Close out the Qwen3.5 4B Miles v5 replication: eval hf-89, hf-99; manual `opd_audit` on the 4B root; final docs pass. | Completes the 2-framework cross-check (2B done 10/10, 4B 8/8 so far). | In progress: 4B job R8CZMP at rollout ~85; hf-89 ~00:30Z, hf-99 ~03:20Z 2026-09-18. |
| 2a | Measure the vLLM-vs-trainer logprob gap against the OPD signal on the sync runs (`debug/vllm_local_reverse_kl` vs `objective/opd_reverse_kl`, W&B `rg1a2gel`, `zz3q6skv`, `8nlm6azd`). | Decides whether flaw 2 matters. | **Done 2026-09-17** (Beaker `01M2RXC2NK173XAYCHEZRYDARY`). Sync: gap 1e-4 to 2e-4 at every step; 4B signal 0.07 (gap negligible), 2B signal decays 0.033 -> 0.0005 so the gap is ~30% of the signal only in the last 30 steps. DPPO mask keeps 99.997% of tokens, ratio 1.0000. Flaw 2 downgraded to minor. Async: gap 0.01-0.036 for 2B (100-300x sync), mask keeps only 92-96% of tokens; 4B async gap ~1e-3, mask 99.0-99.9%. Quantifies flaw 1. |
| 2b | Open Instruct hardening: real-dump OPD audit (advantage identity + alignment), NaN-inside-mask becomes an error, document the temperature coupling. | Flaws 3, 5, 4. | In progress: `validate_opd_logprobs` guard + tests landed; caveats documented; real-dump audit being scoped. |
| 2c | One 2B sync arm with `--use_vllm_logprobs false` (trainer-side student logprobs). | Direct test of flaw 2. Needs 4 nodes for ~4h. | **Proposed skip** after 2a: the gap is 1e-4 against a 0.07 signal at 4B and only matters for the tail of the 2B run. Kevin to confirm. |
| 2d | Decide: fix the async path for OPD or fence it off (assert `async_steps==1` when `opd_pure`). | Flaw 1. | Decision pending. |
| 3 | Miles replication prep: Qwen3-1.7B-Base, Qwen3-4B-Base, Qwen3-8B model profiles; expose minibatching (4 optimizer steps per rollout) and 1 sample/prompt in the OPD TOML; cosine LR; sampled Avg@8/Pass@8 eval matching the Qwen2.5-Math harness. | Paper setting: B=128, mini 32, LR 3e-6 cosine, 4096 response, T=1.0, 3 epochs MATH / 2 epochs DAPO-Math-14k. | In progress (2026-09-18): upstream Miles ships `qwen3-1.7B/4B/8B` profiles, so only the wrapper's model map, minibatch/LR-schedule/eval-sampling knobs and the prompt data need adding. |
| 4 | Run the OPD baseline: arm 2 (Qwen3-4B-Base from Qwen3-8B on DAPO-Math-14k), then arm 1 (Qwen3-1.7B-Base on MATH). Target: MATH500 Avg@8 within ~1 point of 78.8 / 67.8. | Infra validation against a public number. | Not started. |
| 5 | Teacher-entropy diagnostic on Qwen3-8B and on our verifier-DPPO teachers: histogram, % tokens with H>0.8, top-16 mass, % student tokens outside teacher top-16 (paper Fig. 3 / Fig. 9). | D4 gate; calibrates tau for our teachers. | Not started. |
| 6 | Implement EOPD in Miles (teacher top-k via SGLang `top_logprobs_num`, top-k-renormalized entropy proxy, student log-probs gathered at teacher indices under TP, gated FKL term, audit + metrics). A/B vs the matched baseline, 2 seeds. Target: +1.8 Avg@8 / +5 Pass@8 at 4B. | Methodology validation. | Not started. |
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

See the latest Log entry.

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
