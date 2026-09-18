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
| D3 | Evaluation for the replication and EOPD must be **sampled** (T=1.0, top-p 1.0 for Qwen students [0.8 was the paper's Llama setting], 8192 tokens, Avg@8 and Pass@8), matching the Qwen2.5-Math harness the paper uses. | Our greedy pass@1 eval cannot see the diversity effect the paper claims. |
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
| 1 | Close out the Qwen3.5 4B Miles v5 replication: eval hf-89, hf-99; manual `opd_audit` on the 4B root; final docs pass. | Completes the 2-framework cross-check (2B done 10/10, 4B 8/8 so far). | In progress: 4B job R8CZMP running; hf-89 greedy `01M2RYZ320TBZWY8C89W92D84E` = MATH-500 90.0 / DAPO 78.0 / AIME 46.7 / BRUMO 63.3 vs sync step 90 90.0 / 77.0 / 43.3 / 56.7 (on band); hf-99 + audit ~03:20Z 2026-09-18. |
| 2a | Measure the vLLM-vs-trainer logprob gap against the OPD signal on the sync runs (`debug/vllm_local_reverse_kl` vs `objective/opd_reverse_kl`, W&B `rg1a2gel`, `zz3q6skv`, `8nlm6azd`). | Decides whether flaw 2 matters. | **Done 2026-09-17** (Beaker `01M2RXC2NK173XAYCHEZRYDARY`). Sync: gap 1e-4 to 2e-4 at every step; 4B signal 0.07 (gap negligible), 2B signal decays 0.033 -> 0.0005 so the gap is ~30% of the signal only in the last 30 steps. DPPO mask keeps 99.997% of tokens, ratio 1.0000. Flaw 2 downgraded to minor. Async: gap 0.01-0.036 for 2B (100-300x sync), mask keeps only 92-96% of tokens; 4B async gap ~1e-3, mask 99.0-99.9%. Quantifies flaw 1. |
| 2b | Open Instruct hardening: real-dump OPD audit (advantage identity + alignment), NaN-inside-mask becomes an error, document the temperature coupling. | Flaws 3, 5, 4. | **Done.** Real-dump audit `01M2S0BG4NP4878N18VASKNY6P` over the Qwen3-0.6B ← Qwen3-1.7B gsm8k smoke (`01M2RZ8YYFT927SM1EDC7646ST`, 16 steps, `--save_traces`): steps 1/2/8/16, 127 records, 65k response tokens, `advantage == kl_coef·(teacher − rollout)` with max error 0.0, no non-finite logprobs inside the mask, teacher signal up to 32 nats per token (no adv clip in that smoke). The `[:, 1:]` alignment of rollout logprobs, teacher logprobs, mask and advantages holds on real data. Remaining under 2: decision 2d (async fence). |
| 2c | One 2B sync arm with `--use_vllm_logprobs false` (trainer-side student logprobs). | Direct test of flaw 2. Needs 4 nodes for ~4h. | **Proposed skip** after 2a: the gap is 1e-4 against a 0.07 signal at 4B and only matters for the tail of the 2B run. Kevin to confirm. |
| 2d | Decide: fix the async path for OPD or fence it off (assert `async_steps==1` when `opd_pure`). | Flaw 1. | Decision pending. |
| 3 | Miles replication prep: Qwen3-1.7B-Base, Qwen3-4B-Base, Qwen3-8B model profiles; expose minibatching (4 optimizer steps per rollout) and 1 sample/prompt in the OPD TOML; cosine LR; sampled Avg@8/Pass@8 eval matching the Qwen2.5-Math harness. | Paper setting: B=128, mini 32, LR 3e-6 cosine, 4096 response, T=1.0, 3 epochs MATH / 2 epochs DAPO-Math-14k. | **Mostly done 2026-09-18** on `robertb/miles-qwen35-opd`: Qwen3 model map (upstream `qwen3-1.7B/4B/8B` profiles), `training.optimizer_steps_per_rollout`, `optimizer.lr_decay_style/lr_warmup_iters/min_lr`, `inference.top_p/eval_top_p/eval_max_response_length`, `scripts/miles/prepare_eopd_math_prompts.py`, specs `configs/miles/opd/eopd-opd-qwen3-{4b-base-dapo14k,1.7b-base-math}.toml`. Data rendered on Weka (`miles-opd/data/eopd-math-v1`, Beaker `01M2RYRHTNKRMPKNVJXRT7SHTX`): math_train 7496 (4 MATH rows lack a boxed answer), dapo_math_14k 14109 (7 overlap eval sets), math_500 500, aime24 30, aime25 30, amc23 40, minerva 272, olympiadbench 674; non-thinking Qwen3 template, paper suffix. First prepare job `01M2RZHTYE1Y6H0DB9JR0NKX2W` was rejected: Qwen3-Base stops on `<|endoftext|>`, Qwen3-8B on `<|im_end|>` (vocab identical). Added `model.align_eos_with_teacher` (Miles commit d2256c371): the learner adopts the teacher's eos, keeps its own as a second stop id. Without it a base learner would run on after `<|im_end|>` until `<|endoftext|>` or the length cap, which is the likeliest way to silently miss the paper's numbers. Prepare succeeded on attempt 4 (`01M2S0HMZQ7W7VG9W06K4V5J0B`, assets `Qwen3-1.7B-Base-…-eos`, `Qwen3-8B-…`); tiny 4-GPU smoke `01M2S0PHCCAKCY6B4WHWYN257J` was rejected (prepare had been run in the training root; the documented flow gives preparation its own `output.root`), relaunched as `01M2S132HYR6PX5Z1GNH2VPE3N` into root v4, which failed in the HF-to-Megatron conversion (Megatron padded the 151936 vocabulary to 152064 under TP2; mbridge scatters the HF embedding unpadded). Fixed with repository profiles `qwen3-1.7B` / `qwen3-4B` pinning `--padded-vocab-size 151936` (Miles commit abaff5d6a); image `01M2S1VTS5N846S0Z9155QXFNH` built and the smoke relaunched as `01M2S1W1XKQ93S4XXEXWCG0G5K` into root v5 (Miles commit 7b6ccfa39: 3 rollouts × 8 prompts, 4 optimizer steps per rollout, cosine LR, aime24 Avg@4 eval). That smoke converted the learner fine (padded-vocab fix confirmed) and ran the aime24 pre-eval and the first rollout, then died in Miles's `zero_std` rollout metric: with one sample per prompt every group is zero-variance and the metric rounds `sample.reward`, which our reward hook had set to the teacher's scoring payload (a dict). The hook now returns the numeric task reward 0.0 and parks the payload in `sample.metadata` until `post_process` (Miles commit f78a618a1); smoke #4 relaunched into root v6 (see Log). Pass@8: the paper's headline is Avg@8, which Miles in-run eval reports (8 samples, T 1.0, top-p 1.0); a Pass@8 post-hoc path (Open Instruct `--eval_pass_at_k 8` over the rendered sets converted to `messages`/`ground_truth` JSONL) is scoped but deferred until an arm needs it. |
| 4 | Run the OPD baseline: arm 2 (Qwen3-4B-Base from Qwen3-8B on DAPO-Math-14k), then arm 1 (Qwen3-1.7B-Base on MATH). Target: MATH500 Avg@8 within ~1 point of 78.8 / 67.8. | Infra validation against a public number. | Not started. |
| 5 | Teacher-entropy diagnostic on Qwen3-8B and on our verifier-DPPO teachers: histogram, % tokens with H>0.8, top-16 mass, % student tokens outside teacher top-16 (paper Fig. 3 / Fig. 9). | D4 gate; calibrates tau for our teachers. | In progress: `open_instruct/teacher_entropy.py` + `scripts/eopd/teacher_entropy_diagnostic.py` (OI commit 52de60681; exact entropy, top-16 mass, EOPD's renormalized top-k proxy and its gate agreement, sampled-token teacher rank). First run `01M2RZXXTT0GBCQSJCFKYAA76C`: Qwen3-8B on 256 Qwen3-4B-Base DAPO rollouts (T 1.0, 4096 tokens), output under `deletable_checkpoint/kevinfarhat/eopd/teacher_entropy/`. Second run `01M2S08RST44V0RFTGRFRSS72Y` **done**: verifier-DPPO 9B teacher on 256 Qwen3.5-4B rollouts over our math prompts (796k tokens): mean H 0.24, median 0.003, 11.4 % of tokens above tau 0.8, top-16 mass 0.9994, 0.21 % of student tokens outside the teacher's top-16 (1.6 % among high-entropy tokens), proxy gate agreement 99.87 % (false-negative 1.1 %), 54 % of rollouts hit the 4096-token cap. First run `01M2RZXXTT0GBCQSJCFKYAA76C` **done** (paper pair): mean H 1.26, 31 % above tau, top-16 mass 0.91, 12.4 % of student tokens outside the teacher's top-16, proxy gate agreement 98 %. Both runs recorded under "Step 5 results"; step 5 measurement complete, tau 0.8 retained for the replication. |
| 6 | Implement EOPD in Miles (teacher top-k via SGLang `top_logprobs_num`, top-k-renormalized entropy proxy, student log-probs gathered at teacher indices under TP, gated FKL term, audit + metrics). A/B vs the matched baseline, 2 seeds. Target: +1.8 Avg@8 / +5 Pass@8 at 4B. | Methodology validation. | **Coded 2026-09-18** (Miles commit f78a618a1, off by default; see "Step 6 design"): `eopd_math` (renormalised top-k, entropy-proxy gate, FKL, TP-sharded student log-probs at the teacher ids), `eopd_loss.policy_loss` (upstream policy loss + `alpha * gate * FKL`), hooks requesting `top_logprobs_num = k` and storing the top-k in `train_metadata`, `[distillation] eopd/eopd_alpha/eopd_tau/eopd_top_k`, runtime patch (`metadata` in the train-step keys), `audit.json` `eopd` block, `configs/miles/opd/eopd-eopd-qwen3-tiny.toml`. Local tests: 65 passed incl. a two-rank gloo check that the sharded student log-probs and their gradients match the dense computation. Next: OPD tiny smoke passes → EOPD tiny smoke (exercises the loss on GPUs, checks `train/eopd_*` and the audit) → A/B arms once compute is approved. |
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
