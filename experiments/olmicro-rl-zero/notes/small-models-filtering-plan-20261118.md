# Difficulty filtering + RL plan for Olmicro 275M / 450M / 810M / 1.4B (drafted 2026-11-18 8:58 PM PST; nothing launched)

Same protocol as the 2.7B long-context run, per size.

## Inputs
| size | HF revision (stage3 final) | layers | attn heads (q/kv) | GDN heads | full-attn layers |
|---|---|---|---|---|---|
| 275m | stage3-step0047684 | 10 | 8/8 | 8 | 2 |
| 450m | stage3-step0047684 | 15 | 8/8 | 8 | 3 |
| 810m | stage3-step0023842 | 15 | 16/8 | 16 | 3 |
| 1.4b | stage3-step0023842 | 20 | 16/8 | 16 | 4 |
| 2.7b (done) | stage3-step0023842 | 30 | 16/8 | 16 | 6 |

All sizes: head_dim 128 (attention and GDN key), GDN value 256, 64k context, same vocab. The kernel shapes are the same as 2.7B, so FA4, GDN and the FLA config cache carry over. Download each size's HF model plus its Core config (HF bucket, stage3 config.json) and verify SHA256SUMS.

## Steps
1. Qualify (one node, ~1 h): per size, a short native smoke run: SGLang native load, graph-vs-eager greedy parity, Core-vs-HF step-0 logprob parity, two RL updates, and no conv/l2norm autotune.
2. Pass-rate sweep (`tools/format_probe.py`, as for 2.7B):
   - All 13,251 prompts of dolci-math-base-r1-v1 (R1 template), 16 samples, temperature 1.0, 32,768 new tokens, stop "\nUser:".
   - One node per size, 32 resumable shards, urgent, minRuntime = expected + buffer.
   - The 2.7B took 2 h 20 min. Estimate 1–2 h per smaller size (faster decode; possibly longer rambling); two at a time on 2 nodes → about 3–4 h for all four.
3. Score on CPU (MathVerifier, truncated = 0) and analyse lengths (`tools/length_tail.py`).
4. Filter (`data-prep/filter_by_pass_rate.py`): keep 1–14 correct of 16, rescored exactly at the chosen cap with `--cap`.
5. RL: `make_v2_refresh_config.py` generalized to `--size`.
   - Refresh publication, staleness 12, 512 in-flight samples, 400 updates × 32 prompts × 8 samples, lr 1e-6.
   - Evals every 50 updates (4 samples); `checkpoint_keep_every = 50`; FLA cache; `engine_drain_timeout = 1800`.
   - Proposed 1 node each: 2 trainer + 6 inference GPUs for 275M/450M/810M, 4 + 4 for 1.4B.

## Decision points
- **Kept-set size.** The 2.7B kept 4,900 prompts (37% solved at least once); smaller models will keep fewer.
  - 400 updates draw 12,800 prompts, so a 3,200-prompt set is 4 epochs.
  - Proposal: run as is down to ~1,500 prompts (multi-epoch is fine for a demo). Below that (likely 275M), choose between fewer updates or an easier prompt source.
- **Cap.** The 32k sweep lets us rescore 8k/16k exactly.
  - Default to 32k to match the 2.7B main setting.
  - Switch a size to 16k if its sweep shows heavy looping to the cap (2.7B: 0.3% at 32k).
  - Check how many prompts are solvable only above 8k (2.7B: 547 of 4,900).
- **Order.** 1.4B and 810M first (closest to 2.7B, most likely to keep enough prompts), then 450M and 275M.
