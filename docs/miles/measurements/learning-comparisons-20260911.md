# Learning comparisons and performance analysis

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

The September11 extension has two perspectives: the lightly SFT-trained model
matched to its historical olmo-miles experiment, and a longer matched Core/Megatron
pair starting from Abhishek's Dolci-Think SFT step23607. GPU work uses urgent
priority, Holmes and open-instruct-dev. CPU jobs that read WEKA use Saturn.

## Abhishek checkpoint: fresh 500-update pair

The original100 jobs deliberately saved no native optimizer checkpoint, so the
extension starts both backends from the same SFT weights. It is not a continuation
of either final100 policy. The new Core image includes the scoring SwiGLU rounding
fix at OLMo-core290d2ca; this difference from Core100 is explicit.

- Campaign: `gsm8k-core-megatron-20260911-500-v1`.
- Root: `/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260911-abhishek-500-v1`.
- Preserve the exact original400 training prompts,128 official-test questions,
  template/token IDs, verifier and prepared-file hashes. Five ordered passes,
  four prompts and four completions per update:8,000 training responses total.
  Repeated exposure is not2,000 unique training prompts.
- Keep train temperature1, greedy held-out evaluation every20 updates including
  before training, response4096/context6144, LR1e-6 constant, original optimizer,
  GRPO normalization off and clipping0.2/0.28.
- Keep2 expert-parallel trainer GPUs plus one serving GPU, decode graphs,
  concurrency4, KV32768, native-router replay off and1GiB weight-sync buckets.
- Save every100 updates. Core currently retains allfive native saves; Megatron's
  existing retention keeps the latest completed save. Include disk and save-time
  differences in the report. Do not silently delete old artifacts.
- Core minimum runtime8h, timeout18h. Beaker rejected the first Megatron
  submission with12h minimum before creating an experiment: the service maximum
  is8h. Both arms therefore use8h minimum; Megatron's18h budget requires external
  monitoring because its current public launch schema has no deadline field.
- Full-sized native save and fresh-process restore were exact on both ranks before launch. Independent fresh-start trajectories differed, so the broader bitwise reproducibility gate remains failed; see [durability evidence](core-durable-full-20260911.json).

`extended_gsm8k.py` verifies the old preparation before linking it into a separate
campaign. The parameterized runner and auditor preserve100-update defaults.
Audits check ordered repetition, complete prompt groups, token IDs, independently
recomputed rewards, expected policy versions, optimizer/publication sequences,
and every scheduled held-out evaluation, including the final500 endpoint.

## Light SFT historical match

Use the latent-KDA midtrain63802 model after1,000 SFT updates, native iteration999
under `rl-sft1000/sft-v1/checkpoints`. Do not substitute the unrelated31-layer
`s002` SFT1000 checkpoint. The completed200-update historical experiment
`01M12Y23YZS5ZJWBK45CKQJ7DP` is the useful learning anchor; the earlier100-update
precursor has invalid empty evaluation predictions and cannot establish quality.

Preserve separate historical evaluation tracks: in-loop128 questions used sampled
temperature1 chat prompts; offline full-test1,319 used greedy raw Question/Answer
prompts and512-token responses. Offline correct counts230→274 are not the same
measurement as in-loop26→77 out of128. Recover and hash exact source artifacts
before launching the Core reproduction. Source recovery and preparation passed; the historical settings and detected precision/topology differences are recorded in [the light-SFT protocol](light-sft1000-gsm8k.md).

## Required final analysis

1. Compare effective arguments, model/tokenizer/template hashes, data membership
   and ordering, reward extraction, batch sizes, optimizer/LR schedule, numerical
   precision, loss normalization/masking, auxiliary objectives, routing/replay,
   attention/MoE kernels, activation checkpointing, topology, serving settings,
   generation/evaluation settings, compilation/cache state, saving and runtime
   source pins. Distinguish deliberate differences, defaults and unknowns.
2. Plot held-out curves, paired question transitions, response lengths/caps,
   mixed-reward group fraction and training rewards. Report final endpoints and
   full trajectories; preserve unsuccessful attempts and scoring-fix provenance.
3. Compare matched warm update windows and token workloads. Separate queue delay,
   allocation/startup, checkpoint loading, compilation, initial sync/evaluation,
   generation and verification, batch ingress/preflight/scoring, forward/backward
   and optimizer, publication/export/transport, recurring evaluation and saving.
   Retain wall time and allocated GPU-hours; label overlap and instrumentation
   differences. Never obtain an orchestration estimate by subtracting unrelated
   phase averages. Use paired collection timestamps and scoped profiling where
   existing timers cannot split phases.
4. Keep historical timing settings distinct from current matched runs. The older
  512-token workload is not a throughput control for4096-token Dolci-Think runs.
   Report detected configuration differences before attributing performance to
   a backend. The small held-out subset and single seed bound the conclusions.

## Submitted runs and completed reference

| Run | Beaker | Source | W&B |
|---|---|---|---|
| Abhishek Core500 |[01M279ZFM6RBC223RJJ6QHN9MP](https://beaker.org/ex/01M279ZFM6RBC223RJJ6QHN9MP) |OI1ceaf0d35; Core290d2ca |[on25j3v2](https://wandb.ai/ai2-llm/olmo-rl-comparison/runs/on25j3v2) |
| Abhishek Megatron500 |[01M278B5E9HME181B04HT6391P](https://beaker.org/ex/01M278B5E9HME181B04HT6391P) |olmo-miles aa114a1; explicit8h override |[bhyj1mec](https://wandb.ai/ai2-llm/olmo-rl-comparison/runs/bhyj1mec) |
| Light SFT Core200 |[01M2794XP15PHQSWN6M499NQYS](https://beaker.org/ex/01M2794XP15PHQSWN6M499NQYS) |OI290f8abb1; Core290d2ca |[jb8bzz71](https://wandb.ai/ai2-llm/olmo-rl-comparison/runs/jb8bzz71) |
| Original Core100 scorer profile |[01M279F38FPRHMTYQQYWQ17GSN](https://beaker.org/ex/01M279F38FPRHMTYQQYWQ17GSN) |Worker1ceaf0d35; frozen image01M26N80T0V9PREQTS87J849P8 |No training run |

The original100-update pair is complete and independently audited: Core97→93/128,
Megatron96→107/128. [Final learning/performance report](gsm8k-results-20260911.md).
These are completed results; the three new learning allocations above do not yet
have final results. Status snapshot:2026-09-11 approximately04:21UTC. Core500 passed its configuration, storage, and initial-publication checks; its initial greedy evaluation scored99/128 with20 truncated responses. Training generation has begun. See [startup evidence](core500-startup-20260911.json). This is the new run’s starting point, not a learning gain over the old Core100 initial97/128.
