# Experimental Megatron verifier GRPO

This route runs **teacher-free verifier GRPO with a Megatron learner and SGLang
rollout engines** through `python -m open_instruct.miles`, optionally fully async
and optionally with Open Instruct's DPPO loss. It is a sibling of
[Megatron OPD](opd.md) and reuses its runtime: the same pinned MILES driver,
conversion profiles, async producer, measured queue and Beaker launcher. Core
GRPO ([the GRPO guide](grpo.md)) stays the default for `training.algorithm = "grpo"`.

A run file selects this route with `training.algorithm = "grpo"` **and**
`trainer.backend = "megatron"`; `configuration/specs.py` then builds
`MegatronGRPORunSpec` instead of the Core `RunSpec`. The application modules live
in `open_instruct.miles.megatron_grpo`.

## Provenance and qualification

- Ported from the verifier-RL line tagged `runs/miles-qwen35-2b-rl`
  (`74948cf19`, `claude/miles-keep-inflight-requests`): its Qwen3.5-2B recipe is
  Open Instruct's verifier-RL teacher run (DPPO-TV 0.1 against rollout-engine
  log-probabilities, token-mean loss, async with optimizer-step age at most 5).
- Runtime sources are the commits in `runtime/miles/runtime.lock.json`, shared
  with OPD. The old line ran an older MILES with a patch; its async producer,
  buffer and publication now come from the pinned upstream implementations.

CPU validation does not establish GPU support, objective parity, throughput or
learning quality on this runtime. A short mechanics run must pass the checks
below before longer comparisons.

## Run file

The schema matches the tagged run files. Beyond the shared OPD sections
(`training`, `trainer`, `inference`, `optimizer`, `output`, `tracking`, `data`,
`launch`, `[miles]`):

- `model.source` is a prepared local HF directory; `model.architecture` is
  `qwen3.5-2B` (the exercised recipe) or `qwen3-1.7B`. An optional inline
  `[model.native_checkpoint]` table reuses an audited initial TP2/PP1 release
  (see below).
- `[objective]` owns the loss: `policy_loss = "ppo"` (MILES' clipped surrogate
  with `eps_clip`, `eps_clip_high`, `eps_clip_c`) or `"dppo"` (Open Instruct's
  DPPO trust-region mask, `dppo_divergence_type` `tv`/`kl` and
  `dppo_threshold`), `use_rollout_logprobs` (required by DPPO and by async; it
  requires `inference.top_p = 1`) and `std_normalization`.
- `[reward]` holds the registered-verifier gates of the Core backend:
  `zero_truncated` and `final_answer_only`, both off by default, so the whole
  response is graded whether or not it was truncated (the tagged runs' behavior).
- `data.prompt_data`, `data.eval_prompt_data` name/path pairs and the trusted
  `data.reward_config` registry are required;
  `scripts/miles/prepare_qwen35_math_prompts.py` writes them.

The contract owns one optimizer update and one weight publication per rollout,
within-prompt centered verifier rewards, no reference KL or entropy bonus, Adam
settings, PP/CP/EP of 1 and one node of at most eight GPUs
(`trainer.gpus + inference.gpus = launch.gpus_per_replica`). `[miles]` passes
other pinned native options through; options the schema owns, teacher/OPD
options, colocation, offloading and partial rollout are rejected.
`training.keep_checkpoints` must stay 0.

## Rewards and advantages

MILES calls `megatron_grpo.grpo_hooks.reward` (`--custom-rm-path`), which scores
through `rewards.rewards.registered_reward`: the same registry,
`metadata.verifiers`, weighted sum and math-timeout policy as the Core backend.
`grpo_hooks.post_process` (`--custom-reward-post-process-path`) records every
training response in `verifier-rewards.jsonl`, checks complete fixed-fanout
groups with identical tokenized prompts, and mean-centers rewards within each
group (divided by the sample standard deviation plus `1e-6` with
`objective.std_normalization`). MILES' GRPO estimator broadcasts the centered
value over the response tokens. Evaluation uses the reward hook without the
post-process.

## Async

`miles.fully_async = true` with `miles.max_weight_staleness` runs the native
async driver with `grpo_async.VerifierAsyncRollout`: the async OPD producer,
restart ledger and measured queue, with the verifier as the evaluation reward.
It requires `objective.use_rollout_logprobs = true`, rejects
`pause_generation_mode = "abort"`, and checks every consumed sample's age in
optimizer steps against the budget.

### In-flight requests at publication

The pinned MILES never aborts in-flight generation when it publishes new
weights: responses continue across the publication, and each records the
version that produced each token span. A response can therefore mix policy
versions; its age is measured from the **oldest** version. The tagged runs
enabled this behavior with `keep_inflight_requests_on_weight_update`, which is
not a pinned option; the run file rejects it. `plan` warns about mixed-version
responses for every async run.

## Workflow and checks

`python -m open_instruct.miles train` (inside the image, through `run`):

1. prepares the HF model and data under `output.assets`, then stops for
   `training.phase = "prepare"`;
2. checks the HF `model_type` against the architecture, runs the image tests
   (`open_instruct/test_miles_megatron_grpo.py`,
   `tests/miles/test_megatron_grpo_runtime.py`, plus attention, DPPO and async
   tests when selected);
3. converts the learner on the trainer's TP-only mesh with a bounded,
   process-group-killing conversion, after checking that the pinned converter's
   pipeline inference honors `CONVERT_KEEP_PP1`;
4. parses the native arguments in a preflight that asserts the selected
   objective (`fully_async`, `use_rollout_logprobs`, `loss_type`) reached MILES;
5. trains, then audits finite optimizer metrics, rescoring of every response
   and the trainer's advantages in `debug/train_data` against the centered
   rewards, and completes the HF export. A rollout whose groups all have equal
   rewards is recorded as `zero_signal` rather than failed: its advantages and
   gradient norm are legitimately zero;
6. reloads the export in a fresh SGLang process and probes a training prompt.

A previously completed, independently load-audited **initial TP2/PP1 native
release** can be supplied as an inline `[model.native_checkpoint]` provenance
table (HF source/revision/config hash, architecture fingerprint, TP/PP, native
converter hash, release metadata hash, exact five-file size/mtime inventory,
save/load Beaker identities, UTC load audit). The runtime uses it read-only
instead of conversion and checks its inventory again after training. Releases
converted on an older image fail the converter-hash binding by design.

## Local and runtime checks

`open_instruct/test_miles_megatron_grpo.py` and `open_instruct/test_miles_dppo.py`
run on CPU. `tests/miles/test_megatron_grpo_runtime.py` and
`tests/miles/test_opd_async.py` need the MILES runtime image.

Copy `tests/miles/fixtures/megatron_grpo/qwen35-2b-verifier-dppo-async-short.toml`
(the tagged recipe cut to 10 updates) into ignored `runs/` before changing it,
run `plan` and `validate`, and launch through the committed-image wrapper
described in [launching](launching.md).
