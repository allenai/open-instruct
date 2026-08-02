# Patches to upstream open-instruct

Everything added by this fork lives in two new directories:

- `open_instruct/scored_rewards/` — the generic score-based reward layer
- `projects/` — one project's specifics, imported by nothing in `open_instruct/`

Upstream files are touched in **two places, 28 lines**, kept deliberately small
so rebasing onto `allenai/open-instruct` stays a non-event. With none of the new
flags set, behaviour is byte-identical to upstream.

```
open_instruct/data_loader.py | 18 ++++++++++++++++++
open_instruct/grpo_fast.py   | 23 ++++++++++-------------
```

---

## 1. `open_instruct/data_loader.py` — six CLI flags

Six optional fields on `StreamingDataLoaderConfig`, all defaulting to off:
`reward_plugins`, `group_scorer`, `group_reward_mode`, `group_reward_scale`,
`group_scorer_strict`, `score_verifiers`.

They go on `StreamingDataLoaderConfig` rather than `Args` because that is the
config already threaded into reward construction, and because `ArgumentParserPlus`
derives the CLI from these dataclasses — new fields become new flags with no
parser changes.

**Conflict risk on rebase: nil.** Added fields, changed none.

## 2. `open_instruct/grpo_fast.py` — one call swapped, one call added

**(a)** The literal `RewardConfig(...)` construction in `main` becomes
`make_reward_config(args, streaming_config, tools_config)`. That function
contains the identical constructor call and returns it unchanged unless
`--group_scorer` or `--score_verifiers` is set, in which case it returns a
`GroupRewardConfig` subclass.

A subclass rather than an edit to `ground_truth_utils.RewardConfig` because
`RewardConfig` is shared by every trainer in the repo, and this behaviour is
GRPO-specific: `GroupScorer` needs all G samples of a prompt, which only the
grouped rollout path has.

**Conflict risk on rebase: low but real.** If upstream adds a field to
`RewardConfig`, the constructor inside `make_reward_config` needs the same field.
The symptom is an obvious `TypeError` at startup, not silent drift.

**(b)** `load_reward_plugins(streaming_config.reward_plugins)` as the first
statement in `main`.

It has to be first. Plugins register environments as well as scorers, and
`initialize_tools_and_envs` reads `TOOL_REGISTRY` further down — a plugin
imported any later would register into a registry that had already been read.

**Conflict risk on rebase: nil.** One line at the top of a function.

---

## What was deliberately *not* patched

**`vllm_utils.compute_rewards` was left alone.** It already loops per prompt with
that prompt's whole group in `result.responses`, so the group is reachable from
inside `RewardConfig` without touching the caller. The group scorer runs once per
prompt from `GroupRewardConfig` and caches its result for the individual sample
calls that follow.

**No LoRA.** `use_peft` is declared in `model_utils.py` and referenced nowhere in
`grpo_fast.py`; wiring PEFT through DeepSpeed and the vLLM weight sync is a real
change to the training loop, not a patch, and it is out of scope here. The
consequence — a 3B policy will not fit on one 80GB card beside vLLM — is
documented in `projects/tutor/README.md`.

**No changes to the advantage computation.** `normalize_then_sum` in the reward
produces zero-mean-within-group scores, so upstream's default
`advantage_normalization_type=centered` is already the right arithmetic.

---

## Rebasing

```bash
git remote add upstream https://github.com/allenai/open-instruct.git
git fetch upstream && git rebase upstream/main
python -m unittest open_instruct.scored_rewards.test_scored_rewards projects.tutor.test_tutor
```

The tests need no GPU, no ray, no vLLM and no network, so this is a fast check
that the seams still hold.
