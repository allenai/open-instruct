# Multi-Task RL in open-instruct

This document maps out three approaches to training a single policy across multiple skills (e.g. `dr_tulu` search/citation, `swerl_sandbox` terminal coding, AppWorld, generic MCP), explains what the codebase already supports, and lists the concrete changes needed.

The audience is someone who has read [grpo_pipeline_overview.md](grpo_pipeline_overview.md) and [rollout_loop_internals.md](rollout_loop_internals.md) and now wants to extend the pipeline beyond a single skill.

References to existing scripts:
- [scripts/train/dr-tulu/rl_qwen35_4b_drtulu.sh](../../scripts/train/dr-tulu/rl_qwen35_4b_drtulu.sh) — single-task RL on `dr-tulu-rl-data` with `serper_search`, `jina_browse`, `s2_search`, evolving rubrics.
- [scripts/tmax/4b/qwen35_4b_base_tmax_10k_8_podman_services.sh](../../scripts/tmax/4b/qwen35_4b_base_tmax_10k_8_podman_services.sh) — single-task RL on `swerl-tmax-10k` with `swerl_sandbox` (a text env).

Example scripts in [scripts/general_agent/multi_task_rl/](../../scripts/general_agent/multi_task_rl/) show the three approaches.

---

## TL;DR

| Approach | What it is | Already supported? | Required changes |
|----------|------------|-------------------|------------------|
| **(1) Joint, homogeneous batches** | Each step samples a single skill; alternate skills across steps | Yes for tool envs. Needs a skill-aware sampler. | Add a step-level skill sampler, add per-example `env_config` + `dataset` column in each source dataset, and globally configure all tools. |
| **(2) Joint, heterogeneous batches** | A single batch mixes prompts from all skills | Yes for tool envs. **Not** for >1 text env (sandbox/MCP-with-state). | (a) Concatenate datasets with per-example `env_config`/`tools`/`dataset`. (b) Lift the "only one text env" restriction OR keep text envs as `RLEnvironment` subclasses. (c) Optimize `_acquire_and_reset_pools` to acquire only allowed envs. |
| **(3) Sequential cascade (with anti-forgetting)** | RL on skill A → checkpoint → RL on skill B starting from A → ... Optional on-policy distillation from previous checkpoint. | Foundational pieces exist (`--load_ref_policy`, KL-to-ref, checkpoint resumption). | Add a "frozen teacher = last cascade checkpoint" knob; multi-stage launch script. |

Below: detailed mechanism, what to change, and example scripts.

---

## 1. How the current pipeline routes per-example

This is essential context; both joint-batch approaches stand on these primitives.

### 1.1 Dataset rows have *per-example* skill metadata

Every RLVR example is tokenized through `rlvr_tokenize_v1`/`v2`/`v3` in [open_instruct/dataset_transformation.py:1377-1532](../../open_instruct/dataset_transformation.py). The columns that survive transformation and matter at rollout time are:

| Column | Constant | Per-example? | Purpose |
|--------|----------|-------------|---------|
| `messages` | `DEFAULT_SFT_MESSAGES_KEY` | Yes | The conversation. System prompt can be **overridden globally** by `--system_prompt_override_file`. |
| `ground_truth` | `GROUND_TRUTHS_KEY` | Yes | What the verifier compares against. Can be `str` or `list[str]` (multi-verifier). |
| `dataset` | `VERIFIER_SOURCE_KEY` | Yes | **Selects the verifier** at reward time (see §1.3). Can be `str` or `list[str]`. |
| `tools` | `TOOLS_COLUMN_KEY` | Yes | Whitelist of tool *call names* available to this example. Filters the global tool set. |
| `env_config` | `ENV_CONFIG_KEY` | Yes | Per-example overrides for env kwargs (e.g. a specific sandbox image, MCP server, task data). Merged with the run-level base config at rollout time. |

These columns are preserved through caching ([dataset_transformation.py:1770-1774](../../open_instruct/dataset_transformation.py)) and propagated into the `PromptRequest` ([data_loader.py:937-948](../../open_instruct/data_loader.py)) sent to the rollout actor.

The `dataset_mixer_list` concatenates source HF datasets (each becomes its own `DatasetConfig`), but the **same** `transform_fn_args` is applied to all of them ([dataset_transformation.py:2113-2122](../../open_instruct/dataset_transformation.py) and [grpo_fast.py:1262-1282](../../open_instruct/grpo_fast.py)). That means a single `--ground_truths_key`/`--sft_messages_key`/`--system_prompt_override_file` covers every source dataset, **but** the per-row `dataset`, `tools`, and `env_config` columns can already differ row-by-row.

### 1.2 Tools and environments are *partially* per-example

Run-level: `--tools`, `--tool_call_names`, `--tool_configs` ([grpo_fast.py:1310-1364](../../open_instruct/grpo_fast.py)) spawn one `EnvironmentPool` per registered tool. The pool actor classes come from `TOOL_REGISTRY` in [open_instruct/environments/tools/tools.py:651-664](../../open_instruct/environments/tools/tools.py): `python`, `jina_browse`, `s2_search`, `serper_search`, `crawl4ai_browse`, `dr_agent_mcp`, `generic_mcp`, `counter_env`, `guess_number`, `generic_sandbox`, `swerl_sandbox`, `wordle_text_env`.

Per-example:
- The `tools` column whitelists which tool call names this prompt may use. The model only sees those tool schemas in its prompt ([dataset_transformation.py:1462-1480](../../open_instruct/dataset_transformation.py)) and dispatch is gated by `allowed_tools = configured_tools & set(active_tools)` ([vllm_utils.py:1017](../../open_instruct/vllm_utils.py)).
- The `env_config` column merges into the run-level base config via `_merge_env_config` ([data_loader.py:867-891](../../open_instruct/data_loader.py)). The merged `EnvConfig` is shipped with the `PromptRequest` and consumed at `_acquire_and_reset_pools` time to pass `kwargs` to `actor.reset(...)`.

There are **two important caveats** in the current dispatch ([vllm_utils.py:929-987](../../open_instruct/vllm_utils.py)):

1. **`_acquire_and_reset_pools` iterates over the global `configured_tools`, not over `allowed_tools`** ([vllm_utils.py:943](../../open_instruct/vllm_utils.py)). Every rollout acquires one actor from *every* configured pool, even if the prompt only allows a subset. This is fine when tools are cheap (HTTP search) but **wastes a Docker container per rollout** when a heavyweight env (e.g. `swerl_sandbox`) is registered alongside lightweight tools. See §4 for the fix.

2. **Only one *text env* per rollout** ([vllm_utils.py:970-971](../../open_instruct/vllm_utils.py)): `raise ValueError("Only one text environment may be active per rollout, got: ...")`. A text env is one whose `EnvConfigEntry.is_text_env=True` (i.e. it receives the model's full text rather than a parsed tool call — e.g. `swerl_sandbox`, `wordle_text_env`). This is a hard limit: you cannot put both `swerl_sandbox` and another text env in a single rollout. Tool envs (function-calling style) are unrestricted.

### 1.3 Reward selection is *fully* per-example, keyed by `dataset`

[ground_truth_utils.py:1253-1316](../../open_instruct/ground_truth_utils.py) — `apply_verifiable_reward`:

```python
for i, (...ground_truth, dataset, query, rollout_state...) in enumerate(...):
    ground_truth_list = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    dataset_list      = [dataset]      if isinstance(dataset, str)      else dataset
    for gt, ds in zip(ground_truth_list, dataset_list):
        reward_func = reward_fn_mapping.get(ds.lower())
```

`reward_fn_mapping` is built in `build_all_verifiers` ([ground_truth_utils.py:1194-1231](../../open_instruct/ground_truth_utils.py)) and includes every subclass of `VerifierFunction`: `gsm8k`, `math`, `strict_math`, `ifeval`, `flan`, `string_matcher`, `f1`, `puzzle`, `re_search`, `r1_search`, `max_len`, `up_to_max_len`, `code`, `code_stdio`, `passthrough`, `rubric`, plus all `JUDGE_PROMPT_MAP` LLM judges. The `--remap_verifier old=new` flag lets a dataset name like `general_rubric` route to `rubric`.

**Key consequence:** different examples in the same batch can carry different `dataset` strings (or *lists* of dataset strings, in which case multiple verifiers run and their weighted scores sum). Multi-task rewards in one batch are already supported, as long as each row's `dataset` field selects the right verifier.

Evolving rubrics are an exception: they're managed by a single `RubricManager` ([data_loader.py:1428-1431, 1529-1537](../../open_instruct/data_loader.py)) keyed by *query*, and turn on globally via `--apply_evolving_rubric_reward`. To restrict them to a subset of prompts, you'd gate `run_step` on the `dataset` column or short-circuit when the row isn't from a rubric-enabled source.

### 1.4 System prompts are *global*

`--system_prompt_override_file` is read once in [grpo_fast.py:1255-1281](../../open_instruct/grpo_fast.py) and baked into the prompt during preprocessing via `rlvr_tokenize_v*` ([dataset_transformation.py:1389-1393, 1499-1502](../../open_instruct/dataset_transformation.py)). It is **not pluggable per-example** today.

Most multi-task setups need different system prompts per skill (dr_tulu's "research assistant with citations" vs. swerl_sandbox's "bash terminal"). Options:
- **(easy, no code change):** bake the system prompt into each source dataset's `messages[0]` and **omit** `--system_prompt_override_file`. The override only fires when the flag is set ([dataset_transformation.py:1390, 1499](../../open_instruct/dataset_transformation.py)); without it, `messages[0]` is used verbatim.
- **(code change):** add a `system_prompt_override` column to the dataset and respect it in `rlvr_tokenize_v3` instead of the global flag.

The easy option is what the example scripts below assume.

---

## 2. Approach (1): Single joint run, homogeneous batches

> Each training step uses prompts from **one** skill only; skills alternate across steps according to some schedule.

### What it gets you
- Stable per-skill batch statistics (mean reward, advantage normalization) — no cross-skill bleed-through.
- Easier to debug reward shaping (a step's metrics belong to one skill).
- Lets heavyweight envs (swerl_sandbox, AppWorld) be active *only* on their own steps, avoiding the "every rollout acquires every pool" tax.
- Naturally compatible with the "one text env per rollout" restriction — alternating means one text env is active per step.

### Mechanism
1. Build one HF dataset *per skill*, with skill-specific `dataset`, `tools`, `env_config`, and a system prompt baked into `messages`.
2. Train with `--dataset_mixer_list skill_a 1.0 skill_b 1.0 skill_c 1.0`. The concatenated dataset is shuffled once at preprocessing time.
3. **Add a skill-aware sampler** that draws all `num_unique_prompts_rollout` prompts for a step from one skill (round-robin, or weighted by skill loss/reward signal).
4. Globally register every tool/env via `--tools ... --tool_call_names ... --tool_configs ...` so each pool exists; per-example `tools` and `env_config` restrict each step's rollouts to a single skill's subset.

### What needs to change

Concatenation + per-example dispatch already work. The missing piece is **the sampler**.

`HFDataLoader` ([data_loader.py:230-…](../../open_instruct/data_loader.py)) currently shuffles the full concatenated dataset. To get homogeneous batches you need one of:

- **A) Cheap and hacky (no code change):** order the dataset so that contiguous chunks of `num_unique_prompts_rollout` rows are from the same skill (sort by `dataset_source`, then within-skill shuffle each epoch). Then **disable cross-epoch shuffling** of the outer order. This is brittle: a single dropped/active-sampled example breaks alignment.
- **B) Add a `MultiSkillDataLoader`** that holds one inner `HFDataLoader` per skill and yields one skill's worth of prompts per step. This is the right approach.

The data-prep actor (`DataPreparationActor` in [data_loader.py:1352+](../../open_instruct/data_loader.py)) pulls prompts via `next(iter_dataloader)` in `add_prompt_to_generator`; swapping the dataloader is the only required change in the rollout path.

For text envs (swerl_sandbox, AppWorld), this approach also bypasses caveat #2 in §1.2 — at most one text env is active per step.

### Heavyweight envs and pool sizing

If skill A's env is expensive (Docker sandboxes) and skill B's is cheap, you don't want to keep skill A's pool fully warm during skill B's steps. Today the pool size is fixed at construction time. Either:
- Size each pool to the max concurrent rollouts (`num_unique_prompts_rollout * num_samples_per_prompt_rollout`), accept idle Docker containers between skill A steps.
- Or fix the "iterate all pools" bug in `_acquire_and_reset_pools` first (§4) so unused pools aren't even touched during skill B's steps.

### Example script

[scripts/general_agent/multi_task_rl/joint_homogeneous_3skills.sh](../../scripts/general_agent/multi_task_rl/joint_homogeneous_3skills.sh) — Sketch using `dr-tulu-rl-data`, `swerl-tmax-10k`, and a generic-MCP skill. **Requires** the `MultiSkillDataLoader` (not in repo yet — see [scripts/general_agent/multi_task_rl/README.md](../../scripts/general_agent/multi_task_rl/README.md) for the diff sketch).

---

## 3. Approach (2): Single joint run, heterogeneous batches

> Every batch contains a mix of prompts from all skills.

### What it gets you
- Maximally on-policy across skills; the trainer sees a representative reward landscape every step.
- One global advantage normalization (good or bad depending on reward scales).
- Forces the model to maintain all skills simultaneously — strongest anti-forgetting signal.

### Mechanism

Same dataset construction as approach (1) — concatenate per-skill HF datasets with per-example `dataset`/`tools`/`env_config`. **Skip** the custom sampler — let the default `HFDataLoader` shuffle the concatenated dataset, and accept whatever skill mix you get per batch.

### What needs to change

**Tool/MCP-only multi-task is supported today.** Datasets like `dr-tulu` (search tools) + AppWorld-via-MCP + any function-calling skill can be mixed without code changes — register all tools globally, set the per-example `tools` column, set per-example `dataset` for reward routing.

**Heterogeneous batches with multiple text envs (swerl_sandbox + wordle, or two different sandbox tasks) do NOT work today**. The blockers:

1. `_acquire_and_reset_pools` iterates *all* configured pools per rollout ([vllm_utils.py:943](../../open_instruct/vllm_utils.py)). With a Docker-backed sandbox in the run, **every** prompt acquires a Docker container even if it only needs `serper_search`. This is a performance problem, not a correctness one; for small mixes it might be tolerable.
2. The hard cap on text envs per rollout ([vllm_utils.py:970-971](../../open_instruct/vllm_utils.py)) prevents having both `swerl_sandbox` (text env) and `wordle_text_env` (text env) registered together at all.

**The fix** (small surgical change in `_acquire_and_reset_pools`):

```python
# Today
for pool_name in sorted(configured_tools):
    ...
    if entry.is_text_env:
        text_env_names.append(pool_name)
```

Change to: iterate over `env_config.env_configs.keys()` (the per-example active set), or — strictly weaker but still correct — iterate `allowed_tools` plus any pool that the per-example `env_config` explicitly names. After this fix:
- Multiple text envs *can* coexist at run level; only one is active per rollout (per the env_config + dataset's tools column).
- Heavyweight pools are only acquired when needed.

`PoolSetup.text_env_names` already supports this — `active_env_names` and `text_env_names` are computed from the per-rollout set, not the global one.

The check `if len(text_env_names) > 1` can stay — it's still a correct invariant for a single rollout. The bug is that today it triggers from global configuration, not from per-rollout state.

### Reward-scale considerations

Different verifiers return different magnitudes (`code` and `code_stdio` return `[0, 1]`; `passthrough` returns 0 by design; `rubric` returns a weighted-average score in `[0, 1]`). With one global `--verification_reward N`, all scores get multiplied by `N`. For heterogeneous batches you typically want per-skill scaling — encode that into each `VerifierFunction.weight` (already supported, set at construction time) or use the multi-verifier `dataset=[...]` list form so multiple weighted verifiers fire on one prompt.

`--advantage_normalization_type centered` (used by tmax scripts) normalizes within the rollout group (same prompt, different samples). Since one prompt belongs to exactly one skill, group normalization stays per-skill even in heterogeneous batches — that's the right default. `standard` (mean / std) would mix skills together via std and is usually worse here.

### Example script

[scripts/general_agent/multi_task_rl/joint_heterogeneous_tool_only.sh](../../scripts/general_agent/multi_task_rl/joint_heterogeneous_tool_only.sh) — Today-runnable: mixes dr-tulu (tool calls) + generic MCP + a math dataset, no text envs. Requires the user to have pre-built a unified HF dataset (see §5).

[scripts/general_agent/multi_task_rl/joint_heterogeneous_with_sandbox.sh](../../scripts/general_agent/multi_task_rl/joint_heterogeneous_with_sandbox.sh) — Same idea but adds `swerl_sandbox` (text env). **Requires the `_acquire_and_reset_pools` fix** described above.

---

## 4. Approach (3): Sequential cascade with anti-forgetting

> Train RL on skill A to convergence → checkpoint → start RL on skill B from that checkpoint → ... Optionally add KL/distillation pressure toward the previous-stage checkpoint to limit forgetting.

### What it gets you
- Decouples per-skill training dynamics — easy reward shaping, easy debugging.
- Lets each stage use the optimal `--tools` set, `--system_prompt_override_file`, learning rate, response length, batch size, etc.
- The "checkpoint A as anti-forgetting anchor for stage B" is the OPC ("On-Policy Compress") / "self-distillation" trick — preserves earlier-skill behavior at the cost of plasticity on the new skill.

### Mechanism

Stage *k*: run RL on skill *k*'s dataset, starting from stage *k-1*'s final checkpoint, with optional KL-to-anchor:
- Set `--model_name_or_path` = path/name of stage *k-1*'s final HF checkpoint.
- Set `--load_ref_policy True --beta <kl_coeff>` to use the *reference* policy (a frozen copy at run start) as anti-forgetting anchor. With `--load_ref_policy False` (the default) and `--beta 0`, there is no anchor.

What `--load_ref_policy True` does: a separate `PreTrainedModel` is loaded in `PolicyTrainerRayProcess.__init__` ([grpo_fast.py:389-410](../../open_instruct/grpo_fast.py)) onto each learner and held frozen. Loss adds `beta * KL(policy || ref)` ([grpo_utils.py:720-733](../../open_instruct/grpo_utils.py)). The check in [grpo_utils.py:300-303](../../open_instruct/grpo_utils.py) enforces `beta == 0 OR load_ref_policy=True`.

There's even a soft-update knob: `--ref_policy_update_freq N` ([grpo_fast.py:1749-1761](../../open_instruct/grpo_fast.py), `update_ref_policy` at line 552) periodically EMAs the reference toward the current policy. For cascade-with-distillation you typically want this **off** (keep the anchor frozen at stage start).

### What needs to change

For the **basic** cascade (no anti-forgetting):
- **Nothing in the code.** Multi-stage just means launching multiple jobs sequentially, each pointing `--model_name_or_path` at the previous stage's `save_freq` checkpoint.
- Existing flags suffice: `--save_freq`, `--checkpoint_state_freq`, `--keep_last_n_checkpoints -1`, `--output_dir`.

For **anti-forgetting via KL to last-stage checkpoint**:
- Also nothing in the code. `--load_ref_policy True --beta <small>` already takes the *initial model* (= last-stage checkpoint) as ref. `--ref_policy_update_freq null` (default) keeps it frozen.
- **Cost:** an extra copy of the model in memory per learner, plus an extra forward pass. ZeRO-3 sharding ([grpo_fast.py:389-393](../../open_instruct/grpo_fast.py): `get_eval_ds_config(...)`) is used for the ref policy.

For **on-policy distillation** (richer signal than KL — match the previous teacher's full log-distribution over chosen rollouts, not just KL between current and frozen policy):
- **Not currently in `grpo_fast.py`**. The closest existing pieces are:
  - `open_instruct/distillkit/` (separate module — not wired into grpo_fast).
  - `open_instruct/sample_logits_vllm.py` (offline teacher-logit dumping).
- The minimal addition would be: in the loss computation ([grpo_utils.py:720-735](../../open_instruct/grpo_utils.py)), add `+ distill_coeff * KL(policy || teacher)` where `teacher_logprobs` come from a frozen second model loaded the same way as `ref_policy`. That is, **`ref_policy` already gives you the anchor; "on-policy distillation" just means using a higher `beta` and possibly a richer per-token target**.
- A pragmatic alternative: do **SFT-style distillation** between RL stages (use `open_instruct/distillkit/` on the *new* skill's data with the *previous* checkpoint as teacher) as a separate stage.

### Example scripts

- [scripts/general_agent/multi_task_rl/cascade_stage1_drtulu.sh](../../scripts/general_agent/multi_task_rl/cascade_stage1_drtulu.sh) — stage 1, train on dr_tulu, save checkpoint.
- [scripts/general_agent/multi_task_rl/cascade_stage2_swerl_with_kl_anchor.sh](../../scripts/general_agent/multi_task_rl/cascade_stage2_swerl_with_kl_anchor.sh) — stage 2, swerl-tmax-10k, anchored to stage 1's checkpoint via `--load_ref_policy True --beta 0.005`.
- [scripts/general_agent/multi_task_rl/cascade_stageN_template.sh](../../scripts/general_agent/multi_task_rl/cascade_stageN_template.sh) — generic template for further stages.

---

## 5. Building a multi-task RL dataset

This section is independent of the three approaches above — they all need one or more HF datasets with the right per-row columns. Required columns per row:

| Column | Type | Per-row content |
|--------|------|----------------|
| `messages` | `list[dict]` | `[{"role": "system", "content": SKILL_SYSTEM_PROMPT}, {"role": "user", "content": PROMPT}]`. Bake the skill-specific system prompt in here. |
| `ground_truth` | `str` or `list[str]` (or any JSON for `rubric`) | What the verifier checks. |
| `dataset` | `str` or `list[str]` | Verifier name. Examples: `"math"`, `"code"`, `"code_stdio"`, `"rubric"`, `"passthrough"` (env-reward-only). Multi-list form runs multiple verifiers and weights/sums them. |
| `tools` | `list[str]` | Tool *call names* available to this row. Empty list = no tools. |
| `env_config` | `dict` | Per-row env kwargs. Two valid shapes (both auto-normalized by `_normalize_env_config_column` at [dataset_transformation.py:958-989](../../open_instruct/dataset_transformation.py)): `{"env_configs": [{"env_name": "swerl_sandbox", "image": "...", ...}], "max_steps": 50}` or the flat shorthand `{"env_name": "swerl_sandbox", "image": "..."}`. |

For text envs (`swerl_sandbox`, `wordle_text_env`, `generic_sandbox`), the `tools` column should contain the env's `config_name` (e.g. `"swerl_sandbox"`) — the dispatch loop adds it to `allowed_tools` when `is_text_env=True` ([vllm_utils.py:966-968](../../open_instruct/vllm_utils.py)).

A reusable converter pattern (see [scripts/data/convert_swe_sft_to_unified_format.py](../../scripts/data/convert_swe_sft_to_unified_format.py) for inspiration):

```python
def convert_drtulu(row):
    return {
        "messages": [
            {"role": "system", "content": DRTULU_SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ],
        "ground_truth": row["ground_truth"],
        "dataset": "general_rubric",         # routed to RubricVerifier via --remap_verifier
        "tools": ["google_search", "browse_webpage", "snippet_search"],
        "env_config": {"env_configs": []},   # no per-row override; use run-level configs
    }

def convert_swerl(row):
    return {
        "messages": [
            {"role": "system", "content": SWERL_SYSTEM_PROMPT},
            {"role": "user", "content": row["task_prompt"]},
        ],
        "ground_truth": row["task_id"],        # the sandbox env handles verification internally
        "dataset": "passthrough",              # don't double-reward; sandbox produces step rewards
        "tools": ["swerl_sandbox"],            # text env added to allowed_tools at dispatch
        "env_config": {
            "env_configs": [
                {"env_name": "swerl_sandbox", "task_id": row["task_id"]}
            ],
            "max_steps": 50,
        },
    }
```

Push each to HF separately, then list them all in `--dataset_mixer_list a 1.0 b 1.0 c 1.0`. The mixer concatenates them and the columns above route everything downstream.

---

## 6. Decision guide

| You want… | Use approach |
|-----------|--------------|
| Lowest-effort first run, only function-calling skills | **(2)** heterogeneous, no code changes |
| Mix function-calling with a sandbox text env | **(2)** with the `_acquire_and_reset_pools` fix; OR **(1)** to side-step it |
| Mix two sandbox / two text envs | **(1)** with skill-aware sampler (hard limit until §3 fix) |
| Skills with very different reward scales / convergence rates | **(1)** so per-skill steps don't fight each other |
| Strong anti-forgetting needed; OK to spend more wall-clock | **(3)** with `--load_ref_policy True --beta ≥ 0.005` |
| Brand-new skill the model can't yet do at all | **(3)** stage on the new skill alone (no anchor), then **(1)** or **(2)** with all skills |

---

## 7. Known limitations and follow-ups

1. **`_acquire_and_reset_pools` over-acquires** ([vllm_utils.py:943](../../open_instruct/vllm_utils.py)) — every rollout acquires from every globally configured pool. Real fix: iterate per-rollout `allowed_tools ∪ env_config.env_configs.keys()`. **Required** for approach (2) with heavyweight envs in the mix.
2. **One text env per rollout** ([vllm_utils.py:970-971](../../open_instruct/vllm_utils.py)) — by design, a model can interact with only one text env at a time. Approach (1) sidesteps this by alternating skills across steps.
3. **System prompt is global** — bake into `messages` per-row to get per-skill prompts. A `system_prompt_override` *column* would be cleaner; small change in `rlvr_tokenize_v3`.
4. **No built-in skill-aware sampler** — the new `MultiSkillDataLoader` for approach (1) is sketched in [scripts/general_agent/multi_task_rl/README.md](../../scripts/general_agent/multi_task_rl/README.md) but not yet implemented.
5. **Evolving rubric is global** — `--apply_evolving_rubric_reward` applies to all rollouts. To restrict to one skill, gate `RubricManager.run_step` on per-row `dataset`.
6. **No on-policy distillation in grpo_fast.py** — the KL-to-ref hook (`--load_ref_policy True`) gives you the anchor; a true cross-entropy-to-teacher term would need a small loss-function patch ([grpo_utils.py:720-735](../../open_instruct/grpo_utils.py)).

---

## 8. Related docs

- [grpo_pipeline_overview.md](grpo_pipeline_overview.md) — plain-language tour of the full GRPO pipeline.
- [grpo_fast_internals.md](grpo_fast_internals.md) — deep-dive into Ray actors, async pipeline, weight sync, loss/KL/advantage.
- [rollout_loop_internals.md](rollout_loop_internals.md) — what the rollout actor actually does, token by token.
- [rl_with_environments.md](rl_with_environments.md) — how to add a new environment / tool.
- [tool_training.md](tool_training.md) — broader tool-training reference.
- [tmax_4b_script_reference.md](tmax_4b_script_reference.md) — annotated swerl_sandbox script reference.
