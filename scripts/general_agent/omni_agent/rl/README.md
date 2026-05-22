# Multi-Task RL — example scripts

These scripts implement the three approaches described in [docs/algorithms/multi_task_rl.md](../../docs/algorithms/multi_task_rl.md):

| File | Approach | Runnable today? |
|------|----------|-----------------|
| `joint_homogeneous_3skills.sh` | (1) single run, one skill per step | **No** — needs `MultiSkillDataLoader` (sketch below). |
| `joint_heterogeneous_tool_only.sh` | (2) single run, mixed batches, function-calling only | **Yes** — needs a pre-built unified HF dataset. |
| `joint_heterogeneous_with_sandbox.sh` | (2) + swerl_sandbox (text env) | **Conditional** — needs the `_acquire_and_reset_pools` fix in `open_instruct/vllm_utils.py` (see doc §4). |
| `cascade_stage1_drtulu.sh` | (3) stage 1: vanilla single-skill RL | **Yes**. |
| `cascade_stage2_swerl_with_kl_anchor.sh` | (3) stage 2: anchored to stage 1 via `--load_ref_policy` | **Yes**. |
| `cascade_stageN_template.sh` | (3) generic template | **Yes**. |

Each script comments why specific args are set the way they are.

---

## Prerequisites

1. **Pre-built unified HF dataset(s).** The scripts assume datasets where every row carries the columns described in `docs/algorithms/multi_task_rl.md` §5: `messages` (with skill-specific system prompt baked in), `ground_truth`, `dataset`, `tools`, `env_config`. Reusable converter pattern: see `scripts/data/convert_swe_sft_to_unified_format.py`.
2. **A built Beaker image with all tool deps.** Use `./scripts/train/build_image_and_launch.sh <script>` as usual.
3. **Secrets, for the dr_tulu search tools:** `SERPER_API_KEY`, `S2_API_KEY`, `JINA_API_KEY`, `OPENAI_API_KEY`. For swerl_sandbox: Docker/Podman host env, see `scripts/tmax/4b/qwen35_4b_base_tmax_10k_8_podman_services.sh`.

---

## Sketch: `MultiSkillDataLoader` for approach (1)

`open_instruct/data_loader.py` defines `HFDataLoader` (a thin wrapper around an HF `Dataset` with shuffling and batching). For approach (1) you want a wrapper that holds one `HFDataLoader` per skill and yields one skill's worth of prompts per step.

Minimal sketch (not in repo; would live next to `HFDataLoader`):

```python
class MultiSkillDataLoader:
    """Yields one skill's prompts per step (homogeneous batches).

    Args:
        skill_loaders: dict[skill_name -> HFDataLoader]
        prompts_per_step: how many prompts to draw at once
        schedule: "round_robin" or callable(step_idx) -> skill_name
    """
    def __init__(self, skill_loaders, prompts_per_step, schedule="round_robin"):
        self.skill_loaders = skill_loaders
        self.skill_names = list(skill_loaders.keys())
        self.prompts_per_step = prompts_per_step
        self.schedule = schedule
        self._step = 0
        self._epoch = 0

    def __next__(self):
        if self.schedule == "round_robin":
            skill = self.skill_names[self._step % len(self.skill_names)]
        else:
            skill = self.schedule(self._step)
        self._step += 1
        # Pull `prompts_per_step` examples from the chosen skill's loader.
        examples = [next(self.skill_loaders[skill]) for _ in range(self.prompts_per_step)]
        return examples
```

Then in `grpo_fast.py` near where `iter_dataloader` is constructed, branch on a new `--homogeneous_batches` flag to build a `MultiSkillDataLoader` whose per-skill `HFDataLoader`s wrap subsets of the concatenated train dataset filtered by `dataset_source`.

This is the only required code change for approach (1). Everything downstream (`add_prompt_to_generator`, env dispatch, reward computation) already routes per example.
