# Qwen3-4B-Base Deepscaler benchmark

These MILES recipes reproduce the Qwen3-4B-Base Deepscaler baseline settings.
They are research configurations, separate from the maintained MILES starters.
The baseline uses ordinary retry handling, without NGU.

Prepare the training and AIME 2025 evaluation data first:

```bash
uv run python scripts/miles/prepare_deepscaler.py /weka/oe-adapt-default/YOUR_USERNAME/open-instruct/data/deepscaler-10k-base-01 --tokenizer /weka/oe-adapt-default/YOUR_USERNAME/checkpoints/Qwen3-4B-Base
```

The default training dataset is
`mnoukhov/deepscaler-10k-qwen3-4b-base-32samples-quartiles`. The helper renders
`qwen_instruct_user_boxed_math` prompts, excludes reference assistant answers,
and drops prompts longer than 2,048 tokens. It writes `train.jsonl`,
`eval-aime_2025.jsonl`, and `verifiers.json`. Use `--train-dataset` to select
another dataset with the same `messages` and `ground_truth` schema.

| Recipe | Prompts per rollout | Samples per prompt | Global batch | Publication |
| --- | ---: | ---: | ---: | --- |
| `qwen3_4b_base_deepscaler_baseline.toml` | 8 | 16 | 128 | refresh |
| `qwen3_4b_base_deepscaler_baseline_k32n16.toml` | 16 | 32 | 512 | barrier |
| `smoke_saturn_fa2.toml` | 16 | 32 | 512 | barrier |

Both baseline recipes use four trainer and four inference GPUs, learning rate
1e-6, an 8,192-token response limit, and a 10,240-token context. Relative to the
original Open Instruct baseline, these recipes use MILES-native rewards of 1/0
instead of 10/0 and gradient clipping at 1 instead of 5. They are not claims of
identical learning curves. The Saturn recipe runs only five rollouts and checks
hardware/runtime compatibility rather than benchmark quality.

Replace every `YOUR_USERNAME`, review the workspace, budget, secrets and mounts,
and choose a fresh output root before use. The recipes retain the historical run
names for provenance. Plan and validate before launching:

```bash
uv run python -m open_instruct.miles plan configs/ngu/qwen3_4b_base_deepscaler_baseline.toml
uv run python -m open_instruct.miles validate configs/ngu/qwen3_4b_base_deepscaler_baseline.toml
```

Launch through the committed-image workflow described in
[the MILES launch guide](../../docs/miles/launching.md). H100/A100 compatibility
and attention defaults come from the runtime already on main. Dataset preparation
and planning do not reserve compute; a training launch does.
