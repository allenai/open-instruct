# Qwen3 Deepscaler benchmark

`qwen3_4b_deepscaler.toml` is the standard Deepscaler recipe: 16 prompts x 32
completions, a 512-sample global batch, four trainer GPUs and four inference
GPUs. It uses learning rate 1e-6, an 8,192-token response limit, a 10,240-token
context, and barrier publication. Math rewards use the MILES-native 1/0 scale,
and gradient clipping defaults to 1.0. These differ from the original Open
Instruct experiment's 10/0 rewards and gradient clipping at 5; the recipe does
not claim identical learning curves.

Prepare its training and AIME 2025 evaluation inputs:

```bash
uv run python scripts/data/miles/prepare_deepscaler.py /weka/oe-adapt-default/YOUR_USERNAME/open-instruct/data/deepscaler-10k-base-01 --tokenizer /weka/oe-adapt-default/YOUR_USERNAME/checkpoints/Qwen3-4B-Base
```

The default training dataset is
`mnoukhov/deepscaler-10k-qwen3-4b-base-32samples-quartiles`. The helper renders
`qwen_instruct_user_boxed_math` prompts, excludes reference assistant answers,
and drops prompts over 2,048 tokens. It writes `train.jsonl`,
`eval-aime_2025.jsonl`, and `verifiers.json`. `--train-dataset` accepts another
dataset with the same `messages` and `ground_truth` schema.

Replace `YOUR_USERNAME`, review the workspace, budget, mounts and secrets,
and choose a fresh output root. This recipe enables online W&B, so configure
its API-key secret or switch `tracking.wandb_mode` to `"offline"`. Copy the
recipe to `runs/`, then plan and validate before launching:

```bash
uv run python -m open_instruct.miles plan runs/my-deepscaler.toml
uv run python -m open_instruct.miles validate runs/my-deepscaler.toml
```

Prepare the data on the submission host before training; the preparation script
is not included in the runtime image. Follow the [MILES launch guide](../../../docs/miles/launching.md)
for image selection and the committed-image launch workflow.
