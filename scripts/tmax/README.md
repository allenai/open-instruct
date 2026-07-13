# tmax training scripts

These are the training launch scripts for the tmax fork (Qwen 3.5 + terminal agent training).

I recommend starting with the 1 GPU RL debug script (`qwen35_2b_1gpu.sh`), and then scaling up to the full-size scripts.
The debug script should run, take 3 steps, and exit without errors if everything is working.

**Important**: For people at ai2, you should run in beaker session with `BEAKER_ALLOW_SUBCONTAINERS=1` and `BEAKER_SKIP_DOCKER_SOCKET=1` set to avoid using beaker's own docker instance, which is not allowed. For podman to work, you should also use an image with ubuntu 24.04 or newer (e.g., one interactive session command that works is `beaker session create --gpus 1 --remote --bare --cluster ai2/saturn --image beaker://hamishivi/hamishivi-interactive --host-networking --mount src=weka,ref=oe-adapt-default,subpath=hamishi,dst=/root --mount src=weka,ref=oe-training-default,dst=/weka/oe-training-default --mount src=weka,ref=oe-adapt-default,dst=/weka/oe-adapt-default   --workspace ai2/olmo-instruct --workdir /root  --priority urgent --env BEAKER_ALLOW_SUBCONTAINERS=1 --env BEAKER_SKIP_DOCKER_SOCKET=1 -- bash`). You also might need to run `unset LD_LIBRARY_PATH` to avoid conflicts with the system libraries when running stuff in interactives.

### Podman setup

Before running the script, you should install podman and crun by running the following command.
If you already have podman and crun installed, you can skip this step.

```bash
bash scripts/docker/setup_podman.sh
```

This script is made for Ai2 machines, so you may need to modify it to work on your own infrastructure.

## Scripts here

Scripts are split into two folders by training stage:

### `SFT/` — supervised finetuning (`open_instruct/finetune.py`)

| Script | What it does |
| --- | --- |
| `sft_qwen3_8b_small.sh` | SFT of Qwen3-8B on just the tmax SFT data |
| `sft_qwen3_8b_big.sh` | SFT of Qwen3-8B on the full SFT blob (all subsets) |
| `sft_qwen35_9b_small.sh` | SFT of Qwen3.5-9B on just the tmax SFT data |
| `sft_qwen35_9b_big.sh` | SFT of Qwen3.5-9B on the full SFT blob (all subsets) |

### `RL/` — DPPO RL (`open_instruct/grpo_fast.py`)

| Script | What it does |
| --- | --- |
| `qwen35_2b_1gpu.sh` | DPPO RL on Qwen3.5-2B with `swerl-tmax-15k` (1 GPU, for debugging) |
| `qwen35_2b.sh` | DPPO RL on Qwen3.5-2B with `swerl-tmax-15k` |
| `qwen35_4b.sh` | DPPO RL on Qwen3.5-4B with `swerl-tmax-15k` |
| `qwen35_9b.sh` | DPPO RL on Qwen3.5-9B with `swerl-tmax-15k` |
| `qwen36_27b.sh` | DPPO RL on Qwen3.6-27B with `swerl-tmax-15k` |
| `qwen3_8b.sh` | DPPO RL on Qwen3-8B (SFTed version) with `swerl-tmax-15k` |

The RL scripts mostly share the same args; the per-size differences are model name,
node/engine counts, rollout shape, and a few memory/perf flags (e.g. the 27b adds
`--gather_whole_model false` / `--deepspeed_zpg 1` to be able to fit the model during training).

## Reading a script: `mason.py` vs. the regular training command

Every script has the same two-part shape, split by a bare `--`:

```bash
uv run python mason.py \
    <mason / Beaker scheduling args> \
    -- \
    <the actual command that runs on the cluster>
```

### Part 1 — `mason.py` (the launcher)

`mason.py` is Ai2's Beaker job launcher. Everything **before** the `--` configures
*where and how* the job runs on the cluster, not the training itself. Common args:

- `--cluster` / `--workspace` / `--budget` — which Beaker cluster, workspace, and budget
- `--image` — the Docker image to run in (passed as `$1` to the script)
- `--num_nodes` / `--gpus` — how much hardware to request
- `--priority` / `--preemptible` / `--max_retries` — scheduling behavior
- `--pure_docker_mode`, `--mount_docker_socket`, `--env`, `--secret` — container/runtime setup

`mason.py` also does extra bookkeeping for known `open_instruct` commands (e.g.
`finetune.py`, `grpo_fast.py`): caching, auto-resume for GRPO, etc.

If you aren't at Ai2, then you can remove this part of the command. But note how many GPUs and nodes are requested, since you need to match that!

### Part 2 — the regular training command (after `--`)

Everything **after** the `--` is the command mason actually executes on each node.
This is the "regular script" you would run locally if you weren't using Beaker:

- SFT scripts: `accelerate launch ... open_instruct/finetune.py <training args>`
- RL scripts: `... python open_instruct/grpo_fast.py <training args>`

These args (`--model_name_or_path`, `--dataset_mixer_list`, `--learning_rate`,
`--max_seq_length`, `--response_length`, `--tools`, etc.) are the actual
hyperparameters and are documented by `finetune.py` / `grpo_fast.py` themselves.

## Running locally vs. on Beaker

To run without Beaker, drop the `uv run python mason.py ... --` prefix and run the
command after the `--` directly (adjusting `--num_processes`/node settings for your
hardware). To launch on the cluster, just run the script with a Beaker image:

```bash
bash scripts/tmax/SFT/sft_qwen3_8b_small.sh <beaker-image>
```

For people at Ai2, you can just run the script with a Beaker image. I recommend using the build and launch scripts, e.g.:
```bash
bash scripts/train/build_image_and_launch.sh training/open-instruct/scripts/tmax/RL/qwen35_2b.sh
```

## General Tips

Training terminal agents is tough, painful, and requires a lot of patience. Here are some useful tips I found for this stack:

- Sometimes nodes go down because of 'running hot', since we run podman on the same nodes (unideal, but avoids paying money to big sandbox). Hence, turning off the podman janitor/lowering the pool size/lowering the start and exec concurrency can help preventing nodes from going down on restarts, if you repeatedly hit it.
- Increasing group size and output length helps a lot with stability (we show this a bit in the paper). In particular, training with ~65k output length seems quite important for the 9B and 4B model, otherwise the model has too many 'overlong negatives' and too much negative gradient.
- 2 nodes is all you need for training max, and put as much extra as you need for inference.
- Counterintuitively, training larger models can be more stable and just as fast as the smaller models, since often you are bottlenecked on sandbox memory/setup, and so have 'gpu wiggle room' for running slower, larger models.
- Instabilities seem to arise from the model getting into insane entropy states. Adding well-formedness rewards on language / tool calls may help in training stability.
