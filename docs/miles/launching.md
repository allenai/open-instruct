# Launching MILES runs

This guide covers the current **config-driven launcher**. It does not require
sibling olmo-miles, MILES or Core worktrees. The selected runtime image contains
the training dependencies and adapter sources. See [architecture](architecture.md)
for building and updating that image.

## Prepare a submitting host

Use the project's `robertb/miles-olmo-core` branch. Python 3.12 is sufficient for
CPU-only planning and submission from the checkout; those modules use the standard
library. Do not install the full CUDA training environment on a laptop merely to
submit a run. Install and authenticate the Beaker CLI using
[AI2 setup](../get_started/ai2_internal_setup.md); the account needs access to the
configured workspace, budget, secrets, images and WEKA filesystems. Building an
image additionally requires Docker and `jq`.

```bash
git clone --branch robertb/miles-olmo-core https://github.com/allenai/open-instruct.git
cd open-instruct
uv python install 3.12
uv venv --python 3.12 "$HOME/.venvs/oi-miles-submit"
source "$HOME/.venvs/oi-miles-submit/bin/activate"
beaker account login
beaker config test
beaker account whoami
```

The virtual environment directory must be ignored or outside the repository so
that it does not make the checkout dirty. Store personal run files outside the
checkout, or in a Git-ignored directory. All paths below are shell examples;
replace checkpoint/output paths, usernames and image placeholders.

```bash
mkdir -p "$HOME/miles-runs"
cp configs/miles/examples/grpo-basic.toml "$HOME/miles-runs/check.toml"
# Edit check.toml: select a compatible tiny checkpoint and a fresh output root.
python -m open_instruct.miles plan "$HOME/miles-runs/check.toml"
python -m open_instruct.miles validate "$HOME/miles-runs/check.toml"
```

`plan` and structured `validate` inspect schema/options without reading model
weights or allocating GPUs. They do not verify remote mounts, credentials,
memory fit or the installed runtime. Input paths are resolved relative to the
TOML location, with no implicit environment substitution. WEKA paths need not
exist on the submitting host.

## Laptop: choose or build an image

For an **already built compatible image**, obtain its immutable ID and source
provenance from its maintainer or qualification record. Inspect its metadata:

```bash
beaker image get IMMUTABLE_IMAGE_ID --format json
export MILES_EXISTING_IMAGE=IMMUTABLE_IMAGE_ID
python -m open_instruct.miles run "$HOME/miles-runs/check.toml"
```

`MILES_EXISTING_IMAGE` accepts an immutable Beaker ID, not an alias. The launcher
checks the ID, but does **not** prove source compatibility with your checkout.
The submitted config is carried into the job; your local Python changes are not.
Use an image built from the intended source revision. The launch receipt records
the submitting revision and selected image separately.

To deploy source changes, commit them and build the source overlay instead.
Read the immutable base identity from the lock and pull it into local Docker:

```bash
unset MILES_EXISTING_IMAGE
base_id=$(python -c 'import json; print(json.load(open("runtime/miles/runtime.lock.json"))["base_image"]["beaker"])')
beaker image pull "$base_id" miles-core-base
export MILES_BASE_IMAGE=miles-core-base
python -m open_instruct.miles run "$HOME/miles-runs/check.toml"
```

The builder checks the loaded Docker image ID against the lock, builds the source
overlay, uploads it to Beaker, then submits. It reuses a matching image when one
already exists for the commit. This is not an automatic blessed-image resolver.
Do not use the general Open Instruct/vLLM auto image for MILES.

Both modes require a clean, committed checkout and delegate through
`scripts/train/build_image_and_launch.sh --miles`. `run` performs launch checks
before building, including mount coverage. It does not implement olmo-miles'
`--skip-local-gate` option or its image-preflight contract.

## From a Beaker session

To **submit a new allocation**, use the same existing-image procedure above.
Use the session's provisioned BEAKER_TOKEN or authenticate with `beaker account login`,
check `beaker account whoami`, and select a compatible immutable image;
Docker is unnecessary. Session GPUs and mounts are not inherited by the new job:
its TOML must request everything it needs. Do not use a shared Docker daemon just
to resubmit a known image.

To **train inside an existing single-node allocation**, the compatible Core RL
runtime must already be installed, with all configured GPUs, input paths, output
paths and credentials available. From the runtime checkout:

```bash
cd /opt/core-rl
python -m open_instruct.miles validate /path/to/run.toml
python -m open_instruct.miles train /path/to/run.toml
```

`train` does not allocate nodes, mount WEKA or provision managed judges. Use `run`
for multi-node or managed-judge jobs: its cluster bootstrap coordinates ranks,
Ray and services. Do not manually start one independent `train` on each node.
The submitted workflow runs the attention preflight for the supported torch/FA4
selection before training.

## Placement, secrets and results

GPU examples use Holmes, `ai2/open-instruct-dev`, urgent priority and a one-hour
minimum runtime. CPU-only preparation requiring WEKA must use **ai2/saturn**.
Set every model/template/data/output/cache filesystem in `launch.weka_mounts`.
See [topology](topology.md) for replica and engine counts and
[managed judges](managed-judges.md) for preparation and placement.

```toml
[launch.secrets]
HF_TOKEN = "your-hf-beaker-secret"
WANDB_API_KEY = "your-wandb-beaker-secret"
```

These are secret names, not token values. Use `launch.env` only for nonsecret
strings. Reserved Ray/rank/CUDA environment variables belong to the launcher.
Offline W&B examples do not require a W&B credential.

```bash
python -m open_instruct.miles status "$HOME/miles-runs/check.toml"
beaker experiment get EXPERIMENT_ID --format json
beaker job events JOB_ID
beaker job logs JOB_ID
```

`status` uses receipts in `~/.cache/open-instruct/miles/launches`; override with
`MILES_LAUNCH_RECEIPTS`. Keep that directory when switching submitter hosts.
Inspect the latest attempt for **each task**, not the first job of a retried
experiment. Scheduler events explain pending jobs. Small reports/logs are copied
to Beaker results; checkpoints and rollout tensors remain on WEKA. See
[operations](operations.md) for completion, recovery and artifact interpretation.
