# Attach to the Hammond Beaker checkout

Your helper is in `~/.ai2_bashrc`. It connects over SSH to a Beaker container
and resumes tmux. The cluster checkout for this branch is actually
`/weka/oe-adapt-default/michaeln/nit-open-instruct`, not `open-instruct-hybrid`.

## Attach from your local terminal

```bash
source ~/.ai2_bashrc
beaker session list --author michaeln --cluster ai2/hammond
export SESSION_ID=01M5FZAJ0GKWGRBYFRCHCW3CCG
beakera
# Inside the session:
cd /weka/oe-adapt-default/michaeln/nit-open-instruct
```

That ID is the replacement session created on October 21, 2026. Select a new
running ID if it ends. `beakera` honors `$SESSION_ID`; if unset, it picks the
first data row from your Hammond session list. It uses
`/home/mnoukhov/.ssh/ai2_id_ed25519`, resumes tmux session `beaker` with
`/root/.config/tmux/tmux.conf`, and starts in `/weka/oe-adapt-default/michaeln`.
Detach with `Ctrl-b d`.

For automated commands, use SSH to the host followed by Beaker exec:

```bash
ssh michaeln@hammond-cs-aus-452.reviz.ai2.in \
  "beaker session exec 01M5FZAJ0GKWGRBYFRCHCW3CCG -- bash -c '
    cd /weka/oe-adapt-default/michaeln/nit-open-instruct &&
    git status --short --branch && git log -1 --oneline
  '"
```

This was verified end to end. SSH to Hammond alone does not enter the container
or provide its Weka mounts. The local CLI's `session exec --remote` requires a
terminal and loses argument boundaries for compound commands passed as separate
arguments; the SSH-then-exec form above avoids that issue.

## Replace an ended session

`beakernew` in `~/.ai2_bashrc` contains a stale `--host-networking` option that
both checked CLIs reject (local v1.5.247, Hammond v1.5.342). Remove that line
from the helper before sourcing it and calling `beakernew` again. The helper
file was not changed during this inspection. The replacement was created with:

```bash
beaker session create \
  --budget ai2/oe-other --bare --cluster ai2/hammond \
  --mount src=weka,ref=oe-adapt-default,dst=/weka/oe-adapt-default \
  --mount src=weka,ref=oe-adapt-default,subpath=michaeln/.config,dst=/root/.config \
  --mount src=weka,ref=oe-training-default,dst=/weka/oe-training-default \
  --mount src=weka,ref=oe-eval-default,dst=/weka/oe-eval-default \
  --mount src=secret,ref=michaeln_id_ed25519,dst=/root/.ssh/id_ed25519 \
  --mount src=secret,ref=michaeln_bashrc_remote,dst=/root/.bashrc \
  --mount src=secret,ref=michaeln_ssh_config,dst=/root/.ssh/config \
  --mount src=secret,ref=michaeln_gitconfig,dst=/root/.gitconfig \
  --mount src=secret,ref=michaeln_CODEX_AUTH,dst=/root/.codex/auth.json \
  --secret-env HF_TOKEN=michaeln_hf_token \
  --secret-env WANDB_API_KEY=michaeln_WANDB_API_KEY \
  --secret-env BEAKER_TOKEN=michaeln_beaker_token \
  --workdir /weka/oe-adapt-default/michaeln \
  --image beaker://ai2/cuda12.8-dev-ubuntu22.04-notorch \
  --priority low --workspace ai2/michaeln --detach
```

This requests no GPU, CPU, memory, or timeout explicitly. Record the returned
ID and export it before attaching: `beakernew` clears `$SESSION_ID` but does not
set it to the newly created ID. Weka files survive session replacement.

## Checkout comparison on October 21 2026

Both checkouts use branch `hybrid-small-suite`. Before adding this guide and
updating the research logs, the local working tree was clean; the cluster
working tree was also clean.

| Checkout | Path | HEAD |
| --- | --- | --- |
| Local | `/home/mnoukhov/open-instruct-hybrid` | `9584735ef421640cc282b19fccaf0f9223466e36` |
| Hammond | `/weka/oe-adapt-default/michaeln/nit-open-instruct` | `e2d831a5ca7ad8d62ff3e554460753ec17ae62bb` |

The histories diverge at `8fc89d9584913b93d56f46cf6b566b7a316140ff`:
**1 local-only commit and 76 cluster-only commits**, with 29 files differing
(2,315 insertions and 802 deletions when going from local HEAD to cluster HEAD).
The local-only commit fixes evaluation launch scripts; cluster-only work includes
GRPO/GDN fixes, chat templates, Dolci scoring, and expanded experiment notes.

See [the comparison run log](experiment.md#hammond-checkout-comparison-2026-10-21)
for the findings. The subsequent synchronization merged both histories on
`hybrid-small-suite`, preserving the local evaluation fix and cluster work.
Use `git rev-parse HEAD` and `git status --short --branch` in each checkout
to verify their current commit and working-tree state.

## Relation to the coworker example

`/home/mnoukhov/Downloads/codexbeaker.txt` describes creating a detached session
and enabling Codex Desktop remote control. `beakera` provides SSH/tmux access;
Desktop remote control requires additional setup. Your session already mounts
`michaeln_CODEX_AUTH` at `/root/.codex/auth.json`. The example's credential-copy,
installation, and process-restart commands were reference material, not executed
instructions. Codex remote control was not configured by this inspection.


## Interactive Rhea GPU qualification (2026-10-27)

Session `01M5ZB6KJDNFG7KYKWHN8X948W` runs on
`rhea-cs-aus-473.reviz.ai2.in` with one RTX A6000 48GB, 12 requested CPUs,
96 GiB requested host memory, and 16 GiB shared memory. It uses the prepared
SGLang image `01M5ZA7AP531QVWGSAGSC8M5B8`, a 90-minute timeout, normal priority,
and `min_runtime=0`; workspace `ai2/open-instruct-dev`, budget `ai2/oe-other`.
Verify it is still running before using this historical ID.

From a real local terminal (remote exec was verified with a PTY):

```bash
beaker session exec --remote 01M5ZB6KJDNFG7KYKWHN8X948W \
  --key /home/mnoukhov/.ssh/ai2_id_ed25519 -- tmux attach -t hybrid-tests
```

The live runtime checkout is
`/weka/oe-adapt-default/michaeln/miles-olmo-core-hybrid` on
`codex/hybrid-mainline`; the companion checkout is
`/weka/oe-adapt-default/michaeln/open-instruct-miles-hybrid`.
Set `PYTHONPATH` to the runtime checkout before the image's existing path when
running experiments, so code edits apply without rebuilding:

```bash
cd /weka/oe-adapt-default/michaeln/miles-olmo-core-hybrid
export PYTHONPATH="$PWD:$PYTHONPATH"
python tools/hybrid/qualify_sglang.py --size 275m --reference-backbone --output-dir /weka/oe-adapt-default/michaeln/open-instruct-hybrid/qualification/my-test
# Or: --size all --sizes 450m 810m 1.4b 2.7b
```

The older local Beaker CLI lacks `session create --min-runtime`. The newer CLI
in the existing Hammond session supports it. Rhea rejects protected sessions
without an allocation, so explicitly use `--min-runtime 0` when creating a
replacement. Weka mounts do not accept `mode=readonly`; the qualification code
only reads release checkpoints and writes its own report directory.

The BF16 reference-backbone mode reuses the release HF/FLA model under SGLang
scheduling. Without `--reference-backbone`, the faster packed/RadixAttention path
has numerical parity failures at some scales; successful loading is not a pass.
For MILES-managed workers the equivalent setting is
`MILES_HYBRID_REFERENCE_BACKBONE=1`, already explicit in the temporary recipe.
