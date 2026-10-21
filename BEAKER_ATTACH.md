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
