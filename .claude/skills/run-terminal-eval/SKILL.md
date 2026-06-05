---
name: run-terminal-eval
description: Run Terminal-Bench (harbor) evals of a model — serve it with vLLM and run an agent (mini-swe-agent / Vanillux2Agent / SWE-agent) against terminal-bench tasks — locally on a dev VM (real Docker) or on Beaker. Use when the user wants to evaluate a model on terminal-bench / tmax terminal tasks, run launch_eval.sh or run_eval_local.sh, smoke-test an agent+model on terminal tasks, or "test the terminal eval". NOT for RL-run analysis (use analyze-terminal-rl for that).
---

# Run Terminal-Bench evals (harbor) locally or on Beaker

Serves an HF/weka model with vLLM and runs a harbor agent against terminal-bench
tasks (`--env docker`). Lives in the **tmax repo** (`~/code/tmax`), not
open-instruct:

- `beaker_configs/launch_eval.sh` — launch one Beaker task (Gantry).
- `scripts/beaker/run_eval_in_job.sh` — inner script run inside the Beaker task
  (also the reference for the full pipeline). Uploaded from the local tree by
  Gantry, so local edits take effect without committing (but the repo is *cloned*
  at the pushed git ref, so push HEAD first).
- `beaker_configs/run_eval_local.sh` — local mirror (real Docker, no Daytona).

See also memory `project_tmax_harbor_eval.md` and the tmax `scripts/beaker/README.md`.

## Agent + parser compatibility (READ FIRST — this is where runs fail)

The repo's `uv.lock` pins **harbor 0.6.6**. Pick the agent accordingly:

| Agent (`--agent`) | Works on 0.6.6? | Provider | Tool parser | Notes |
|---|---|---|---|---|
| `VanilluxAgent:VanilluxAgent` | **NO** | hosted_vllm | hermes | Imports `ExecInput`/`create_run_agent_commands`, absent from 0.6.6 / all released harbor / main. Fails at import (`cannot import name 'ExecInput'`). This is the stock `launch_eval.sh` default — **don't use until the harbor pin is fixed.** |
| `mini-swe-agent` (built-in) | YES | **openai** | any (hermes ok) | Parses `bash` code blocks from plain text, so it does NOT depend on the tool-call parser. |
| `Vanillux2Agent:Vanillux2Agent` | YES | **openai** | **qwen3_xml** (Qwen3.5) | `BaseAgent` with its own litellm loop using *structured* tool-calls → needs the right parser (see below). Runs host-side. |
| `TassieAgent:TassieAgent` | YES | **openai** | **qwen3_xml** (Qwen3.5) | Bash-only litellm loop, also **structured** tool-calls (`tools=[bash_tool]`) → same parser requirement as Vanillux2Agent. Runs host-side. |
| `TassumAgent:TassumAgent` | YES | **openai** | **qwen3_xml** (Qwen3.5) | TassieAgent + context summarisation; same structured tool-calls. |
| `terminus-2` (built-in) | YES | **openai** | qwen3_xml (safe) | Harbor's built-in agent. |
| `swe-agent` (built-in) | YES | hosted_vllm | hermes | Upstream SWE-agent inside the sandbox. |

**Tool-call parser (`--tool-call-parser`)**: `Qwen3.5` emits
`<function=name><parameter=…>` XML. Agents that use litellm structured
tool-calling (**TassieAgent, TassumAgent, Vanillux2Agent**) need **`qwen3_xml`** — with the default `hermes`
the tool-calls are silently dropped and the agent loops on "Format error" then
gives up (0 useful steps). `qwen_xml` is NOT a valid name; the valid ones include
`hermes, qwen3_coder, qwen3_xml, …`. Agents that parse text bash blocks
(mini-swe-agent) are unaffected by the parser.

**Provider (`--model-provider`)**: the installed harbor's litellm has no usable
`hosted_vllm` path, so built-in agents and Vanillux2Agent use **`openai/<served>`**
(+ `OPENAI_API_BASE`/`OPENAI_API_KEY=dummy`). Only the SWE-agent import-path agents
use `hosted_vllm/`. `run_eval_in_job.sh` defaults this per agent type; override
with `--model-provider`. Do NOT set `MSWEA_API_KEY` (mini-swe-agent forwards only
that and skips `OPENAI_API_KEY` → litellm "Missing credentials").

## Run on Beaker (preferred — verifier patches applied automatically)

Prereqs: HEAD pushed (Gantry clones the ref); a `*_DOCKER_PAT` and `HF_TOKEN`
secret in the workspace. Keep it small with the sample dataset (there's no
`--n-tasks` flag; `terminal-bench@2.0` is 89 tasks).

```bash
cd ~/code/tmax
# Vanillux2Agent (verified working config):
DOCKER_PAT_SECRET=shashankg_DOCKER_PAT ./beaker_configs/launch_eval.sh Qwen/Qwen3.5-4B \
  --name qwen35-4b-vanillux2 \
  --agent Vanillux2Agent:Vanillux2Agent \
  --tool-call-parser qwen3_xml \
  --model-provider openai \
  --gpus 1 --dataset terminal-bench-sample@2.0 --max-model-len 32768 \
  --workspace ai2/general-tool-use
```

- `--max-model-len 32768`: Qwen3.5's default context is 262k; cap it so vLLM's KV
  cache fits one GPU and the run actually starts.
- For a weka checkpoint instead of an HF id, pass the path as `<model_path>` and
  add `--revision <branch>` for HF revisions.
- Stock-config reproduction (to show the VanilluxAgent failure to developers):
  drop the agent/parser/provider flags.

Verify the spec before walking away:
```bash
beaker experiment spec <EXP_ID> --format json | uv run python -c "
import sys,json; d=json.load(sys.stdin); t=(d[0] if isinstance(d,list) else d)['tasks'][0]
print({e['name']:e.get('value') for e in t.get('envVars',[]) if e['name'] in
('AGENT_IMPORT_PATH','VLLM_TOOL_CALL_PARSER','MODEL_PROVIDER','DATASET','MAX_MODEL_LEN')})"
```

Common launch failures: `BeakerWorkspaceNotFound` → pass `--workspace ai2/general-tool-use`;
`no secret found …DOCKER_PAT` → set `DOCKER_PAT_SECRET=<user>_DOCKER_PAT`.

## Run locally (dev VM, real Docker)

`run_eval_local.sh` serves vLLM on one GPU and runs harbor against the host Docker
daemon. It auto-installs the `docker compose` plugin, **authenticates to Docker
Hub** (reads `$DOCKER_PAT` or the `shashankg_DOCKER_PAT` beaker secret, `docker
login -u shashankg209`, hard-aborts on failure — no anonymous fallback; also
strips a broken dev-containers `credsStore`), and applies the `network_mode:
host` patch. Defaults: `mini-swe-agent`, `Qwen/Qwen3.5-4B`, 2 tasks.

```bash
cd ~/code/tmax
./beaker_configs/run_eval_local.sh Qwen/Qwen3.5-4B --n-concurrent 1 --task fix-git
```

Two local-only realities (NOT present on Beaker, where podman runs inside the job
container so `localhost` works):

1. **Sibling-container networking.** Harbor task containers are siblings of this
   session container via the shared Docker daemon; `network_mode: host` puts them
   on the *real host* netns, so `localhost:8008` does NOT reach a vLLM running in
   this container. For SWE-style **in-container** agents, point the agent at this
   container's bridge IP (`hostname -i`, e.g. `172.17.0.4`). `run_eval_local.sh`
   does this automatically. **Vanillux2Agent runs host-side** (only bash execs go
   into the container), so `localhost` works for it.

2. **Verifier patches → rewards.** Producing `reward.txt` needs harbor's
   verifier/oracle/paths chmod patches (`run_eval_in_job.sh` applies them; editing
   `site-packages` may be permission-gated). Without them the agent still runs but
   the trial ends in `RewardFileNotFoundError` (empty `verifier/` dir). To confirm
   end-to-end rewards locally, apply those three patches (see `run_eval_in_job.sh`
   step 3) or just run on Beaker.

Iterating fast: serve vLLM once in the background with the right parser, then call
`uv run harbor run … --include-task-name <task>` directly:
```bash
CUDA_VISIBLE_DEVICES=0 uvx vllm==0.19.1 serve Qwen/Qwen3.5-4B \
  --served-model-name Qwen3.5-4B --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml --port 8008 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --tensor-parallel-size 1 &
# wait until curl -sf localhost:8008/v1/models succeeds (~4 min), then:
export OPENAI_API_KEY=dummy OPENAI_API_BASE=http://localhost:8008/v1 OPENAI_BASE_URL=http://localhost:8008/v1
uv run harbor run --dataset terminal-bench@2.0 --include-task-name fix-git \
  --agent-import-path Vanillux2Agent:Vanillux2Agent --model openai/Qwen3.5-4B \
  --agent-kwarg api_base=http://localhost:8008/v1 --env docker -n 1 \
  --job-name smoke --yes -k 1
```

## Tasks, results, monitoring

- Limit tasks: `--n-tasks/-l N` (first N) or `--include-task-name/-i <name>`
  (NOT `--task`, which wants an `org/name` registry ref). tb2 tasks come from
  `github.com/laude-institute/terminal-bench-2`.
- Results: `jobs/<job-name>/` → `result.json` (stats + exception buckets),
  per-trial `<task>__<rand>/{agent/trajectory.json, verifier/reward.txt,
  exception.txt, trial.log}`, plus `metrics.json`/`stats.txt` from
  `scripts/compute_stats.py`. On Beaker these are copied to `/results` (and weka).
- Monitor a Beaker run with the `monitor-experiment` skill, or poll
  `beaker experiment get <id> --format json` (job `status.finalized`/`exitCode`).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `cannot import name 'ExecInput'` | VanilluxAgent vs locked harbor 0.6.6 — use Vanillux2Agent or mini-swe-agent. |
| Agent loops on "Format error", ~0 progress | Wrong tool parser — use `--tool-call-parser qwen3_xml` for Qwen3.5 structured-tool agents. |
| litellm "Missing credentials" | `MSWEA_API_KEY` is set (mini-swe-agent skips OPENAI_API_KEY), or wrong provider — use `openai/`, set `OPENAI_API_KEY=dummy`, unset `MSWEA_API_KEY`. |
| litellm "Connection error" (local) | Agent used `localhost` but is a sibling container — use this container's bridge IP. |
| `RewardFileNotFoundError`, empty `verifier/` | harbor verifier/oracle/paths patches not applied (local manual runs) — apply them or run on Beaker. |
| `'compose' is not a docker command` | docker compose plugin missing — install v2 plugin (set_dev_vm.sh does this on dev VMs). |
| vLLM never ready / OOM | 262k default context — pass `--max-model-len 32768`. |
