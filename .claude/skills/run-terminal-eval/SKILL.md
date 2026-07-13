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

**BRANCH: use `omni_agent`** (renamed from `omni_agent_evals_rebased`). All the
working eval scripts + fixes live here — `run_eval_local.sh`, the `--mirror-url`
pull-through mirror, the podman-login auth fix, and the `--model-provider` flag.
`master` (the public release) has an OLDER `launch_eval.sh`/`run_eval_in_job.sh`
and NO `run_eval_local.sh`. `git checkout omni_agent` before launching, and push
HEAD (Gantry clones by SHA — an unpushed commit fails the clone; else pass
`--repo-ref omni_agent`).

See also memory `project_tmax_harbor_eval.md` and the tmax `scripts/beaker/README.md`.

## Agent + parser compatibility (READ FIRST — this is where runs fail)

**The public release deleted every agent except `Vanillux2Agent`** (VanilluxAgent,
TassieAgent, TassumAgent are gone from `master`/`omni_agent`). Use
`Vanillux2Agent:Vanillux2Agent` (the current stock default) or a harbor built-in
(`mini-swe-agent`, `terminus-2`, `swe-agent`). The rows below for the deleted
agents are kept only for archival reference (restore from `origin/omni_agent_evals_pre_release` if ever needed).

**Parser by model family (for Vanillux2Agent's structured tool-calls):**
- **Qwen3.5 family** (Qwen3.5-*, tmax-2b/4b/9b/27b, Qwen3.6-27B — all arch
  `Qwen3_5ForConditionalGeneration`): `--tool-call-parser qwen3_xml`, no reasoning parser.
- **Qwen3 family** (Qwen3-8B, tmax-8b — arch `Qwen3ForCausalLM`): `--tool-call-parser hermes`
  **plus `--reasoning-parser qwen3`** (Qwen3 emits `<think>`; without the reasoning
  parser tool-calls don't split cleanly). Reasoning models are slow → higher `AgentTimeout` at high k.

**`--language-model-only`:** the Qwen3.5 arch is multimodal-capable, so vLLM tries
to load an image processor. The `allenai/tmax-*` models are text-only fine-tunes
that ship NO `preprocessor_config.json` → they **need `--language-model-only`** or
vLLM dies with `OSError: Can't load image processor`. The `Qwen/Qwen3.5-*` +
`shatu/*-Reasoning-Fix` bases DO ship it → don't pass the flag. (Qwen3 `*ForCausalLM`
models are text-only by arch and never need it.) Check via the HF repo's file list.

**⚠️ open-instruct RL/SFT Qwen3.5 checkpoints need CONVERSION, not the flag.** They
save as `architectures=["Qwen3_5ForCausalLM"]` / `model_type=qwen3_5_text`, which
**vLLM does not register** (only `Qwen3_5ForConditionalGeneration`) → it errors
`Model architectures ['Qwen3_5ForCausalLM'] are not supported`. `--language-model-only`
does NOT help. Convert first (lossless), then eval the `_cg` output WITHOUT the flag:
`tmax/scripts/beaker/convert_qwen35_causallm_to_cg.py --src <step_N> --donor Qwen/Qwen3.5-9B --out <..._cg>`
(grafts text weights onto the donor's vision tower + CG config + processors). Served-name
must not contain "ada". See memory `reference_qwen35_causallm_to_cg_conversion` and that
script's docstring / the tmax `scripts/beaker/README.md`.

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

Prereqs: HEAD pushed (Gantry clones the ref); `HF_TOKEN` + a `*_DOCKER_PAT`
secret. On `omni_agent` the Docker defaults are already `shashankg_DOCKER_PAT` /
username `shashankg209`, so no `DOCKER_PAT_SECRET=` prefix is needed. Keep it small
with `--n-tasks N` and/or the sample dataset — `terminal-bench@2.0` is 89 tasks.
`launch_eval.sh` also exposes Harbor resource/timeout overrides
(`--override-cpus/-gpus/-memory-mb`, the `--*-timeout-multiplier` flags) and
`--harbor-env daytona`.

```bash
cd ~/code/tmax && git checkout omni_agent
# Vanillux2Agent, Qwen3.5 (verified working config):
./beaker_configs/launch_eval.sh Qwen/Qwen3.5-4B \
  --name qwen35-4b \
  --agent Vanillux2Agent:Vanillux2Agent \
  --tool-call-parser qwen3_xml \
  --model-provider openai \
  --mirror-url jupiter-cs-aus-137.reviz.ai2.in:5000 \
  --gpus 1 --dataset terminal-bench@2.0 --max-model-len 32768 --n-attempts 1 \
  --workspace ai2/general-tool-use
```

**Docker Hub pulls (why runs used to fail under co-location):** harbor pulls the
`alexgshaw/*` task images via the podman socket. `run_eval_in_job.sh` does BOTH
`docker login` AND `podman login` (podman reads `containers/auth.json`, not
`~/.docker/config.json` — a plain docker login left pulls anonymous → shared-IP
`toomanyrequests` when jobs co-locate). With the enterprise `shashankg_DOCKER_PAT`
that's unlimited. **Better: `--mirror-url <host:port>`** points podman at a
docker.io pull-through cache (e.g. the AI2 registry mirror), so co-located jobs
pull over the local network and never touch Docker Hub; podman falls back to
authenticated docker.io if the mirror is down. Empty/omitted = direct pulls (fine
for one job). See `project_tmax_harbor_eval.md`.

- **k / pass@k:** `--n-attempts N` runs N attempts/task; `compute_stats.py` reports
  pass@1 and pass@N from one job (any k≤N is recomputable from stored scores).
- `--max-model-len`: Qwen3.5 default is ~262k, Qwen3-8B 40960 — cap it (32768
  typical; 65536 needs more KV, may want TP≥2) so vLLM's KV cache fits and it starts.
- 27B on H100 80GB: use `--gpus 2` (TP=2) for KV headroom at 32k+.
- For a weka checkpoint instead of an HF id, pass the path as `<model_path>` and
  add `--revision <branch>` for HF revisions.
- The `launch_eval.sh` "Harbor model:" banner is cosmetically wrong with
  `--model-provider` (shows `hosted_vllm/`); the real provider is derived in-job —
  verify via the job spec's `MODEL_PROVIDER`, not the banner.

Verify the spec before walking away:
```bash
beaker experiment spec <EXP_ID> --format json | uv run python -c "
import sys,json; d=json.load(sys.stdin); t=(d[0] if isinstance(d,list) else d)['tasks'][0]
print({e['name']:e.get('value') for e in t.get('envVars',[]) if e['name'] in
('AGENT_IMPORT_PATH','VLLM_TOOL_CALL_PARSER','VLLM_REASONING_PARSER','MODEL_PROVIDER',
'VLLM_LANGUAGE_MODEL_ONLY','MIRROR_URL','DATASET','MAX_MODEL_LEN','N_ATTEMPTS')})"
```
**exitCode 0 ≠ success:** always check `result.json` → `stats.n_errored_trials`
(and the `exception_stats` buckets). A job can finish "green" with every trial
errored (bad auth → all pulls fail; wrong served-name → all `NotFoundError`).

Common launch failures: `BeakerWorkspaceNotFound` → pass `--workspace ai2/general-tool-use`;
`no secret found …DOCKER_PAT` → set `DOCKER_PAT_SECRET=<user>_DOCKER_PAT`.

## Registry datasets (pass with `--dataset`, no path needed)

harbor 0.6.6's registry knows these — use `--dataset <name>` directly:
- `terminal-bench@2.0` — 89 tasks, the default.
- **`openthoughts-tblite@2.0` ("TBlite") — 100 tasks.** A normal registry dataset,
  so `--dataset openthoughts-tblite@2.0` works directly (NOT `--dataset-path`).
  Same family/parser/`--language-model-only` rules as TB2.0. Good lighter/complementary
  eval to run alongside terminal-bench. `--job-name <name>-vanillux2-k5` by convention.
- `terminal-bench-pro@1.0`, `terminal-bench-sample@2.0` — also registered.

Only **TerminalBench 2.1** is off-registry (needs `--dataset-path`, below).

## Off-registry datasets (e.g. TerminalBench 2.1)

Harbor 0.6.6's registry only knows `terminal-bench@2.0`. **TerminalBench 2.1**
(`terminal-bench/terminal-bench-2-1`, 89 revised tasks) lives only on the *current*
hub (hub.harborframework.com), which 0.6.6 can't reach — but its tasks use the
same `task.toml`+`environment/` format and `alexgshaw/*` images as 2.0, so 0.6.6
runs them fine via a **local path**. No harbor upgrade needed.

1. Download once with a newer harbor as resolver (namespaced ref required), onto
   the weka fs the beaker jobs mount:
   ```bash
   uvx harbor==0.18.0 datasets download terminal-bench/terminal-bench-2-1 \
     -o /weka/oe-adapt-default/shashankg/datasets --export
   # -> /weka/oe-adapt-default/shashankg/datasets/terminal-bench-2-1/<task>/ (89 dirs)
   ```
2. Run via **`--dataset-path`** (added to launch_eval.sh + run_eval_in_job.sh +
   run_eval_local.sh; maps to harbor `--path`, overriding `--dataset`):
   ```bash
   ./beaker_configs/launch_eval.sh allenai/tmax-4b \
     --dataset-path /weka/oe-adapt-default/shashankg/datasets/terminal-bench-2-1 \
     --agent Vanillux2Agent:Vanillux2Agent --model-provider openai \
     --tool-call-parser qwen3_xml --language-model-only \
     --gpus 1 --max-model-len 65536 --n-attempts 5 \
     --cluster ai2/jupiter --mirror-url jupiter-cs-aus-137.reviz.ai2.in:5000 \
     --workspace ai2/general-tool-use
   ```
   The dir MUST be under `oe-adapt-default` (launch_eval mounts it by default).
   All other rules (parser/provider/`--language-model-only`, mirror) are unchanged
   from 2.0. See memory `reference_terminalbench_2_1_eval.md`.

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
| `toomanyrequests: unauthenticated pull rate limit` (esp. co-located jobs) | podman pulls anonymous / Docker Hub shared-IP cap — ensure `omni_agent` (has `podman login`) and/or pass `--mirror-url` to use the pull-through mirror. |
| every trial `NotFoundError: model 'X' does not exist` | either co-located jobs collided on a fixed vLLM port (fixed on `omni_agent` by port randomization) OR `--harbor-model-name` model part ≠ served name — prefer `--model-provider` which derives `<provider>/<served-name>`. |
| `OSError: Can't load image processor for '<tmax model>'` | text-only fine-tune of the Qwen3.5 (multimodal) arch, no `preprocessor_config.json` — pass `--language-model-only`. |
