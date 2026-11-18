# Agent episodes with Core refresh

The experimental agent path trains Olmo 3.5 Tiny with OLMo-core while native
MILES TITO runs a linear, multi-turn agent episode. Ordinary weight publication
pauses serving, updates weights and resumes requests without cancelling the
agent coroutine or its sandbox. Later turns can therefore use newer weights.
This path requires the matching MILES revision in the runtime lock and a rebuilt
application image. CPU tests exercise provenance, native trajectory assembly,
tool masks, routing replay and lifecycle boundaries; GPU qualification of this
new combination is separate.

## Configure a run

Copy the disaggregated [node example](../../configs/miles/examples/tiny/node.toml)
to an ignored `runs/` file. Keep its Tiny model, trainer topology and inference
profile. Replace its data section and add the native generator settings:

```toml
[data]
agentic = true
prompt_data = "/weka/oe-training-default/YOUR_USERNAME/tmax/episodes.jsonl"

[async]
fully_async = true

[core]
publication_mode = "refresh"

[miles]
custom_generate_function_path = "miles.rollout.generate_hub.agentic_tool_call.generate"
custom_agent_function_path = "tmax_agent_function.run"
pause_generation_mode = "in_place"
rollout_stop = ["<|im_end|>", "<|endoftext|>"]
```

Merge these settings into the existing sections rather than declaring duplicate
TOML tables. Remove `data.reward_config`: the agent supplies the episode verdict.
Agent data selects TITO v2, `tito_model=olmo35`, a trajectory token limit equal to
Core's context limit, and the episode-reward bridge. Explicit values still win
and must satisfy validation. Do not enable `apply_chat_template`.

Each input row has an `input` messages list and an arbitrary `metadata` object:

```json
{"input":[{"role":"system","content":"You can use the bash tool to solve the task."},{"role":"user","content":"Inspect and repair the task."}],"metadata":{"task_id":"task-1","image":"your-task-image"}}
```

Fresh prompts support text system/user messages and must end with a user turn.
Preparation preserves the messages and task metadata, checks initial prompt
lengths and overlap, and records immutable input/tokenizer hashes. Tools are
provided by the agent at runtime, so their rendered instructions also consume
the session's context budget. Convert the colleague's prepared TMax JSONL by
renaming its `prompt` field to `input`; preserve the complete messages and metadata.
Do not pass it through the single-turn verifier preparation pipeline.

The agent function and its sandbox dependencies must be importable inside the
application image. The colleague's `tmax_agent_function.run` can retain its
native MILES signature and return `{"reward": ..., "agent_metrics": ...}`.
Keep its task directories, gateway/Podman setup and credentials available on the
rollout host. These services are supplied by the deployment, not started by Core.
Use the Tiny checkpoint's native Olmo template; the Qwen template and
`tito_model=qwen35` are not interchangeable with it.

Run `plan` and `validate`, inspect GPU placement and mount coverage, then use the
[committed-image launcher](launching.md). For a first smoke run disable
zero-variance filtering, use a small episode budget and keep enough episode
concurrency that at least one unfinished episode spans an optimizer update.

## Training and lifecycle contracts

TITO retains the exact sampled tokens, original behavior logprobs, per-call
weight-version spans and expert routes. Tool observations and inserted turn
delimiters have zero loss masks. Core accepts gaps in version coverage only at
masked positions; every trainable token must have a behavior version. Missing
provenance fails before training. A completed group is still discarded if its
oldest sampled token exceeds the configured policy-lag budget, even when its
last turn is recent.

Shared-engine evaluation and export close admission for **new episodes** and
wait for active episodes, including tools and collection, to finish. They do not
interrupt an episode between model calls. Set `core.engine_drain_timeout` to
cover that wait and bound episode/tool timeouts in the agent. Terminal failures
cancel owned episode tasks and use native agent cleanup. Checkpoint resume
regenerates pending prompts with new episodes; live sandbox state is not restored.

One trajectory per prompt sample is supported. Multiple TITO leaves, custom
session pickers/postprocessors, arbitrary generation functions, engine-drain
publication, TP>1 serving and automatic engine replacement remain unsupported.
Agents own their verdicts, including truncation and sandbox failure semantics;
the registered-verifier reward gates do not run on already rewarded episodes.

This uses Core's existing trainer-scored GRPO anchor with TIS. The colleague's
rollout-anchored DPPO-TV objective and Megatron OPD flags are not enabled by this
integration. Porting those objectives is separate from running the same TMax
environment and agent on Tiny.
