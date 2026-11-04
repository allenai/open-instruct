# Screening training sources

`python -m open_instruct.miles screen` measures a fixed policy on the **same
prepared source population and registered rewards** as its MILES training run.
It calls the shared `prepare_run_data` entry point and `registered_reward` bridge;
there is no separate screening adapter, answer-extraction rule, or eligibility
filter. Existing training configurations keep their behavior.

Start a fixed-weight SGLang server for the run's `model.source`, using the serving
profile appropriate to its model and GPU. The screening command does not launch
a server or reserve GPUs. Run it in the MILES environment with the source's
optional verifier dependencies and configured external reward services:

```bash
python -m open_instruct.miles screen runs/source.toml \
  --tokenizer /path/to/the/same/hf-checkpoint \
  --server http://localhost:30000 \
  --output runs/source-screen-v1 \
  --prompts-per-source 96 --draws 16 --concurrency 64
```

Use the same `[data]`, reward settings and generation settings for training and
screening. A named catalog can produce the run's `[data]` fragment through
`python -m open_instruct.miles.datasets.source_catalog`. The preparation validates
the full source pool before sampling. An invalid/overlength row is an error,
not permission to silently remove it. Dataset revisions, prompt files, tokenizer
files, reward registries and sampled identities are recorded in the contract.

Sampling is uniform without replacement **within each prepared source**, up to
the requested count. Source fractions and pool sizes are recorded, but aggregate
mixture performance is not inferred from equally sized source panels. Retain
each source's intended internal subtask proportions in its prepared pool.

Training's temperature, top-p, top-k, stops, response budget and reward policies
are reused. Capped outputs follow `core.reward_zero_truncated`; failures to
generate or score are unknown, not valid zero rewards. The screen does not drop
constant-reward groups. Pass@k is reported only for complete binary-reward groups;
reward variation is also reported for complete nonbinary groups. Both report
their denominators. Inspect unresolved groups before interpreting the rates.

Individual completions, token IDs, rewards, verifier diagnostics and source
identities are saved under `draws/`. Rerunning the same command reuses completed
records and generates missing ones. Changed contracts require a fresh output
directory. Error records are preserved; they are not silently retried or replaced.
Keep model weights fixed throughout a screen: matching the advertised model path
does not authenticate a server whose weights are being updated in place.

Named judges require the same resolved judge registry as training. This initial
command supports single-turn text generation; custom rollout functions, tool
environments and reward postprocessors are rejected explicitly. A next-action
prediction must not be presented as an end-to-end environment screen.

When extending source support, share the verifier implementation first. Preserve
native prompts, targets, reward semantics and subtask weights unless creating an
explicitly named new source variant. Validate gold/wrong/near-miss controls and
offline/training parity on saved responses. Rescore existing completions when
only scoring changes; changed prompts or newly admitted populations need matched
generation evidence. Readiness, screen coverage and downstream training gains are
separate claims.

## Structured output sources

Install `scripts/data/rl_readiness/requirements.txt` in the screening and training
image before using optional candidate verifiers. Registry factory
`open_instruct.miles.rewards.structured_outputs.StructuredOutputVerifier` accepts
a target with `format` (`json`, `yaml`, `toml`, `xml`, `csv`), `schema`, an explicit
`schema_policy` (`as_declared` or `nemo_strict_v1`), and optional `coerce_types`.
These rewards measure structural compliance, not factual correctness. Preserve
original messages; no JSON-only suffix is needed for non-JSON sources.

`nemo_strict_v1` and XML/CSV type coercion preserve the pure helpers extracted from
NVIDIA NeMo Gym's Apache-2.0 `resources_servers/structured_outputs/app.py`
(SHA256 `5e9d85edaa6743abcdf0fd80859321bb9ce386e461846399069e80bb25cbbc70`).
Unlike the legacy JSON-only slice, the contract does not silently strengthen a
schema: the policy is part of every target and therefore the artifact hash.
Duplicate mapping keys, nonfinite values, external schema references and XML
entities are rejected. Local references and regex schema keywords are supported.
A worker process and a configurable response deadline (`timeout_seconds`, default
1 second) bound parsing and schema regex matching together. Timeout diagnostics
follow `OI_MILES_FORMAT_TIMEOUT_POLICY`, falling back to the math policy (`zero`
by default; `raise` preserves a scoring failure). Configuration errors propagate.
The old `JSONSchemaVerifier` remains unchanged.
