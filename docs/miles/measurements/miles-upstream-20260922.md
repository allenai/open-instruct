# MILES upstream migration, September 22, 2026

The private fork's `main` now includes our runtime port onto upstream
`e89b45f7f85a0e76fba6a99474b1dd9b67a3c20e`. Its integration commit is
[`571560bb4`](https://github.com/allenai/miles/commit/571560bb4d22259cd1373dc6a2bfa1fb7551d651).
The previous runtime, `da91ac5c4`, remains available on
`robertb/open-instruct-runtime` for reproducing earlier images. Builds consume
immutable Git commits; no source patches are applied.

## Implementation

- Adopt upstream worker specifications and worker management. Register the Core
  backend in the trainer specification and retain its explicit successful-step
  result, checkpoint finalization, and transport teardown.
- Separate inference control from rollout execution. Core publication returns
  the completed optimizer version; the driver publishes it to the executor.
- Use asynchronous SGLang clients for publication and independent delivery.
  Engine drain holds the inference-controller update window until all submitted
  deliveries finish. Failed updates release the lock without admitting engines.
  Initial publication completes the controller handshake before router admission.
- Restore compiler caches before trainer imports and inside the serving child
  process. The serving environment explicitly includes the Olmo model package.
- Preserve native per-call token-version spans through serialization and training;
  refresh metadata retains response-relative spans and its separate replay version.
  Audit tools and rollout metrics read both native spans and historical flat
  version lists.
- Retain prompt ownership during cancellation, missing-reward filtering, queue
  metrics, router retirement/quarantine, tracking cleanup, and Bridge registration.
- Port the sibling consumer's depth-two IPC pipeline to upstream's transport
  protocol. Keep allocations alive through receiver acknowledgement and drain
  pending transfers before closing the publication session.
- Migrate the retained publication and update-zero diagnostic drivers. The old
  router spawn-target test is removed because that function no longer exists;
  upstream worker-specification and native router integration tests cover its
  replacement.

The fork delta is 45 files, +832/-132 lines against the upstream base. Deleted
upstream actor-group and engine-wrapper modules are not restored.

## Checks

Local RTX 4090 checks passed with the existing binary dependencies:

- Tiny Olmo MoE serving through upstream's publication client: change an actual
  weight tensor, observe changed generation, restore it, and reproduce the
  original tokens and log probabilities exactly.
- Qwen3: two optimizer updates, initial and repeated weight-equality checks,
  checkpoints, HF export, and teardown. A fresh process restores the checkpoint,
  completes another update, and exports again.
- Olmo 3: the same two-update lifecycle with sliding and full attention.

Focused CPU/runtime checks cover router retirement and health isolation,
worker specifications, controller lifecycle, arguments, tracking, async rollout,
native provenance, checkpoint ordering, and bounded IPC buffer lifetime. The
engine-drain tests exercise the real controller lock on success and failure.

The directly fetched runtime layer is
`sha256:4f5afc59a4fd7ece108cf78ab4064075751add67428a103d6c5c43d6375d0aff`.
Its source revisions match `runtime.lock.json`; its parser supplies the refreshed
option schema and native-help reference. The binary foundation, OLMo-core commit,
and olmo-sglang commit are unchanged.

## Limits

These checks do not repeat the earlier multi-node, EP, full-policy, refresh, or
engine-drain GPU qualifications. Their historical reports retain their original
image provenance. Local controller/transport tests are not a substitute for
those workloads.

The current SGLang dependency does not support every new upstream feature:
upstream's GDN LoRA target-parser test rejects `in_proj_qkvz`, and an Anthropic
session test imports a helper absent from this image. Neither path is part of
Core's supported full-weight training configuration. The real publication check
shows that the new begin/end update fields do not require a SGLang upgrade for
that configuration.

The broader suites are not entirely green. Seven behavioral failures reproduce
on the previous Open Instruct commit and the exact old MILES pin `da91ac5c4`:
five router logging/keep-alive expectations, a model-backend test fixture without
its required model, and a verifier test expecting failure rather than the current
zero-reward policy. The migration does not change those behaviors.

Historical tests also need inputs/dependencies missing from this runtime: the
sibling `policy_contract_schema` capture helper, an older private Core checkpoint
converter, and the legacy baseline's `/stage` source tree and DeepSpeed package.
These prevent claiming a clean run of the entire test suite.
