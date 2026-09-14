# MILES GRPO

Use MILES GRPO for reinforcement learning through Open Instruct, with OLMo-core
training and SGLang inference. This guide covers setup, the verified runtime image
and a first run. [Support and boundaries](feature-parity.md) remain specific to the
model and topology; retain each run's image, configuration and checkpoint identity.

## Run MILES GRPO

1. Obtain the matching source bundle or supplied checkout, then follow the
   [laptop/session setup](launching.md). Sibling development worktrees are unnecessary.
2. Copy [grpo-sharing.toml](../../configs/miles/examples/grpo-sharing.toml) to
   Git-ignored `runs/my-grpo.toml`. Set your name and fresh output root; keep the
   supplied checkpoint for the first three-B300-GPU check. W&B is offline, with
   no extra HF/W&B credentials required for these inputs.
3. Run `plan` and `validate`, then set `MILES_EXISTING_IMAGE` to the immutable
   image ID and invoke `python -m open_instruct.miles run /path/to/run.toml`.
4. Retain the launch receipt, submitted TOML and [completion artifacts](operations.md).
   Report a symptom with the experiment ID, image, model and configuration.

For an internal source handoff, the standalone `open-instruct-miles-sharing.bundle`
can be cloned without publishing the integration history to the public GitHub
repository:

```bash
git clone -b robertb/miles-olmo-core /path/to/open-instruct-miles-sharing.bundle open-instruct
cd open-instruct
```

Then follow laptop/session setup above. The image contains the runtime dependencies;
the submitting laptop does not need Core, MILES or SGLang sibling checkouts.
Documentation/evidence and the host-only audit launcher added after image revision
`2de5c5ba4` do not change its runtime code. Use the supplied source with this image;
do not silently combine it with a different runtime branch.

The [support matrix](feature-parity.md) and [measurements](measurements/index.md)
distinguish short lifecycle checks, learning-path evidence and longer experiments.
Use those recorded boundaries when choosing a model, topology or recipe.

## Image and qualification

The binary foundation remains Beaker `01M24E7MSDGN2QFW1T8Z31BCKS`. The new reusable
`runtime-base` stage prepares the pinned Core/MILES/serving trees and verifier
dependencies. The `application` stage adds the consolidated code, tests, configs
and documentation. No CUDA/framework upgrade or opportunistic package removal
is included. Historical Megatron dependencies in the binary foundation are not
selected as the trainer.

The immutable Beaker image is **`01M2E5QR5C60WF7H0TDEF4CD3S`**
(`robertb/miles-core-sharing-2de5c5ba4`), built from application revision
`2de5c5ba421d653a95981fa1bb35dcb64231aeb3`. See the
[qualification record and provenance](measurements/sharing-20260913/README.md).
The combined GPU gate, separate full-SFT check and independent retained-sample
audit all passed with exit code zero. The gate
runs runtime tests, dense synthetic save/resume, two-rank FSDP diagnostics, EP2
packing/recomputation and live MoE replay/resume. A separate two-collection full-SFT
[config](../../configs/miles/qualification/sharing-sft-20260913.toml) checks async
TIS, packing/replay, initial/final evaluation and startup-only weight audits.

One optional cross-backend diagnostic test module requires the separate
olmo-miles comparison checkout and is explicitly excluded from the standalone
runtime suite. It is not a missing dependency of the training product. GPU tests
from this gate are qualification evidence, not the repository CI GPU-test receipt.

## Documentation and agent entry point

Agents discover this workflow from the repository README and AGENTS.md; no
special prompt is needed. Follow the [documentation index](index.md) for model
support, configuration, data, parallelism and operations. The dated
[consolidation record](measurements/sharing-20260913/README.md#consolidation-decisions-september-13)
tracks which branches supplied the current implementation.
