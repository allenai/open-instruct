# Internal sharing candidate

Use the MILES path for new GRPO runs. This candidate consolidates the verified
project work for a small colleague pilot; [support and boundaries](feature-parity.md)
remain model/topology-specific. Long baselines should use the same immutable
candidate image, with each run's configuration and checkpoint identity retained.

## Consolidation decisions, September 13

| Repository / branch | Decision |
|---|---|
| Open Instruct `robertb/miles-colleague-exercises` (`095e2f301`) | Incorporated: managed-judge host networking, debug-stop/export resume fix, readiness fixtures, tests and retained audits |
| Open Instruct `robertb/miles-engine-drain` (`2afb99e41`) | Incorporated: qualified opt-in engine drain, immutable staging/transfer, resume and failure checks; barrier remains the default |
| Open Instruct `robertb/miles-sanity-plan` (`0878b8bcd`) | Historical proposal archived under plans; it does not prescribe current defaults |
| Cache publication, GRPO deprecation/docs, dense Olmo 3, publication profiling, sequence packing branches | Already incorporated before this consolidation; no second merge required |
| MILES `robertb/engine-drain` (`bc582bc5c`) | Companion runtime hook pinned by this candidate; ordinary producer selection stays unchanged |
| OLMo-core `robertb/miles-rl-adapter` (`3d35ab326`) | Current primary pin; no additional Core merge needed |
| olmo-sglang `robertb/miles-serving` (`02ccb5d`) | Current primary pin; no additional serving merge needed |
| Open Instruct/MILES policy-refresh feature branches | Held: serving-only checks passed, but the recorded full training gate failed; subsequent repairs require their own qualification |
| `robertb/miles-throughput-profiles` | Owned by the parallel configuration/measurement work; inherits experimental refresh and is not automatically included |
| Old gdn2 prototype worktrees | Superseded lineage, not outstanding changes to merge into the hero-HF-based adapter |

The consolidation passed its gates and is promoted locally from
`robertb/miles-sharing-candidate` to `robertb/miles-olmo-core`. Source distribution and remote
branch availability are separate from the Beaker image; use the exact candidate
checkout supplied for the pilot. Do not assume a remote branch already includes
unpublished local commits.

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

## Tester workflow

1. Obtain the matching source bundle or supplied candidate checkout, then follow the
   [laptop/session setup](launching.md). Sibling development worktrees are unnecessary.
2. Copy a [structured example](../../configs/miles/examples/README.md) outside the
   checkout; choose a supported checkpoint, accessible mounts and a fresh run root.
3. Run `plan` and `validate`, then set `MILES_EXISTING_IMAGE` to the immutable
   candidate ID and invoke `python -m open_instruct.miles run /path/to/run.toml`.
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

We have successful short lifecycle and learning-path evidence, not evidence of
frequent unexplained failures on those supported configurations. Tonight's long
dense/light-SFT/broader-SFT baselines should test endurance and learning on this
same candidate. Automatic distributed failure recovery remains a separate feature;
its absence is not, by itself, a reason to defer a bounded internal pilot.
