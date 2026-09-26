# MILES package organization — September 25, 2026

This is the code-organization part of phase 1 (prepare MILES for main). It does
not remove the deprecated GRPO implementations or claim that the whole branch is
ready to merge. Dead-code removal, simplification, CI coverage and the broader
documentation review remain separate follow-ups.

## Source and scope

Both repositories use `robertb/miles-polish` in isolated worktrees:

- Open Instruct: parent `08cc88a85`; tested smoke source
  `e99a7b9a0573162c4cbea0cb09de739c77574a9c`.
- AllenAI MILES: parent `9ecde5bc80cdadf2348600c49056449c74fb4ec3`;
  adapter hook update `592905363e9efbbb1e45d7e838ec48b7b716dc69`.
- OLMo-core remains `e505356353aa7ce1f6ff83e24d6eb945f463714e`;
  olmo-sglang remains `5514bf5885e690f7cfcd9cfb08e03111fbd70a78`.

The 72 flat Python files are organized into configuration, execution, training,
rollout, publication, rewards, datasets, evaluation and infrastructure packages.
Only `__init__.py`, `__main__.py` and shared exceptions stay at the root. See the
[package map](../architecture.md#adapter-package-layout).

The move updates Python imports, configured factory strings, worker setup hooks,
subprocess entrypoints, model-backend dispatch, repository-root discovery,
parser assets, tests and source links. The public CLI is unchanged. Fixed-image
historical diagnostic launchers translate their embedded trainer imports back to
the paths present in those immutable images; recorded manifests keep their
original module names and hashes.

The parser snapshot was regenerated against the paired runtime. All 1,465 action
definitions matched; only source provenance changed. Native help was recaptured
from Docker image `sha256:e9309b0d650e4b499d0ae941563e301ee1869542eae6f47c27b0b90646c0dc42`.

## Local validation

| Check | Result |
|---|---|
| Repository Ruff checks and format check | Passed |
| Generated reference check and MkDocs build | Passed; existing unrelated link warnings remain |
| Import every adapter module in the built image | All 80 discovered modules imported |
| CPU CLI without site-packages | 8 passed: plan and validate for all four maintained tiers |
| CLI/options/documentation/launcher checks | 74 passed |
| Paired-runtime hook resolution | 3 passed: barrier, refresh and engine drain |
| Fixed-image launcher and diagnostic checks | 32 passed |
| MILES fork trainer/argument tests | 247 passed |
| Full Open Instruct host suite | 1,588 passed, 20 skipped, 13 failed; no collection errors with pinned Core source |
| Host fixture follow-up | All 194 tests in the affected files passed after hydrating and verifying the four LFS fixtures |
| Broad suite in the MILES runtime | 1,946 passed, 60 skipped, 20 failed, 28 collection errors before final corrections |
| Corrected runtime regression checks | 61 passed |
| Full MILES fork fast suite | 9,254 passed, 54 skipped, 5 xfailed, 4 failed, 6 collection errors |
| Fork environment follow-up | 223 passed, 1 failed with Git metadata mounted and optional Tinker SDK installed without replacing runtime dependencies |

Counts overlap across suites and follow-ups; they are not an aggregate total.
The broad runs are not clean passes. Their unresolved results are retained below.

The original checkout reproduced 18 of the 20 broad runtime failures. The two
move-induced failures were historical source links and a frozen diagnostic's
manifest-path assertion; both were corrected and rechecked. The pre-existing
parser-snapshot revision mismatch was also corrected after checking parser-action
equality. Twelve of the 13 host failures were unhydrated JSON LFS fixtures; the
remaining readiness-service failure reproduced on the original checkout.

The four fork failures reproduced in the original pinned runtime. Mounting the
worktree's Git metadata resolved the two Git-inventory failures and the related
collection error. Installing the optional Tinker SDK resolved its four collection
errors. The pinned SGLang still lacks the newer Anthropic conversion API expected
by two fork tests and the GDN LoRA target accepted by another test.

Type checking reported the same 66 diagnostics (including identical diagnostic
message counts) on the starting checkout and reorganized checkout with the host's
installed dependencies. Using the exact Core source reduced the count to 60;
this does not establish a clean type-check gate.

## Beaker validation

Validation runs finished on 2026-09-26. Both training smokes passed on B300s;
the completed GPU suite still has pre-existing test failures and image/test
collection mismatches. This is not a clean merge gate.

Training smoke image: `01M3DMJ5D88G1SEPG7J1EMYKD3`, built through the committed-image
wrapper from Open Instruct `e99a7b9a0` with MILES `592905363`.
The final GPU unit run uses the same image and source revision.

Both training runs copy `configs/miles/examples/small.toml` into ignored `runs/`,
use the qualified tiny model at
`/weka/oe-training-default/robertb/open-instruct/runs/standard-examples-20260918/tiny/hf`,
and request four collections, one trainer GPU and one inference GPU. W&B is
offline; checkpoint saving and HF export are disabled. Refresh uses the MILES
router and six optimizer updates of allowed policy lag.

Each smoke has one Beaker task on one node, two total GPUs, a one-hour minimum
runtime, a two-hour timeout, Holmes placement and the `oe-training-default` WEKA
mount. No multi-node replica group is involved. The GPU unit run has one GPU,
a 30-minute minimum and a one-hour timeout, with no WEKA mount. Rendered and
submitted specs, per-job metadata and scheduler events are retained with the
local run evidence. Latest attempts are selected per task and replica.

Runs:

1. Runtime GPU tests: [Beaker](https://beaker.org/ex/01M3DN7BDTA71BTFPBVV8G6CVQ) — CUDA canary passed; pytest exited 2 with four collection errors.
2. Barrier mechanics: [Beaker](https://beaker.org/ex/01M3DMNGY2M5R3AV72RZTJ6KGJ) — passed, exit 0.
3. Async-refresh mechanics: [Beaker](https://beaker.org/ex/01M3DMNM2TRGWK841Q5M0FKDBW) — passed, exit 0.
4. Runtime GPU tests with continued collection: [Beaker](https://beaker.org/ex/01M3E041TJ98F1H2FW4F199N96) — exit 1: 1,117 passed, 10 skipped, 7 failed, 4 collection errors in 963.91 seconds.

Both smoke workflows report `complete`, with rollout IDs 0–3, optimizer steps
1–4 (none skipped), four post-update publications and three evaluations. Every
driver stage reports success, including cleanup and the refresh run's final
generation drain. Barrier behavior versions advance 0–3; refresh consumes
version-0 samples while optimizer/publication steps advance. The recorded policy
objectives were zero, and barrier gradient/update diagnostics were zero. These
checks establish execution mechanics, not nonzero learning or mixed-version
responses. Checkpoint and export qualification were intentionally out of scope.

The GPU test collection errors come from tests importing `core_checkpoint_stream`,
`gsm8k_test_eval` and `original_baseline`, which the existing Docker ignore file
excludes as local-only experiment helpers. The follow-up kept those errors
visible with `--continue-on-collection-errors` and executed the remaining tests
using the same immutable image. It did not treat omitted helpers as passing.
All seven test failures match both test names and failure causes reproduced on
the original checkout: one stale `driver.asyncio` mock, one trainer mock missing
`model`, two router mocks missing `host`, and three router log-capture assertions.
No additional failing test was found in this run. JUnit output and complete
logs are retained in the Beaker results and local run evidence.

The initial Holmes GPU-unit submission `01M3DM77A6H284KEH4R17MTQH2` was
canceled while still queued. A one-GPU Saturn compatibility attempt
[failed before tests](https://beaker.org/ex/01M3DN3Q2E5WHBE5Q6WD5J533H):
its driver reports CUDA 12.8 and cannot initialize PyTorch 2.13.0+cu130.
No training ran there; the final test job returned to Holmes. Training and GPU
qualification are restricted to H100s or B300s per the project owner; Saturn is
not a fallback for these runs. All final submissions were constrained to
Holmes and ran on B300 GPUs.

These are MILES runtime tests and training exercises, not the general Open
Instruct `scripts/test/run_gpu_pytest.sh` experiment. They must not be used as a
`GPU_TESTS=` CI override for that separate suite.

## Packaging follow-up — September 26

Removed the Docker exclusions for `core_checkpoint_stream.py`, `gsm8k_test_eval.py`
and `original_baseline.py` in source commit `f692e8780`. These three helpers total
63,435 bytes. They are dependencies of test files already copied into the image;
excluding them saved negligible space and prevented those files from collecting.
Standalone learning-probe exclusions remain unchanged.

The locally rebuilt image
`sha256:4b12d0cbf22c39b0a58f22325b2ae5b3c58c382e9ded8d1af78bca032844acde`
collects 1,201 tests without the four helper-import errors. This check uses the
same pre-existing exclusion of `test_core_policy_contract.py`, whose optional
comparison package is absent from the MILES image. No source-file mounts were
used for the rebuilt-image verification.

Running the four affected files locally in that rebuilt image gives **58 passed,
9 failed, no collection errors**. The checkpoint-drift, engine-drain learning-audit
and GSM8K evaluator files all pass. The nine historical-baseline failures are the
previously observed environment requirements: seven need the exact frozen
`/stage/open_instruct/grpo_fast.py`, and two need DeepSpeed. Including the helper
source fixes collection; it does not supply that historical trainer environment.

Ruff checks passed and the runtime-source tests passed (8 tests). Evidence is in
`runs/polish/helpers-built-image-tests.{log,xml}` and
`runs/polish/helpers-built-image-collection.log`. The earlier Beaker results above
remain evidence for their original immutable image; no GPU rerun was needed for
this source-file packaging correction.

## Follow-up gates before merge

- Reconcile the readiness-service exhaustion expectation with the current
  verifier failure policy.
- Update stale runtime-test mocks for driver timeouts, trainer model context and
  router host/logging behavior; these failures also occur before the move.
- Separate historical baseline tests requiring `/stage/open_instruct/grpo_fast.py`,
  DeepSpeed or the optional `olmo_miles.evaluation.policy_contract_schema` from
  the standalone MILES image contract.
- Align the fork's CPU test dependencies with the pinned SGLang surface and make
  Git metadata / optional SDK requirements explicit in test runners.
- Resolve existing type-check diagnostics and review default CPU/GPU CI coverage.

Local detailed evidence is under `runs/polish/` in the Open Instruct polish
worktree. Hydrated fixtures were hash-verified for testing and restored to their
tracked LFS pointers; no fixture contents or training defaults changed.
