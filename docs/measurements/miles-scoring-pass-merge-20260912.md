# Scoring-pass merge review

The feature branch `robertb/miles-scoring-pass` at `b7560295c` was reviewed against
primary integration revision `aa7e99ff6`. It contains four commits based on
`7aa5a3d51`; the merge also preserves the later workflow audits and compiler-cache
changes on the primary branch. The subsequent input-validation commit
`855b22a0b` was also incorporated; scoring options now use its shared `InputError`
validation so invalid inputs produce actionable CLI errors.

## GPU evidence supplied with the branch

[Beaker experiment 01M29E275SQQPWQWDDER1KT52P](https://beaker.org/ex/01M29E275SQQPWQWDDER1KT52P)
completed with exit code 0 on Holmes. Its job command chains the test suite,
two-update smoke, and separate-process resume with `&&`.

- Original image suite: **606 passed, 6 skipped**, with six test modules excluded.
- Initial process: optimizer events `checked`, `skipped`.
- Resumed process: appended `checked`, proving the first-process check resets.
- Each checked update reported **25 active tokens, mean absolute difference 0,
  maximum absolute difference 0** between standalone and training scores.
- Serving publication and equality checks remained enabled.

This is tiny-model validation. It establishes neither full-SFT numerical
agreement nor a measured full-SFT speedup. The added merge-review tests below
are separate from this Beaker result; they were not in its image.

## Integration fixes

1. Coordinate rank-local score extraction, shape, and finite-value failures before
   tensor collectives. A two-rank Gloo regression injects shape and NaN faults on
   rank 1; both ranks must fail, then successfully complete a valid check.
2. Restore the historical admission exercise's complete **16 prompts × 4
   responses** configuration. It had inherited only the new starter's prompt
   count, producing 32 samples with a global batch of 64. Maintained researcher
   examples remain **8 × 8**. The historical exercise explicitly requires
   standalone scoring because its timing/replay audit expects that phase on
   every collection.
3. Restore an import mock after the update-zero driver test. Its leaked replacement
   of `importlib.import_module` broke later gradient-capture and Megatron tests.
4. Include debug launch scripts and the Dockerfile in the runtime image, and
   admit those scripts through the Docker build-context allowlist. Existing
   tests inspect/run these files.
5. Correct the live documentation: a rollout-anchor PPO ratio need not be one;
   TIS still compares detached trainer scores against behavior scores when
   standalone scoring is skipped.

## Validation and remaining dependency gap

The original image was independently rerun on CPU with only the policy-contract
module excluded: **15 failed, 641 passed, 32 skipped**. Eleven failures arose from
the incomplete historical batch geometry, two from the leaked import mock, and
two from image packaging. These are not all missing upstream dependencies.

After fixes, the focused runtime suite passed **94 tests, 2 skipped**, including
the distributed regression and all five restored modules. Local workflow/config
checks initially passed **123 tests**; `make style && make quality` passed, including type
checking. Eight additional real-MILES loss tests passed bit-for-bit on loss,
gradients, and metrics with TIS clipping active, both policy anchors, both loss
reductions, and reference KL enabled/disabled. A Docker COPY build using the
runtime allowlist verified the added debug scripts are present in build context.
Runtime validation uses the qualified image with the merged source,
tests, and packaging additions mounted read-only.

The final broader CPU runtime run passed **673 tests, 32 skipped, zero failed**
with only the cross-backend policy-contract module excluded. GPU tests skip on
this CPU rerun; the separate Beaker smoke above supplies the GPU evidence.

After integrating `855b22a0b`, the combined workflow/config checks passed
**205 tests**, including invalid scoring-tolerance inputs. The eight tests in
`open_instruct/test_miles_core.py` passed separately in the pinned image (the
host environment lacks the MoE-v2 Core modules). Lint and type checking passed
on the combined tree. The complete runtime CPU rerun on that combined tree
also passed **673 tests, 32 skipped, zero failures** (247 seconds, with CPU
thread counts bounded to two).

The launcher now excludes only `tests/miles/test_core_policy_contract.py`, which
requires `olmo_miles.evaluation.policy_contract_schema` absent from the pinned
base image. That cross-backend module still needs its matching dependency before
it can join this smoke gate. No blanket exclusions remain for replay, control
exercises, historical data, or update-zero diagnostics.

The rollback switch is `core.scoring_pass_required=true`. Eligible recipes check
the first update of each process and every 50 updates by default, failing when
the global mean absolute difference exceeds `core.scoring_check_tolerance`
(default `1e-3`). Maximum error is reported separately. A passing tolerance gate
is not, in general, a claim of bit equality.
