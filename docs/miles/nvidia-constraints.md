# NVIDIA constraint-source trials

The optional `nvidia_constraints.CalendarVerifier` and
`nvidia_constraints.InstructionVerifier` factories accept explicitly prepared
calendar and NeMo instruction-following targets. Existing configs are unchanged.
The IF adapter uses `open_instruct.IFEvalG.instructions_registry`, the same maintained
checkers used by `IFEvalVerifier`; it does not install a separate NeMo checker library.
Build its optional image from a committed MILES application image using
`scripts/data/rl_readiness/Dockerfile.nvidia` with the repository root as context.
The pinned binary runtime is preserved.

Calendar targets are nonempty event dictionaries with duration, time windows and
an optional at/before/after/between constraint. Reward requires every event exactly
once, correct durations, valid windows and no overlap. Event names are not graded.
Only independently satisfiable targets are admitted by preparation.

Instruction targets have `family: nemo` and a `target` containing parallel
`instruction_id_list` and `kwargs` arrays. Preparation must exercise positive and
negative controls for each admitted checker, preserve explicit parameters, and
reject unsupported or randomly repaired targets. The scorer also applies the
existing unsuitable-IF-target policy. Language classification and the two counting
families excluded during initial qualification remain excluded; enabling them needs
separate qualification. The schema tag `family: nemo` identifies the prepared labels,
not a separate scoring backend. Reward is all-constraints-pass unless the target
explicitly requests `grading_mode: fraction`; the existing `IFEvalVerifier` always
uses fractional aggregation. This adapter preserves that explicit target contract
and stricter validation without duplicating the constraint implementations.

The existing `RepeatPhraseChecker` already requires one interior-word change in
**every** repetition (fixed in #1044). Regression tests cover both this checker and
the adapter, including unchanged repetitions, mixed unchanged/changed repetitions
in both orders, wrong counts, and multiple word changes. Unexpected checker return
types raise with checker identity and return type rather than silently becoming
zero rewards. The old external NeMo pin had regressed this fix; it is removed.

The registered reward wrapper handles thinking extraction and truncated-response
zeros. Invalid targets raise rather than become incorrect model answers. These
contracts measure constraint compliance, not factual quality or endpoint transfer.
The optional image includes the readiness modules so qualification runs through
the same registered reward and mixture-loader paths as training.
