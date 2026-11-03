# NVIDIA constraint-source trials

The optional `nvidia_constraints.CalendarVerifier` and
`nvidia_constraints.InstructionVerifier` factories accept explicitly prepared
calendar and NeMo instruction-following targets. Existing configs are unchanged.
The instruction library is pinned in `scripts/data/rl_readiness/requirements.txt`.
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
existing unsuitable-IF-target policy. Language classification and the pinned
library's known outdated letter/lowercase counters are excluded.

`copy:repeat_phrase` uses a local correction: every repetition must preserve the
phrase length and endpoints and replace exactly one interior word. The pinned
native checker falls through to `None` for unchanged repetitions and checks the
one-word-change requirement only on its last repetition. The correction retains
its case-sensitive, single-line span extraction, with literal endpoint words.
Regression controls cover unchanged repetitions, mixed unchanged/changed
repetitions in both orders, wrong counts, and multiple word changes. Other
unexpected checker return types still raise, with the checker name and return
type in the diagnostic; they are not silently converted to zero rewards.

The registered reward wrapper handles thinking extraction and truncated-response
zeros. Invalid targets raise rather than become incorrect model answers. These
contracts measure constraint compliance, not factual quality or endpoint transfer.
The optional image includes the readiness modules so qualification runs through
the same registered reward and mixture-loader paths as training.
