"""Replay response selection over saved JSONL exports without executing graders.

Rows contain ``full`` (complete raw generated text), plus optional source/set/id/
draw/prompt fields. The compact report compares current-main selection with the
shared contract; it measures input/gate changes, not correctness or score flips.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from open_instruct.miles.rewards import deterministic_verifiers, grading, response_text


def replay(path, *, reasoning_mode="required"):
    digest = hashlib.sha256()
    counts = Counter()
    sources = Counter()
    old_gates = Counter()
    new_gates = Counter()
    examples = []
    with Path(path).open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            raw = row["full"]
            if not isinstance(raw, str):
                raise ValueError("Every full field must be a complete response string")
            clean = response_text.strip_terminal_special_tokens(raw)
            _, separator, previous = clean.rpartition("</think>")
            old_gate = "missing_final_answer" if not separator or not previous.strip() else None
            parts = grading.split_response(raw, reasoning_mode=reasoning_mode, prompt=row.get("prompt"))
            # Exact old common.answer_text implementation, including its fallback.
            previous_legacy = response_text.strip_terminal_special_tokens(raw.rsplit("</think>", 1)[-1]).strip()
            counts["rows"] += 1
            sources[str(row.get("source", "unknown"))] += 1
            old_gates[old_gate or "eligible"] += 1
            new_gates[parts.gate or "eligible"] += 1
            if deterministic_verifiers.answer_text(raw) != previous_legacy:
                counts["legacy_input_changes"] += 1
            if old_gate is None and parts.gate is None and parts.response != previous:
                counts["eligible_input_changes"] += 1
            if old_gate != parts.gate:
                counts["gate_changes"] += 1
                if len(examples) < 5:
                    examples.append(
                        {
                            **{key: row[key] for key in ("set", "source", "id", "draw") if key in row},
                            "previous_gate": old_gate,
                            "new_gate": parts.gate,
                        }
                    )
            if parts.closing_markers > 1:
                counts["multiple_closing_markers"] += 1
    return {
        "schema_version": 1,
        "comparison_revision": "46d2dbbb5bc13baa7c821afb82cd16a95be52e41",
        "measurement": "input-and-gate-selection-only; no grader execution or score/accuracy claim",
        "limitations": (
            "Selection over supplied text; completeness is not independently certified. "
            "No finish-state/truncation gating or token alignment is inferred. Missing prompts cannot certify prefills."
        ),
        "input": str(Path(path).resolve()),
        "input_sha256": digest.hexdigest(),
        "reasoning_mode": reasoning_mode,
        "counts": {
            key: counts[key]
            for key in (
                "rows",
                "legacy_input_changes",
                "eligible_input_changes",
                "gate_changes",
                "multiple_closing_markers",
            )
        },
        "sources": dict(sorted(sources.items())),
        "previous_gates": dict(old_gates),
        "new_gates": dict(new_gates),
        "changed_examples": examples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reasoning-mode", choices=grading.REASONING_MODES, default="required")
    args = parser.parse_args()
    report = replay(args.input, reasoning_mode=args.reasoning_mode)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
