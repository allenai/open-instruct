"""Export the exact audited release drop lists for open-instruct issue 1964.

The private audit files stay outside this repository. Pass their common parent
directory with --audit-root; the exported manifest can be used by
prepare_olmo35_release_data.py on original Dolci or an already-cleaned derivative.
"""

import argparse
import hashlib
import json
from pathlib import Path

from pyarrow import parquet as pq

SOURCE_REVISION = "7668c638cc84100951973b456069a1a462d6d915"
EXPECTED_UNION = 24906
BUNDLE = "k3-audit/handoff3/bundle_B_v3_ids.parquet"
EXTRAS = "k3-audit/verify-ho2/u05-identity-stale--mass-and-recall/10_unflagged_identity_misses_ids.parquet"
EVIDENCE = {
    BUNDLE: "3abbbbd590c0b1622ca61ad76537dadf03a7d9ccba814c67428312386fc48f92",
    EXTRAS: "840d8298c9612b52c93079bc8357cd7bed3e9ecc5dfc6a36168d0004c848c2c8",
}
FILTERS = {
    "dated_cutoff": ("b06_dated_cutoff_noidentity", 1021),
    "stale_identity": ("b07_olmo_identity_rows", 58),
    "hidden_safety_prompt": ("b09_safety_overrefusal_hidden", 8323),
    "aya_wrong_language": ("b10_aya_wrong_language", 15593),
}


def export_manifest(audit_root: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for relative, expected in EVIDENCE.items():
        actual = hashlib.sha256((audit_root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Audit evidence changed: {relative} ({actual} != {expected})")
    bundle = pq.read_table(audit_root / BUNDLE, columns=["id", "filter"]).to_pylist()
    groups = {}
    for name, (label, expected_count) in FILTERS.items():
        ids = [row["id"] for row in bundle if row["filter"] == label]
        if len(ids) != expected_count or len(set(ids)) != expected_count:
            raise ValueError(f"Unexpected count or duplicate IDs for {name}")
        groups[name] = sorted(ids)
    extras = pq.read_table(audit_root / EXTRAS, columns=["id"])["id"].to_pylist()
    if len(extras) != 21 or len(set(extras)) != 21:
        raise ValueError("Expected 21 distinct additional stale-identity IDs")
    groups["stale_identity_extra"] = sorted(extras)
    if len(set().union(*map(set, groups.values()))) != EXPECTED_UNION:
        raise ValueError(f"Expected {EXPECTED_UNION} distinct release-drop IDs across all groups")
    manifest = {
        "issue": "https://github.com/allenai/open-instruct/issues/1964",
        "source_dataset": "allenai/Dolci-Think-SFT",
        "source_revision": SOURCE_REVISION,
        "groups": groups,
        "evidence": [{"path": path, "sha256": sha} for path, sha in EVIDENCE.items()],
        "disposition": {
            "dated_cutoff": "Drop; these rows were also included in the separate baseline cleanup.",
            "stale_identity": "Drop originals; replacement identity data is a separate artifact.",
            "stale_identity_extra": "Drop the 21 additional audited Dolci IF rows with incorrect identity facts.",
            "hidden_safety_prompt": "Drop pending regeneration; this manifest does not claim regeneration.",
            "aya_wrong_language": "Drop the precision-verified tight list; no language heuristic is added.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = export_manifest(args.audit_root, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({name: len(ids) for name, ids in manifest["groups"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
