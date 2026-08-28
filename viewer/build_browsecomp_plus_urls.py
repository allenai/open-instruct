from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _urls(documents: Any) -> list[str]:
    if not isinstance(documents, list):
        return []
    return list(
        dict.fromkeys(
            str(document.get("url") or "").strip()
            for document in documents
            if isinstance(document, dict) and str(document.get("url") or "").strip()
        )
    )


def build_mapping(source: Path, destination: Path) -> tuple[int, int]:
    """Write the compact BrowseComp-Plus question-to-source URL mapping."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    missing_positive_urls = 0
    with source.open(encoding="utf-8") as source_file, destination.open("w", encoding="utf-8") as output_file:
        for line_number, line in enumerate(source_file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {source}:{line_number}: {error}") from error
            evidence_urls = _urls(record.get("evidence_docs"))
            positive_urls = _urls(record.get("gold_docs"))
            evidence_set = set(evidence_urls)
            missing_positive_urls += sum(url not in evidence_set for url in positive_urls)
            output_file.write(
                json.dumps(
                    {
                        "query_id": str(record.get("query_id") or ""),
                        "evidence_urls": evidence_urls,
                        "positive_urls": positive_urls,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count, missing_positive_urls


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compact BrowseComp-Plus source URL map")
    parser.add_argument("--input", type=Path, required=True, help="BrowseComp-Plus decrypted JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Destination compact JSONL")
    args = parser.parse_args()
    count, missing = build_mapping(args.input, args.output)
    print(f"Wrote {count} questions to {args.output}; {missing} positive URLs were absent from evidence URLs")


if __name__ == "__main__":
    main()
