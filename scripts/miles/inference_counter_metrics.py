"""Compare retained engine counters without treating absent observations as zero."""

import math
import re

COUNTER = re.compile(
    r"^(sglang:(?:realtime_tokens_total|forward_execution_seconds_total|prefill_effective_tokens_total))"
    r"(\{.*\})?\s+(\S+)(?:\s+\S+)?$"
)


def parse(text):
    values = {}
    for line in text.splitlines():
        match = COUNTER.fullmatch(line)
        if match is None:
            continue
        name, labels, raw = match.groups()
        key = (name, labels or "")
        value = float(raw)
        if key in values or not math.isfinite(value) or value < 0:
            raise ValueError("Engine counter is duplicated, negative or nonfinite")
        values[key] = value
    return values


def analyze(document):
    starts = {row["url"]: row for row in document["measure_start"]}
    ends = {row["url"]: row for row in document["measure_end"]}
    reports = []
    for url in sorted(starts.keys() | ends.keys()):
        first, last = starts.get(url), ends.get(url)
        report = {"url": url, "series": []}
        reports.append(report)
        if first is None or last is None or "error" in first or "error" in last:
            report["missing"] = "Missing or failed boundary scrape"
            continue
        minimum = last["started"] - first["finished"]
        maximum = last["finished"] - first["started"]
        if minimum <= 0 or maximum < minimum:
            raise ValueError("Engine counter scrape timestamps are inconsistent")
        seconds = (minimum + maximum) / 2
        report.update(seconds=seconds, scrape_window_seconds_bounds=[minimum, maximum])
        before, after = parse(first["text"]), parse(last["text"])
        for name, labels in sorted(before.keys() | after.keys()):
            key = (name, labels)
            row = {"name": name, "labels": labels}
            report["series"].append(row)
            if key not in before or key not in after:
                row["missing"] = "Counter absent at one boundary"
            elif after[key] < before[key]:
                row["missing"] = "Counter decreased; possible process restart"
            else:
                delta = after[key] - before[key]
                row.update(
                    delta=delta,
                    per_second=delta / seconds,
                    per_second_scrape_bounds=[delta / maximum, delta / minimum],
                )
    return {
        "engines": reports,
        "scope": "Native scheduler execution counters, including work on unfinished responses. Decode tokens exclude prefill-selected first tokens. Scrape bounds describe timing uncertainty, not statistical confidence. Forward categories are not a complete wall-time partition; absent counters remain unknown.",
    }
