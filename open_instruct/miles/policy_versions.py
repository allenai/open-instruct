"""Read native MILES token spans and historical integer-version ledgers."""

from dataclasses import asdict, is_dataclass


def version_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Invalid policy version")
    if isinstance(value, str) and (not value.isascii() or not value.isdigit()):
        raise ValueError("Invalid serialized policy version")
    if int(value) < 0:
        raise ValueError("Invalid negative policy version")
    return int(value)


def spans(calls):
    """Flatten native objects or their wire form, validating absolute positions."""
    result = []
    previous_end = 0
    for call in calls:
        entries = call.spans if hasattr(call, "spans") else call
        if not isinstance(entries, (list, tuple)) or not entries:
            raise ValueError("Every generation call must carry policy spans")
        for entry in entries:
            entry = asdict(entry) if is_dataclass(entry) else entry
            if not isinstance(entry, dict) or set(entry) != {"version", "abs_start", "abs_end"}:
                raise ValueError("Invalid serialized policy span")
            start, end = entry["abs_start"], entry["abs_end"]
            if type(start) is not int or type(end) is not int or not previous_end <= start < end:
                raise ValueError("Invalid or overlapping absolute policy spans")
            result.append({**entry, "version": version_number(entry["version"])})
            previous_end = end
    return result


def versions(calls):
    if not isinstance(calls, (list, tuple)) or not calls:
        raise ValueError("Every sample must carry its behavior policy version")
    if all(isinstance(value, (int, str)) for value in calls):
        return [version_number(value) for value in calls]
    return [span["version"] for span in spans(calls)]
