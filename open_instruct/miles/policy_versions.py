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
        if not isinstance(entries, (list, tuple)):
            raise ValueError("Every generation call must carry a policy span list")
        for entry in entries:
            entry = asdict(entry) if is_dataclass(entry) else entry
            if not isinstance(entry, dict) or set(entry) != {"version", "abs_start", "abs_end"}:
                raise ValueError("Invalid serialized policy span")
            start, end = entry["abs_start"], entry["abs_end"]
            if type(start) is not int or type(end) is not int or not previous_end <= start <= end:
                raise ValueError("Invalid or overlapping absolute policy spans")
            version = version_number(entry["version"])
            if start == end:
                continue
            result.append({**entry, "version": version})
            previous_end = end
    return result


def versions(calls, *, allow_empty=False):
    if not isinstance(calls, (list, tuple)):
        raise ValueError("Every sample must carry its behavior policy version")
    if all(isinstance(value, (int, str)) for value in calls):
        result = [version_number(value) for value in calls]
    else:
        result = [span["version"] for span in spans(calls)]
    if not result and not allow_empty:
        raise ValueError("Every sample must carry its behavior policy version")
    return result
