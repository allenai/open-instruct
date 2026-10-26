"""Render each planning warning once, consistently across CLI commands."""

import sys
import textwrap


def emit(plan):
    """Keep structured JSON on stdout and human-readable advisories on stderr."""
    for warning in plan.get("launch_warnings", plan.get("warnings", [])):
        print(f"Warning: {warning}", file=sys.stderr)
    runtime = plan.get("runtime") or plan
    for warning in (runtime.get("async_capacity") or {}).get("warnings", []):
        print(f"Warning: {warning}", file=sys.stderr)
    for section in ("throughput", "inference_profile"):
        for warning in (runtime.get(section) or {}).get("warnings", []):
            print(f"Warning [{warning['code']}]: {warning['message']}", file=sys.stderr)

    tips = (runtime.get("inference_profile") or {}).get("tips", [])
    if tips:
        print("+-- Configuration tips (settings unchanged) --", file=sys.stderr)
        for tip in tips:
            for line in textwrap.wrap(tip["message"], width=88):
                print(f"| {line}", file=sys.stderr)
        print("+--------------------------------------------", file=sys.stderr)
