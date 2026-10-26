"""Render each planning warning once, consistently across CLI commands."""

import sys


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
