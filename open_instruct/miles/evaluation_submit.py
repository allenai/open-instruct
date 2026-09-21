"""One isolated Beaker API attempt; the coordinator enforces a process deadline."""

import json
import os
import re
import signal
import sys

from beaker import Beaker


def diagnostic(error):
    """Keep useful API errors without persisting credentials echoed by dependencies."""
    message = str(error)
    for name, value in os.environ.items():
        if value and any(part in name.upper() for part in ("TOKEN", "KEY", "SECRET", "PASSWORD")):
            message = message.replace(value, "[REDACTED]")
    message = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", message)
    return {"error": type(error).__name__, "message": message[:2000]}


def main():
    path, workspace, name, deadline = sys.argv[1:]
    # The deadline survives trainer exit (the daemon caller may disappear first).
    signal.alarm(int(deadline))
    try:
        with Beaker.from_env(default_workspace=workspace, check_for_upgrades=False) as client:
            experiment = client.experiment.create(spec=path, name=name)
            print(json.dumps({"id": experiment.experiment.id}))
    except Exception as error:
        print(json.dumps(diagnostic(error)), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
