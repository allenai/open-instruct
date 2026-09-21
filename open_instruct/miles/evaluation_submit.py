"""One isolated Beaker API attempt; the coordinator enforces a process deadline."""

import json
import signal
import sys

from beaker import Beaker


def main():
    path, workspace, name, deadline = sys.argv[1:]
    # The deadline survives trainer exit (the daemon caller may disappear first).
    signal.alarm(int(deadline))
    with Beaker.from_env(default_workspace=workspace, check_for_upgrades=False) as client:
        experiment = client.experiment.create(spec=path, name=name)
        print(json.dumps({"id": experiment.id}))


if __name__ == "__main__":
    main()
