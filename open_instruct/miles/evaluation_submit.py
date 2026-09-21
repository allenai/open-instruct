"""One isolated Beaker API attempt; the coordinator enforces a process deadline."""

import json
import sys

from beaker import Beaker


def main():
    path, workspace, name = sys.argv[1:]
    with Beaker.from_env(default_workspace=workspace, check_for_upgrades=False) as client:
        experiment = client.experiment.create(spec=path, name=name)
        print(json.dumps({"id": experiment.id}))


if __name__ == "__main__":
    main()
