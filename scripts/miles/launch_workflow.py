"""Submit a structured run file through the committed-image build wrapper."""

import argparse
import json
from pathlib import Path

from open_instruct.miles import launch
from open_instruct.miles.run_spec import RunSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--render-only", action="store_true")
    options = parser.parse_args()
    spec = (
        RunSpec.from_dict(
            json.loads(options.config.read_text()), config_path=options.config, overrides=options.overrides
        )
        if options.config.suffix == ".json"
        else RunSpec.load(options.config, options.overrides)
    )
    if options.render_only:
        print(json.dumps(launch.specification(options.image, spec), indent=2))
    else:
        launch.submit(options.image, spec)


if __name__ == "__main__":
    main()
