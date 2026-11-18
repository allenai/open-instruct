"""Submit the run file once the committed-image wrapper has selected its image.
This small command-line bridge loads the configuration and overrides, then either
renders the Beaker specification for inspection or passes it to the launch module.
Keeping image construction outside this step allows the same submission code to
use a newly built image or an explicitly reused one.
"""

import argparse
import json
from pathlib import Path

from open_instruct.miles.configuration import specs
from open_instruct.miles.execution import data_preflight, launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--data-path", action="append", default=[], metavar="CONFIGURED_PATH=LOCAL_PATH")
    options = parser.parse_args()
    if options.data_path and not options.check_data:
        parser.error("--data-path requires --check-data")
    spec = (
        specs.from_dict(
            json.loads(options.config.read_text()), config_path=options.config, overrides=options.overrides
        )
        if options.config.suffix == ".json"
        else specs.load(options.config, options.overrides)
    )
    if options.check_data:
        print(json.dumps(data_preflight.check(spec, options.data_path), indent=2))
    if options.render_only:
        print(json.dumps(launch.specification(options.image, spec), indent=2))
    else:
        launch.submit(options.image, spec)


if __name__ == "__main__":
    main()
