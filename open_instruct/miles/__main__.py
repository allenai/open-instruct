"""Run with python -m open_instruct.miles {plan,validate,train,run,status} CONFIG.toml."""

import argparse
import importlib
import json
from pathlib import Path

from open_instruct.miles import validation
from open_instruct.miles.config import RunConfig
from open_instruct.miles.errors import InputError
from open_instruct.miles.run_spec import RunSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "validate", "train", "run", "status"))
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a setting with a TOML value; repeatable, quote strings",
    )
    parser.add_argument("--debug", action="store_true", help="Show a full traceback for input errors")
    options = parser.parse_args()
    try:
        execute(parser, options)
    except InputError as error:
        if options.debug:
            raise
        parser.error(f"{options.config}: {error}")


def execute(parser, options):
    payload = validation.read_document(options.config)
    structured = "schema_version" in payload or "model" in payload
    config = (
        RunSpec.from_dict(payload, config_path=options.config, overrides=options.overrides)
        if structured
        else RunConfig.from_dict(payload, options.overrides)
    )
    if options.command == "plan":
        print(json.dumps(config.plan(), indent=2))
        return
    if options.command in ("run", "status"):
        if not structured:
            parser.error("run/status require a schema_version=1 run file; raw configs support plan/validate/train")
        launch = importlib.import_module("open_instruct.miles.launch")
        if options.command == "run":
            launch.run(options.config, options.overrides)
        else:
            print(json.dumps(launch.status(config), indent=2))
        return
    workflow = importlib.import_module("open_instruct.miles.workflow")
    if options.command == "validate":
        if structured:
            config.compile().arguments()
            print("Run schema, topology and MILES/Core options validated; inputs and runtime checked during train")
        else:
            workflow.parse_runtime(config)
            print("MILES arguments validated for OLMo-core")
        return
    if structured:
        workflow.execute(config)
    else:
        workflow.train_config(config)


if __name__ == "__main__":
    main()
