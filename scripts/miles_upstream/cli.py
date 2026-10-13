"""Plan, validate, or submit the shared upstream MILES TOML subset."""

import argparse
import json
from pathlib import Path

from scripts.miles_upstream import config, launch

from open_instruct.miles.errors import InputError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "validate", "run"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--schema-dir", type=Path, default=Path("/opt/miles-upstream/schemas"))
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--image", help="Immutable Beaker application image ID")
    parser.add_argument("--receipt", type=Path, default=Path("runs/miles-upstream-launch.json"))
    parser.add_argument("--dry-run", action="store_true", help="Render the Beaker specification without submitting")
    args = parser.parse_args()
    try:
        document, path = config.read(args.config, args.set)
        schema_path = args.schema_dir / f"{config.backend(document)}.json"
        if not schema_path.is_file():
            raise InputError(f"Missing {schema_path}; export schemas from the application image and pass --schema-dir")
        plan = config.compile_config(document, path, schema_path)
        if args.action == "plan":
            print(json.dumps(plan, indent=2))
        elif args.action == "validate":
            launch.launch_settings(plan)
            print("Upstream TOML and single-node allocation are valid; runtime/model checks still run in the job.")
        else:
            if not args.image:
                raise InputError("run requires --image with an immutable Beaker application image ID")
            if args.dry_run:
                print(json.dumps(launch.specification(args.image, plan), indent=2))
            else:
                launch.submit(args.image, plan, args.receipt)
    except (InputError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
