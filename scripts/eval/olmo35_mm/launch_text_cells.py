"""Launch the recorded GPU text eval cells (olmo-eval on vLLM) on a multimodal SFT checkpoint's hf_text export.

    python launch_text_cells.py HF_TEXT_DIR LABEL --olmo-eval-dir DIR --ledger DIR --cells ifeval gpqa ... \\
        [--cluster ai2/holmes] [--workspace ai2/oe-olmo3p5-mt] [--priority urgent] [--user NAME] \\
        [--plugin-dir DIR] [--specs DIR] [--dry-run]

Each cell's Beaker spec is built by olmo-eval's ``scripts/olmo35_mmsft/text_cells.py`` ``build()`` (allenai/olmo-eval
branch jasonr/olmo35-mmsft-eval, H231/H271's runs used 11dce79286f3787238e4d051e137b534e6b7f82e) from the ledger's
recorded H182 text SFT control specs (``experiments/h182/eval/{specs,holmes-cu130/specs}``): task, sampling, olmo-eval
commit, vLLM and the OLMoE3 plugins stay as recorded; only the model path, names, per-user secrets and placement
change. A refused priority is retried one level lower (urgent -> high -> normal), per cell.

``--plugin-dir`` points the cells' OLMoE3 vLLM/transformers plugin installs at a job-visible copy of this repo's
``scripts/export/olmo35_mm`` instead of the recorded scaling-ladders trees. Off by default: the recorded runtime is
what the GPU arms were scored with, so a TPU-vs-GPU or seed comparison must keep it.

Prints one ``<experiment name>: <id>`` line per cell (``DRYRUN <spec path>`` with --dry-run), the format
run_eval.sh records and compare.py reads.
"""

import argparse
import json
import re
import subprocess
from importlib import util
from pathlib import Path

PRIORITIES = ("urgent", "high", "normal")
# Recorded plugin installs: a requirement `olmoe3-...-plugin @ file:///.../ladders/olmoe3/<pkg>` (vLLM cells) or an
# editable install of an unpacked scaling-ladders archive `-e /tmp/hero-ladder/ladders/olmoe3/<pkg>` (BFCL cells).
PLUGIN_PATH = re.compile(r"(file://)?/[^\s,\"']*?/ladders/olmoe3/(vllm_plugin|transformers_plugin)")


def load_text_cells(olmo_eval_dir: Path):
    path = olmo_eval_dir / "scripts" / "olmo35_mmsft" / "text_cells.py"
    spec = util.spec_from_file_location("olmo_eval_text_cells", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def use_plugin_dir(spec: dict, plugin_dir: str) -> int:
    """Re-point the cell's OLMoE3 plugin installs at ``plugin_dir``/{vllm_plugin,transformers_plugin}; returns hits."""
    hits = 0

    def sub(text: str) -> str:
        nonlocal hits
        new, n = PLUGIN_PATH.subn(lambda m: f"{m.group(1) or ''}{plugin_dir.rstrip('/')}/{m.group(2)}", text)
        hits += n
        return new

    task = spec["tasks"][0]
    task["arguments"] = [sub(a) for a in task["arguments"]]
    # The holmes (cu130) copies also install the plugins in the gantry install command.
    task["envVars"] = [
        {**entry, "value": sub(entry["value"])} if isinstance(entry.get("value"), str) else entry
        for entry in task.get("envVars", [])
    ]
    return hits


def cell_source(ledger: Path, prefix: str, cell: str, cluster: str) -> Path:
    base = ledger / "experiments" / "h182" / "eval"
    holmes = base / "holmes-cu130" / "specs" / f"{prefix}{cell}.json"
    plain = base / "specs" / f"{prefix}{cell}.json"
    # The holmes copies install the B300 cu130 runtime; other clusters take the recorded (jupiter, cu128) spec.
    source = holmes if "ai2/holmes" in cluster.split(",") and holmes.exists() else plain
    if not source.exists():
        raise SystemExit(f"no recorded spec for cell {cell!r} under {base}")
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="hf_text dir, as Beaker jobs see it")
    parser.add_argument("label")
    parser.add_argument("--olmo-eval-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--cluster", default="ai2/holmes")
    parser.add_argument("--workspace", default="ai2/oe-olmo3p5-mt")
    parser.add_argument("--priority", default="urgent", choices=PRIORITIES)
    parser.add_argument("--user", required=True, help="prefix of the <user>_HF_TOKEN / <user>_WANDB_API_KEY secrets")
    parser.add_argument("--plugin-dir", default=None)
    parser.add_argument("--specs", type=Path, default=Path("specs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    text_cells = load_text_cells(args.olmo_eval_dir)
    args.specs.mkdir(parents=True, exist_ok=True)
    for cell in args.cells:
        source = cell_source(args.ledger, text_cells.SOURCE_PREFIX, cell, args.cluster)
        name = f"mmsft-{args.label}-{cell}"
        result = None
        for priority in PRIORITIES[PRIORITIES.index(args.priority) :]:
            spec = text_cells.build(str(source), args.model, name, args.user, args.cluster, priority)
            if args.plugin_dir and use_plugin_dir(spec, args.plugin_dir) == 0:
                raise SystemExit(f"{source}: no OLMoE3 plugin install to re-point")
            path = args.specs / f"{name}.json"
            path.write_text(json.dumps(spec, indent=1))
            if args.dry_run:
                result = f"DRYRUN {path}"
                break
            res = subprocess.run(
                ["beaker", "experiment", "create", str(path), "--workspace", args.workspace, "--name", name],
                capture_output=True,
                text=True,
            )
            out = res.stdout + res.stderr
            ids = re.findall(r"01[0-9A-Z]{24}", out)
            if ids:
                result = ids[0]
                break
            result = "FAILED " + out[-300:].replace("\n", " ")
            if "priorit" not in out.lower():
                break
        print(f"{name}: {result}", flush=True)


if __name__ == "__main__":
    main()
