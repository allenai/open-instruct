"""Submit a MILES run file like `python -m open_instruct.miles run`, but keep it off specific Beaker nodes.

The launcher has no node exclusion and Beaker makes cluster and hostname constraints mutually exclusive, so this
passes `launch.specification()` a hostname allowlist of every uncordoned node of the run's cluster except
--exclude. Everything else (image resolution, spec, launch receipt) mirrors `open_instruct.miles.execution.launch.submit`.
Run from a clean, committed Open-Instruct checkout.

Usage: python submit_excluding.py IMAGE_ID RUN.toml --exclude holmes-cs-aus-511.reviz.ai2.in [--exclude ...]
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from open_instruct.miles.configuration import specs, topology
from open_instruct.miles.execution import launch, workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("config", type=Path)
    parser.add_argument("--exclude", action="append", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Render and check the spec without submitting")
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=launch.ROOT, text=True).strip():
        raise SystemExit("Commit the current changes before launching (same rule as the MILES wrapper)")
    spec = specs.load(args.config, [])
    cluster = spec.launch["cluster"]
    nodes = json.loads(subprocess.check_output(["beaker", "cluster", "nodes", cluster, "--format", "json"], text=True))
    nodes = nodes if isinstance(nodes, list) else nodes.get("data", nodes)
    hostnames = sorted(n["hostname"] for n in nodes if not n.get("cordoned") and n["hostname"] not in set(args.exclude))
    missing = set(args.exclude) - {n["hostname"] for n in nodes}
    if missing:
        raise SystemExit(f"not nodes of {cluster}: {sorted(missing)}")

    images = json.loads(subprocess.check_output(["beaker", "image", "get", args.image, "--format", "json"], text=True))
    resolved_image = (images[0] if isinstance(images, list) else images)["id"]
    if resolved_image != args.image:
        raise SystemExit("pass an immutable Beaker image ID, not an alias")
    compiled = spec.compile()
    document = launch.specification(resolved_image, spec, hostnames=hostnames, compiled=compiled)
    document["description"] += f" [excluding {', '.join(args.exclude)}]"
    if args.dry_run:
        task = document["tasks"][0]
        print(json.dumps({"allowlisted_nodes": len(hostnames), "excluded_present": bool(set(args.exclude) & set(task["constraints"]["hostname"])),
                          "constraints_keys": sorted(task["constraints"]), "context": task["context"], "gpus": task["resources"]["gpuCount"],
                          "budget": document["budget"], "retry": document.get("retry")}, indent=1))
        return
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=launch.ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix="miles-workflow-") as directory:
        path = Path(directory) / "experiment.json"
        workflow.write_json(path, document)
        response = subprocess.check_output(
            ["beaker", "experiment", "create", str(path), "--workspace", spec.launch["workspace"], "--format", "json"], text=True
        )
    experiments = json.loads(response)
    experiment = experiments[0] if isinstance(experiments, list) else experiments
    receipt = dict(
        experiment_id=experiment["id"], image=resolved_image, requested_image=args.image, revision=revision,
        spec_sha256=workflow.fingerprint(spec.to_dict()), spec=spec.to_dict(), allocation=topology.plan(spec, compiled),
        excluded_hostnames=args.exclude,
        placements=[{"task": t["name"], "hostnames": t["constraints"].get("hostname", []), "gpus": t["resources"]["gpuCount"],
                     "replicas": t.get("replicas", 1)} for t in document["tasks"]],
    )
    target = launch.receipt_path(spec)
    if target.exists():
        old = json.loads(target.read_text())
        workflow.write_json(target.with_name(f"{target.stem}-{old['experiment_id']}.json"), old)
    workflow.write_json(target, receipt)
    job = experiment["jobs"][0]["id"] if experiment.get("jobs") else None
    print(json.dumps({"url": f"https://beaker.org/ex/{experiment['id']}", "job": job, "receipt": str(target),
                      "allowlisted_nodes": len(hostnames), "excluded": args.exclude}, indent=2))


if __name__ == "__main__":
    main()
