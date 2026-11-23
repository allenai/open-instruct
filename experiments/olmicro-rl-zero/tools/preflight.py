"""Check what `open_instruct.miles plan/validate` does not: inputs exist, mounts cover them, output is fresh.

Usage: python preflight.py RUN.toml [--image BEAKER_IMAGE]
Exits nonzero if any check fails. Reads secret names only, never values.
"""

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--image", default="01M6HMYBFW8D9ZK4SKGJ1XD6HW")
    args = parser.parse_args()
    spec = tomllib.loads(args.run.read_text())
    failures = []

    def check(ok, message):
        print(("ok    " if ok else "FAIL  ") + message)
        if not ok:
            failures.append(message)

    model = Path(spec["model"]["source"])
    check((model / "config.json").is_file(), f"model config.json: {model}")
    weights = sorted(model.glob("*.safetensors"))
    check(bool(weights) and all(w.resolve().is_file() for w in weights), f"model safetensors resolve: {[w.name for w in weights]}")
    check((model / "tokenizer.json").is_file(), "model tokenizer.json")
    if (model / "SHA256SUMS").is_file():
        listed = {line.split()[-1] for line in (model / "SHA256SUMS").read_text().splitlines() if line.strip()}
        check(all((model / name).is_file() for name in listed), "every SHA256SUMS entry present (run `sha256sum -c` for content)")

    core = Path(spec["core"]["model_config"])
    check(core.is_file(), f"core.model_config: {core}")
    if core.is_file():
        document = json.loads(core.read_text())
        hf = json.loads((model / "config.json").read_text())
        model_cfg = document.get("model", document)
        dims = (model_cfg.get("n_layers"), model_cfg.get("d_model"), model_cfg.get("vocab_size"))
        check(dims == (hf["num_hidden_layers"], hf["hidden_size"], hf["vocab_size"]), f"core vs HF n_layers/d_model/vocab: {dims}")

    data = spec["data"]
    paths = [data["prompt_data"], data["reward_config"], *data.get("eval_prompt_data", [])[1::2]]
    env = spec.get("launch", {}).get("env", {})
    smoke = env.get("OI_NATIVE_LIFECYCLE_SMOKE") == "1"
    if smoke:
        paths.append(env["OI_NATIVE_SMOKE_EVAL_SOURCE"])
    root = Path(spec["output"]["root"])
    for path in paths:
        # The lifecycle smoke writes its eval copy under output.root at runtime.
        if smoke and Path(path).is_relative_to(root):
            continue
        check(Path(path).is_file(), f"data: {path}")
    if data.get("reward_config"):
        registry = json.loads(Path(data["reward_config"]).read_text())
        for path in [p for p in paths if p.endswith(".jsonl") and Path(p).is_file()]:
            names = {v["name"] for line in Path(path).read_text().splitlines() for v in json.loads(line)["metadata"]["verifiers"]}
            check(names <= set(registry), f"verifier names {sorted(names)} registered for {Path(path).name}")

    check(not root.exists(), f"output.root does not exist yet: {root}")

    mounts = [m["mount_path"] for m in spec["launch"].get("weka_mounts", [])]
    for path in [model, core, root, *map(Path, paths)]:
        check(any(Path(path).is_relative_to(m) for m in mounts), f"mounted: {path}")

    workspace = spec["launch"]["workspace"]
    listing = subprocess.run(["beaker", "secret", "list", "--workspace", workspace], capture_output=True, text=True, timeout=60)
    names = {line.split()[0] for line in listing.stdout.splitlines()[1:] if line.strip()}
    for secret in spec["launch"].get("secrets", {}).values():
        check(secret in names, f"secret {secret} in {workspace}")
    image = subprocess.run(["beaker", "image", "get", args.image], capture_output=True, text=True, timeout=60)
    check(image.returncode == 0, f"image readable: {args.image}")

    print(f"\n{len(failures)} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
