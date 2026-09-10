"""Launch bounded math/IF datasource trials through the committed image wrapper."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

COMMAND = """set -euo pipefail
cd /opt/core-rl
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export NCCL_CUMEM_ENABLE=1
export HF_HOME=/tmp/hf-cache
export RUN_ROOT=/weka/oe-training-default/robertb/open-instruct/miles-core-datasources/$BEAKER_EXPERIMENT_ID
mkdir -p /output "$RUN_ROOT"
# Weights and generated responses remain on WEKA. Copy only compact reports.
copy_reports() {
  for task in math ifeval; do
    mkdir -p "/output/$task"
    for name in preparation.json arguments.json audit.json; do
      if [ -f "$RUN_ROOT/$task/$name" ]; then
        cp "$RUN_ROOT/$task/$name" "/output/$task/"
      fi
    done
  done
  for name in descriptor.json verifier-fixtures/fixtures.json; do
    if [ -f "$RUN_ROOT/$name" ]; then cp "$RUN_ROOT/$name" /output/; fi
  done
}
trap copy_reports EXIT
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), torch.version.cuda)'
python scripts/miles/preflight_attention.py --backend flash_4
python -m scripts.miles.datasource_trials fixtures "$RUN_ROOT/verifier-fixtures"
python - <<'PY'
import hashlib
import json
import os
import struct
from pathlib import Path

from scripts.miles import sft_gsm8k

root = Path(os.environ["RUN_ROOT"])
source = sft_gsm8k.HF_SOURCE
assert (source / "config.json").is_file()
shards = sorted(source.glob("*.safetensors"))
assert shards
raw = sft_gsm8k.TEMPLATE_SOURCE.read_bytes().removesuffix(b"\\n")
assert hashlib.sha256(raw).hexdigest() == sft_gsm8k.TEMPLATE_SHA256
hf = root / "hf"
hf.mkdir()
for path in source.iterdir():
    if path.is_file() and path.name != "chat_template.jinja":
        (hf / path.name).symlink_to(path)
(hf / "chat_template.jinja").write_bytes(raw)
headers = {}
for shard in shards:
    with shard.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        assert 0 < size < shard.stat().st_size - 8
        headers[shard.name] = hashlib.sha256(stream.read(size)).hexdigest()
report = {
    "source": str(source),
    "source_header_sha256": headers,
    "config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
    "template_source": str(sft_gsm8k.TEMPLATE_SOURCE),
    "template_sha256": sft_gsm8k.TEMPLATE_SHA256,
}
(root / "descriptor.json").write_text(json.dumps(report, indent=2) + "\\n")
print("DATASOURCE_HF_DESCRIPTOR_PREPARED", json.dumps(report), flush=True)
PY
for task in __TASKS__; do
  python -m scripts.miles.datasource_trials prepare "$RUN_ROOT/$task" --task "$task" --hf "$RUN_ROOT/hf"
  python -m scripts.miles.datasource_trials validate "$RUN_ROOT/$task"
  python -m scripts.miles.datasource_trials run "$RUN_ROOT/$task"
done
"""


def specification(image, tasks):
    command = COMMAND.replace("__TASKS__", " ".join(shlex.quote(task) for task in tasks))
    return {
        "version": "v2",
        "description": f"Open-instruct / MILES / Core: SFT EP2+SG1, two updates per datasource: {', '.join(tasks)}",
        "tasks": [
            {
                "name": "datasources",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 3, "sharedMemory": "100 GiB"},
                "context": {"priority": "urgent", "minRuntime": "30m", "autoResume": False},
                "constraints": {"cluster": ["ai2/holmes"]},
                "timeout": "90m" if len(tasks) > 1 else "45m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--task", choices=("math", "ifeval"), action="append")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    tasks = list(dict.fromkeys(args.task or ["math", "ifeval"]))
    document = json.dumps(specification(args.image, tasks), indent=2) + "\n"
    if args.render_only:
        print(document, end="")
        return
    with tempfile.TemporaryDirectory(prefix="miles-datasource-trial-") as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
