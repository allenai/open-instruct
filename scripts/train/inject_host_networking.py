"""Inject network_mode: host into Harbor task compose files."""

import sys
from pathlib import Path

import yaml

task_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/tblite")
count = 0
for env_dir in sorted(task_root.glob("*/environment")):
    f = env_dir / "docker-compose.yaml"
    cfg = yaml.safe_load(f.read_text()) if f.exists() else {}
    cfg.setdefault("services", {}).setdefault("main", {})["network_mode"] = "host"
    f.write_text(yaml.dump(cfg))
    count += 1
print(f"Injected network_mode: host into {count} tasks")
