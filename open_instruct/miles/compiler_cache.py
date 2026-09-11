"""Verified immutable compiler-cache generations and private mutable restores."""

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

SCHEMA = 1
FAMILIES = {
    "triton": "TRITON_CACHE_DIR",
    "tilelang": "TILELANG_CACHE_DIR",
    "torchinductor": "TORCHINDUCTOR_CACHE_DIR",
    "fa4": "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
    "deepep": "EP_JIT_CACHE_DIR",
    "deepgemm": "DG_JIT_CACHE_DIR",
}
HEX = re.compile(r"[0-9a-f]{64}")
SOURCE_SUFFIXES = {".py", ".c", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".so"}
NON_COMPILE_MILES = {
    "hf_checkpoint",
    "save",
    "load",
    "prompt_data",
    "eval_prompt_data",
    "save_debug_rollout_data",
    "save_debug_train_data",
    "wandb_project",
    "wandb_group",
    "wandb_entity",
    "wandb_run_name",
}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def source_identity(root):
    files = {}
    for path in sorted(root.rglob("*")):
        if any(part in {".git", "__pycache__", ".venv"} for part in path.relative_to(root).parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Source code symlink requires an explicit resolved source root: {path}")
        if path.suffix in SOURCE_SUFFIXES and path.is_file():
            files[path.relative_to(root).as_posix()] = sha256(path)
    if not files:
        raise ValueError(f"No runtime source files under {root}")
    return {"sha256": digest(files), "files": len(files)}


def fingerprint(*, image, runtime_lock, sources, model_config, run_config, toolchain, compiler_env):
    if not re.fullmatch(r"(?:sha256:[0-9a-f]{64}|[0-9A-HJKMNP-TV-Z]{26})", image):
        raise ValueError("Use an immutable Docker SHA256 or Beaker image ID")
    required = {"olmo-core", "miles", "open-instruct", "olmo-sglang"}
    if not required <= sources.keys():
        raise ValueError(f"Fingerprint requires runtime source trees: {sorted(required)}")
    settings = json.loads(encoded(run_config))
    if settings.get("core", {}).get("model_config"):
        raise ValueError("External Core model-config files are not qualified by this wrapper")
    for key in NON_COMPILE_MILES:
        settings.get("miles", {}).pop(key, None)
    settings.get("core", {}).pop("reward_config", None)
    identity = {
        "schema_version": SCHEMA,
        "backend": "open-instruct-miles-olmo-core",
        "image": image,
        "runtime_lock": runtime_lock,
        "sources": sources,
        "model_config": model_config,
        "compile_settings": settings,
        "toolchain": toolchain,
        "compiler_env": compiler_env,
        "cache_families": FAMILIES,
    }
    return digest(identity), identity


def local_environment(root):
    root = root.resolve()
    if str(root).startswith(("/weka/", "/net/")):
        raise ValueError("Mutable compiler cache must be node-local")
    mount = None
    mount_length = -1
    mountinfo = Path("/proc/self/mountinfo")
    if mountinfo.exists():
        for line in mountinfo.read_text().splitlines():
            fields = line.split()
            path = Path(fields[4].replace("\\040", " "))
            if root.is_relative_to(path) and len(str(path)) > mount_length:
                mount, mount_length = fields[fields.index("-") + 1], len(str(path))
        if mount in {"weka", "wekafs", "nfs", "nfs4", "cifs", "lustre", "ceph", "gpfs"}:
            raise ValueError(f"Mutable compiler cache cannot use shared filesystem {mount}")
    env = {}
    for family, variable in FAMILIES.items():
        directory = root / family
        directory.mkdir(parents=True, exist_ok=False)
        env[variable] = str(directory)
    env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    return env


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Cache contains a link or special file: {path}")
        result[path.relative_to(root).as_posix()] = {
            "sha256": sha256(path),
            "size": info.st_size,
            "executable": bool(info.st_mode & 0o111),
        }
    return result


def relocate_triton(root, destination=None):
    """Canonical relative group paths on publish; private absolute paths on restore."""
    count = 0
    for path in root.rglob("__grp__*"):
        data = json.loads(path.read_text())
        children = data.get("child_paths")
        if set(data) != {"child_paths"} or not isinstance(children, dict):
            raise ValueError("Unknown Triton group metadata format")
        converted = {}
        for name, value in children.items():
            child = Path(value)
            if destination is None:
                if not child.is_absolute() or not child.is_relative_to(root):
                    raise ValueError("Triton group references another cache root")
                relative = child.relative_to(root)
            else:
                if child.is_absolute() or ".." in child.parts:
                    raise ValueError("Unsafe relative Triton cache group")
                relative = child
            if not (root / relative).is_file():
                raise ValueError("Triton group references a missing artifact")
            converted[name] = str(destination / relative) if destination else relative.as_posix()
        path.write_bytes(encoded({"child_paths": converted}))
        count += 1
    return count


def snapshot(source, destination, family):
    before = inventory(source)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    if inventory(destination) != before or inventory(source) != before:
        raise ValueError("Compiler cache changed during snapshot")
    if family == "triton":
        # Relocate copied absolute paths through the original namespace before
        # canonicalizing them; only the documented group JSON is rewritten.
        for path in destination.rglob("__grp__*"):
            data = json.loads(path.read_text())
            if set(data) != {"child_paths"} or not isinstance(data["child_paths"], dict):
                raise ValueError("Unknown Triton group metadata format")
            changed = {}
            for name, value in data["child_paths"].items():
                original = Path(value)
                if not original.is_absolute() or not original.is_relative_to(source):
                    raise ValueError("Triton group references another cache root")
                changed[name] = str(destination / original.relative_to(source))
            path.write_bytes(encoded({"child_paths": changed}))
        relocate_triton(destination)
    marker = str(source.parent).encode()
    for path in destination.rglob("*"):
        if not path.is_file():
            continue
        tail = b""
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                data = tail + chunk
                if marker in data:
                    raise ValueError(f"Unqualified embedded private-cache path: {family}/{path.name}")
                tail = data[-len(marker) :]


def artifact_root(shared, key, family):
    if not HEX.fullmatch(key) or family not in FAMILIES:
        raise ValueError("Invalid fingerprint or cache family")
    validate_shared_root(shared)
    return shared / f"core-v{SCHEMA}" / key / family


def validate_shared_root(shared):
    """Validate retention policy before creating reports or cache artifacts."""
    if not shared.is_absolute():
        raise ValueError("Compiler cache root must be an absolute path")
    resolved = shared.resolve()
    if resolved.is_relative_to("/weka") and not any(
        re.fullmatch(r"tmp-[1-9][0-9]*[hdwmy]", part) for part in resolved.parts
    ):
        raise ValueError("WEKA cache root requires an expiry component such as tmp-30d")


def extract_verified(base, key, family, destination, generation=None):
    pointer = base / "CURRENT"
    if generation is None:
        if pointer.is_symlink():
            raise ValueError("Cache pointer must not be a symlink")
        if not pointer.exists():
            return None
        generation = pointer.read_text().strip()
    if not HEX.fullmatch(generation):
        raise ValueError("Invalid cache generation pointer")
    selected = base / "generations" / generation
    if any(path.is_symlink() for path in (selected, selected / "manifest.json", selected / "cache.tar.gz")):
        raise ValueError("Cache generation must not contain symlinks")
    manifest = json.loads((selected / "manifest.json").read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError("Malformed cache manifest")
    for name, record in manifest["files"].items():
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError("Unsafe cache manifest path")
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("sha256"), str)
            or not HEX.fullmatch(record["sha256"])
            or type(record.get("size")) is not int
            or record["size"] < 0
            or type(record.get("executable")) is not bool
        ):
            raise ValueError("Malformed cache file inventory")
    expected = {"schema_version": SCHEMA, "fingerprint": key, "family": family, "generation": generation}
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError("Incompatible cache generation metadata")
    if digest(manifest["files"]) != generation:
        raise ValueError("Cache file inventory digest changed")
    archive = selected / "cache.tar.gz"
    if sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("Cache archive checksum changed")
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or str(path) != member.name or not member.isfile():
                raise ValueError("Unsafe cache archive member")
            if member.name in seen or member.name not in manifest["files"]:
                raise ValueError("Duplicate or unlisted cache archive member")
            seen.add(member.name)
            if member.size != manifest["files"][member.name]["size"]:
                raise ValueError("Cache archive member size changed")
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError("Missing cache archive member contents")
            with stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output)
            target.chmod(0o700 if manifest["files"][member.name]["executable"] else 0o600)
    if inventory(destination) != manifest["files"]:
        raise ValueError("Restored cache file inventory changed")
    return {
        "generation": generation,
        "archive_bytes": archive.stat().st_size,
        "local_bytes": sum(item["size"] for item in manifest["files"].values()),
    }


def restore(shared, local, key, family):
    started = time.monotonic()
    destination = local / family
    if any(destination.iterdir()):
        raise ValueError("Restore requires a fresh private cache directory")
    try:
        with tempfile.TemporaryDirectory(prefix=".restore-", dir=local) as directory:
            stage = Path(directory)
            evidence = extract_verified(artifact_root(shared, key, family), key, family, stage)
            if evidence is None:
                return {"family": family, "status": "miss", "seconds": time.monotonic() - started}
            relocated = relocate_triton(stage, destination) if family == "triton" else 0
            evidence["relocated_groups"] = relocated
            destination.rmdir()
            os.replace(stage, destination)
        return {"family": family, "status": "hit", "seconds": time.monotonic() - started, **evidence}
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        return {"family": family, "status": "rejected", "seconds": time.monotonic() - started, "reason": str(error)}


def publish(shared, local, key, family):
    """Run only after the entire local child process tree has stopped writing."""
    started = time.monotonic()
    base = artifact_root(shared, key, family)
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix=".merge-", dir=base) as directory:
            stage = Path(directory)
            # Corrupt current generations are never overwritten/repaired implicitly.
            extract_verified(base, key, family, stage)
            existing = inventory(stage)
            with tempfile.TemporaryDirectory(prefix=".incoming-", dir=base) as incoming_directory:
                incoming_root = Path(incoming_directory)
                snapshot(local / family, incoming_root, family)
                incoming = inventory(incoming_root)
                for name, record in incoming.items():
                    if name in existing and existing[name] != record:
                        raise ValueError(f"Conflicting compiler-cache key: {family}/{name}")
                    if name not in existing:
                        target = stage / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(incoming_root / name, target)
            files = inventory(stage)
            if not files:
                return {"family": family, "status": "empty", "seconds": time.monotonic() - started}
            generation = digest(files)
            generations = base / "generations"
            generations.mkdir(exist_ok=True)
            target = generations / generation
            if target.exists():
                with tempfile.TemporaryDirectory(prefix=".verify-", dir=base) as verification:
                    extract_verified(base, key, family, Path(verification), generation=generation)
            else:
                with tempfile.TemporaryDirectory(prefix=".publish-", dir=generations) as temporary:
                    prepared = Path(temporary)
                    archive = prepared / "cache.tar.gz"
                    with tarfile.open(archive, "w:gz", compresslevel=1) as output:
                        for name in files:
                            output.add(stage / name, arcname=name, recursive=False)
                    manifest = {
                        "schema_version": SCHEMA,
                        "fingerprint": key,
                        "family": family,
                        "generation": generation,
                        "files": files,
                        "archive_sha256": sha256(archive),
                        "created_unix": time.time(),
                    }
                    (prepared / "manifest.json").write_bytes(encoded(manifest))
                    os.rename(prepared, target)
            current = base / "CURRENT"
            old = current.read_text().strip() if current.exists() else None
            with tempfile.NamedTemporaryFile("w", dir=base, prefix=".CURRENT-", delete=False) as stream:
                stream.write(generation + "\n")
                pointer = Path(stream.name)
            os.replace(pointer, current)
            return {
                "family": family,
                "status": "unchanged" if old == generation else "published",
                "generation": generation,
                "local_bytes": sum(item["size"] for item in files.values()),
                "seconds": time.monotonic() - started,
            }
