"""Download a generation-pinned GCS model manifest with a short-lived read token.

No bucket listing, credential refresh, or credential output is performed.
Existing files must match; incomplete .part files may be retried.
"""

import argparse
import base64
import concurrent.futures
import datetime
import json
import shutil
from pathlib import Path
from urllib import parse

import google_crc32c
import requests


def checksum(path):
    crc = google_crc32c.Checksum()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            crc.update(chunk)
    return {"size": path.stat().st_size, "crc32c": base64.b64encode(crc.digest()).decode()}


def safe_relative(name):
    path = Path(name)
    assert name and not path.is_absolute() and ".." not in path.parts, "Unsafe object path"
    assert str(path) == name, "Noncanonical object path"
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--credential", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    credential = json.loads(args.credential.read_text())
    prefix = manifest["source_prefix"]
    assert prefix.startswith(credential["allowed_prefix"]) and prefix.endswith("/")
    parsed = parse.urlsplit(prefix)
    assert parsed.scheme == "gs" and parsed.netloc == "ai2-olmo-dev"
    expires = datetime.datetime.fromisoformat(credential["expires_at"])
    assert (expires - datetime.datetime.now(datetime.timezone.utc)).total_seconds() > 600, (
        "Read token expires too soon"
    )
    entries = manifest["objects"]
    assert len(entries) == 197, "Unexpected model checkpoint object count"
    for name, expected in entries.items():
        safe_relative(name)
        assert expected["generation"].isdigit() and expected["size"] >= 0
    args.destination.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(args.destination).free > 150 * 1024**3, "Need 150 GiB free for staging/export"

    def download(item):
        name, expected = item
        path = args.destination / safe_relative(name)
        wanted = {key: expected[key] for key in ("size", "crc32c")}
        if path.exists():
            assert checksum(path) == wanted, f"Existing object mismatch: {name}"
            return name
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        object_name = parsed.path.lstrip("/") + name
        url = "https://storage.googleapis.com/storage/v1/b/ai2-olmo-dev/o/" + parse.quote(object_name, safe="")
        # Each worker owns its connection; no credential is included in URLs/logs.
        with requests.get(
            url,
            params={"alt": "media", "generation": expected["generation"]},
            headers={"Authorization": "Bearer " + credential["access_token"]},
            stream=True,
            timeout=(30, 180),
            allow_redirects=False,
        ) as response:
            assert response.status_code == 200, f"GCS object HTTP {response.status_code}: {name}"
            with part.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    stream.write(chunk)
        assert checksum(part) == wanted, f"Downloaded object mismatch: {name}"
        part.replace(path)
        return name

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(download, sorted(entries.items())), start=1):
            if i % 20 == 0 or i == len(entries):
                print("VERIFIED-OBJECTS", i, len(entries), flush=True)
    receipt = {
        "source_prefix": prefix,
        "object_count": len(entries),
        "bytes": sum(item["size"] for item in entries.values()),
        "all_sizes_crc32c_and_source_generations_verified": True,
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    args.destination.with_name(args.destination.name + "-download-receipt.json").write_text(
        json.dumps(receipt, indent=2)
    )
    print("DOWNLOAD-VERIFIED", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
