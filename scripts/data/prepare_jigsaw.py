"""Download open photographs, create paired jigsaw puzzles, and freeze BLINK eval.

Run as ``python -m scripts.data.prepare_jigsaw --download-pets --with-blink
--cache /path/to/cache --output /path/to/new-dataset``. The original benchmark
order is retained in blink.jsonl; candidate-swap controls go in a separate file.
"""

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from urllib import request

from PIL import Image
from pyarrow import parquet

from open_instruct.miles.datasets import jigsaw

PETS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
PETS_MD5 = "5c4f3ee8e5d25df40f4fd59a7f44e54c"  # Published torchvision resource checksum.
BLINK_REVISION = "a3666eb249237ba3d5eca8db21176cc47967e040"
BLINK_FILE = "Jigsaw/val-00000-of-00001.parquet"
BLINK_URL = f"https://huggingface.co/datasets/BLINK-Benchmark/BLINK/resolve/{BLINK_REVISION}/{BLINK_FILE}"


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        with request.urlopen(url, timeout=60) as response, temporary.open("wb") as destination:
            shutil.copyfileobj(response, destination)
        temporary.rename(path)
    return path


def pets(cache: Path) -> tuple[Path, dict]:
    archive = download(PETS_URL, cache / "images.tar.gz")
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != PETS_MD5:
            raise ValueError("Oxford-IIIT Pets archive checksum mismatch")
    target = cache / "oxford-pets"
    target.mkdir(parents=True, exist_ok=True)
    # Extract only regular JPEGs; ignore resource-fork files, links and other entries.
    with tarfile.open(archive) as bundle:
        for member in bundle:
            name = Path(member.name)
            if not member.isfile() or name.suffix.lower() != ".jpg" or name.name.startswith("._"):
                continue
            if name.parts[:1] != ("images",) or len(name.parts) != 2:
                raise ValueError(f"Unexpected archive path: {member.name}")
            with bundle.extractfile(member) as source, (target / name.name).open("wb") as destination:
                shutil.copyfileobj(source, destination)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return target, {
        "dataset": "Oxford-IIIT Pet",
        "url": PETS_URL,
        "archive_sha256": digest,
        "license": "CC-BY-SA-4.0",
        "attribution": "Parkhi, Vedaldi, Zisserman, and Jawahar. Cats and Dogs. CVPR 2012.",
        "homepage": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
    }


def blink(cache: Path, output: Path) -> dict:
    path = download(BLINK_URL, cache / f"blink-jigsaw-{BLINK_REVISION}.parquet")
    source = parquet.read_table(path).to_pylist()
    original, swapped = [], []
    for index, row in enumerate(source):
        gold = row["answer"].strip().strip("()")
        if gold not in ("A", "B"):
            raise ValueError(f"BLINK row {index}: missing evaluation label")
        images = []
        for number in (1, 2, 3):
            with Image.open(io.BytesIO(row[f"image_{number}"]["bytes"])) as image:
                images.append(image.convert("RGB"))
        prompt = "<|image|><|image|><|image|>\n" + row["prompt"] + "\nReturn only A or B.\nAnswer:"
        # paired_rows takes [puzzle, correct, wrong]; select the benchmark's original orientation afterward.
        if gold == "B":
            images[1], images[2] = images[2], images[1]
        variants = jigsaw.paired_rows(
            images,
            output,
            f"blink-jigsaw:val:{index}",
            {
                "source_dataset": "BLINK-Benchmark/BLINK",
                "source_revision": BLINK_REVISION,
                "source_row": index,
                "source_image_sha256": sorted({jigsaw.pixels_sha256(image) for image in images}),
                "jigsaw_mode": "blink",
            },
            prompt=prompt,
        )
        original.append(variants[gold == "B"])
        swapped.append(variants[gold != "B"])
    artifacts = {}
    for name, rows in (("blink", original), ("blink-swapped", swapped)):
        raw = b"".join(jigsaw.encoded(row) for row in rows)
        (output / f"{name}.jsonl").write_bytes(raw)
        artifacts[name] = {"path": f"{name}.jsonl", "sha256": jigsaw.sha256(raw), "rows": len(rows)}
    return {
        "dataset": "BLINK-Benchmark/BLINK",
        "revision": BLINK_REVISION,
        "split": "Jigsaw/val",
        "parquet_sha256": jigsaw.sha256(path.read_bytes()),
        "artifacts": artifacts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--download-pets", action="store_true")
    source.add_argument("--image-dir", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-blink", action="store_true")
    parser.add_argument("--train-photos", type=int, default=256)
    parser.add_argument("--eval-photos", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; choose a new immutable dataset directory")
    config = jigsaw.Config(seed=args.seed, train_photos=args.train_photos, eval_photos=args.eval_photos)
    root, source = pets(args.cache) if args.download_pets else (args.image_dir, {"dataset": "local"})
    manifest = jigsaw.generate(jigsaw.source_photos(root), args.output, config)
    manifest["source"] = source
    if args.with_blink:
        benchmark = blink(args.cache, args.output)
        manifest["benchmark"] = benchmark
        manifest["artifacts"].update(benchmark["artifacts"])
    (args.output / "jigsaw-manifest.json").write_bytes(jigsaw.encoded(manifest))
    print(json.dumps({"output": str(args.output.resolve()), "artifacts": manifest["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
