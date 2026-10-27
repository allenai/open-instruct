"""Deterministic missing-patch puzzles with disjoint source-photo partitions.

No model or GPU dependency. Candidate swaps are paired evaluation controls, not
independent photographs. Labels and crop provenance belong only in metadata.
"""

import dataclasses
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PROMPT = (
    "<|image|><|image|><|image|>\n"
    "The first image has a rectangular region blacked out. Which candidate is the exact missing piece?\n"
    "A. the second image\nB. the third image\nReturn only A or B.\nAnswer:"
)
VERIFIERS = {"jigsaw": {"factory": "open_instruct.miles.rewards.jigsaw.JigsawVerifier"}}


def encoded(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pixels_sha256(image: Image.Image) -> str:
    image = image.convert("RGB")
    return sha256(encoded({"mode": "RGB", "size": image.size}) + image.tobytes())


@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 17
    canvas: int = 512
    piece: int = 192
    train_photos: int = 256
    eval_photos: int = 32
    modes: tuple[str, ...] = ("different_photo", "same_photo")
    minimum_candidate_distance: float = 4.0

    def __post_init__(self):
        if any(type(x) is not int for x in (self.seed, self.canvas, self.piece, self.train_photos, self.eval_photos)):
            raise ValueError("Seed, dimensions, and photo counts must be integers")
        if self.seed < 0 or not 16 <= self.piece <= self.canvas // 2 or self.canvas > 2048:
            raise ValueError("Require seed >= 0 and 16 <= piece <= canvas/2 <= 1024")
        if min(self.train_photos, self.eval_photos) < 2:
            raise ValueError("Each source split needs at least two photographs")
        if not self.modes or len(set(self.modes)) != len(self.modes):
            raise ValueError("Distractor modes must be nonempty and unique")
        if set(self.modes) - {"different_photo", "same_photo", "shift"}:
            raise ValueError("Unsupported distractor mode")
        if not 0 < self.minimum_candidate_distance < 255:
            raise ValueError("Candidate distance must be in (0, 255)")


def source_photos(root: Path) -> list[dict]:
    """Deduplicate decoded pixels as well as file bytes, before splitting."""
    photos, seen = [], set()
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png") or path.name.startswith("._"):
            continue
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            pixels = pixels_sha256(image)
            if pixels in seen:
                continue
            seen.add(pixels)
            photos.append({"path": str(path.resolve()), "sha256": sha256(path.read_bytes()), "pixels_sha256": pixels})
    return photos


def split_photos(photos: list[dict], config: Config) -> dict[str, list[dict]]:
    ordered = sorted(photos, key=lambda row: sha256(f"{config.seed}:{row['pixels_sha256']}".encode()))
    count = config.train_photos + config.eval_photos
    if len(ordered) < count:
        raise ValueError(f"Requested {count} distinct photographs, found {len(ordered)}")
    return {"train": ordered[: config.train_photos], "heldout": ordered[config.train_photos : count]}


def canvas(photo: dict, side: int) -> Image.Image:
    path = Path(photo["path"])
    raw = path.read_bytes()
    if sha256(raw) != photo["sha256"]:
        raise ValueError(f"Source photograph changed: {path}")
    with Image.open(io.BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return ImageOps.fit(image, (side, side), method=Image.Resampling.BICUBIC)


def crop(image: Image.Image, xy: tuple[int, int], side: int) -> Image.Image:
    x, y = xy
    return image.crop((x, y, x + side, y + side))


def make_puzzle(photo: dict, other: dict, config: Config, mode: str, index: int) -> tuple[list, dict]:
    """Reject ambiguous crops; inward shifts cannot clamp back onto the answer."""
    rng = random.Random(f"{config.seed}:{photo['pixels_sha256']}:{mode}:{index}")
    image = canvas(photo, config.canvas)
    distractor_canvas = canvas(other, config.canvas) if mode == "different_photo" else image
    hi, size = config.canvas - config.piece, config.piece
    positions = [(0, 0), (hi, 0), (0, hi), (hi, hi), (hi // 2, hi // 2)]
    hole = positions[index % len(positions)]
    correct = crop(image, hole, size)
    for _ in range(128):
        if mode == "shift":
            dx, dy = rng.choice([(size // 4, 0), (-size // 4, 0), (0, size // 4), (0, -size // 4)])
            wrong_xy = (hole[0] + dx, hole[1] + dy)
            if not all(0 <= x <= hi for x in wrong_xy):
                continue
        else:
            wrong_xy = (rng.randint(0, hi), rng.randint(0, hi))
        if mode != "different_photo" and wrong_xy == hole:
            continue
        wrong = crop(distractor_canvas, wrong_xy, size)
        distance = float(np.abs(np.asarray(correct, dtype=np.float32) - np.asarray(wrong, dtype=np.float32)).mean())
        if distance >= config.minimum_candidate_distance:
            puzzle = image.copy()
            puzzle.paste((0, 0, 0), (*hole, hole[0] + size, hole[1] + size))
            return [puzzle, correct, wrong], {
                "hole_xy": hole,
                "distractor_xy": wrong_xy,
                "candidate_mean_absolute_difference": distance,
                "source_image_sha256": sorted(
                    {photo["pixels_sha256"], other["pixels_sha256"]}
                    if mode == "different_photo"
                    else {photo["pixels_sha256"]}
                ),
            }
    raise ValueError(f"Could not construct distinct candidates for {photo['path']} ({mode})")


def save_image(image: Image.Image, root: Path) -> dict:
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    raw = buf.getvalue()
    digest = sha256(raw)
    path = root / "images" / f"{digest}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f"Image artifact collision: {path}")
    path.write_bytes(raw)
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "pixels_sha256": pixels_sha256(image),
        "width": image.width,
        "height": image.height,
    }


def paired_rows(images: list, root: Path, identity: str, provenance: dict, prompt: str = PROMPT) -> list[dict]:
    """Write both answer orientations with identical text and swapped image bytes."""
    descriptors = [save_image(image, root) for image in images]
    if descriptors[1]["pixels_sha256"] == descriptors[2]["pixels_sha256"]:
        raise ValueError("Identical candidates cannot carry different rewards")
    rows = []
    for label, order in (("A", [0, 1, 2]), ("B", [0, 2, 1])):
        ordered = [descriptors[i] for i in order]
        rows.append(
            {
                "input": prompt,
                "label": label,
                "images": [d["path"] for d in ordered],
                "metadata": {
                    "prepared_sample_id": f"{identity}:{label}",
                    "pair_id": identity,
                    "verifiers": [{"name": "jigsaw", "target": label}],
                    "vision": {"schema_version": 1, "prompt_format": "document", "images": ordered},
                    **provenance,
                },
            }
        )
    return rows


def generate(photos: list[dict], root: Path, config: Config) -> dict:
    """Create an immutable pilot; an incomplete output is never silently reused."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    splits = split_photos(photos, config)
    artifacts = {}
    for split, selected in splits.items():
        rows = []
        for index, photo in enumerate(selected):
            for mode in config.modes:
                images, provenance = make_puzzle(photo, selected[(index + 1) % len(selected)], config, mode, index)
                identity = sha256(
                    encoded({"config": dataclasses.asdict(config), "photo": photo["pixels_sha256"], "mode": mode})
                )
                rows.extend(paired_rows(images, root, identity, {"jigsaw_mode": mode, **provenance}))
        random.Random(f"{config.seed}:{split}:order").shuffle(rows)
        raw = b"".join(encoded(row) for row in rows)
        (root / f"{split}.jsonl").write_bytes(raw)
        artifacts[split] = {"path": f"{split}.jsonl", "sha256": sha256(raw), "rows": len(rows)}
    (root / "verifiers.json").write_bytes(encoded(VERIFIERS))
    manifest = {"schema_version": 1, "config": dataclasses.asdict(config), "sources": splits, "artifacts": artifacts}
    (root / "jigsaw-manifest.json").write_bytes(encoded(manifest))
    return manifest
