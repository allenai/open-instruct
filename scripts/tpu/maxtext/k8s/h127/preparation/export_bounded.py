"""Run the pinned Olmo 3.5 exporter without duplicate full-precision layout copies.

Usage: export_bounded.py PARAMS HF_REFERENCE OUTPUT ATOMIC_TOKENIZER [--check-base]
The reference may contain reference-shapes.json instead of its original weights.
"""

import gc
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

PINNED_SHA256 = "e1b5f0ac7c9326d3d822cb6963c9c35498ee94bb3ddfc760f4928040bde88944"


def tensor_hashes(path):
    with path.open("rb") as stream:
        header_bytes = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_bytes))
        result = {}
        for key, info in sorted(
            ((key, info) for key, info in header.items() if key != "__metadata__"),
            key=lambda item: item[1]["data_offsets"][0],
        ):
            assert info["dtype"] == "BF16", (key, info["dtype"])
            begin, end = info["data_offsets"]
            stream.seek(8 + header_bytes + begin)
            remaining = end - begin
            digest = hashlib.sha256()
            while remaining:
                chunk = stream.read(min(remaining, 16 * 1024 * 1024))
                assert chunk, "Truncated safetensors file"
                values = np.frombuffer(chunk, dtype="<u2")
                assert not np.any((values & 0x7F80) == 0x7F80), f"Nonfinite BF16 tensor: {key}"
                digest.update(chunk)
                remaining -= len(chunk)
            result[key] = digest.hexdigest()
    return result


def main():
    assert len(sys.argv) in (5, 6), __doc__
    params, reference, output, tokenizer = sys.argv[1:5]
    check_base = len(sys.argv) == 6
    assert not check_base or sys.argv[5] == "--check-base"
    reference, output, tokenizer = map(Path, (reference, output, tokenizer))
    source_path = Path(__file__).with_name("export_maxtext_to_hf.py")
    source = source_path.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == PINNED_SHA256
    replacements = {
        "ckpt = ocp.PyTreeCheckpointer()": "ckpt = ocp.PyTreeCheckpointer(ocp.PyTreeCheckpointHandler(restore_concurrent_gb=2))",
        "np.take(x, r, axis=1)": "np.moveaxis(x, 1, 0)[r]",
        "T = lambda x: np.ascontiguousarray(np.asarray(x).T)": "T = lambda x: np.asarray(x).T",
    }
    for before, after in replacements.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    # Read the exact same target shape table captured from the source HF model,
    # without requiring a second copy of its 25 GB weights beside the metadata.
    begin = source.index('index_path = os.path.join(ref, "model.safetensors.index.json")')
    end = source.index('with open(os.path.join(ref, "config.json"))', begin)
    if (reference / "reference-shapes.json").exists():
        source = (
            source[:begin] + 'target = json.load(open(os.path.join(ref, "reference-shapes.json")))\n' + source[end:]
        )
    namespace = {"__name__": "__main__", "__file__": str(source_path)}
    exec(compile(source, str(source_path), "exec"), namespace)
    namespace.clear()
    gc.collect()
    hashes = tensor_hashes(output / "model.safetensors")
    if check_base:
        expected = json.loads((reference / "expected-seeded-bf16-sha256.json").read_text())
        assert hashes == expected, "Base roundtrip does not match independent source HF weights"
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy2(tokenizer / name, output / name)
    # Remove only validation metadata just copied into this new export; keep the
    # original reference files intact and put this export's receipt below.
    for name in (
        "reference-shapes.json",
        "expected-seeded-bf16-sha256.json",
        "metadata-receipt.json",
        "tokenizer-export-audit.json",
        "_HERO_CONVERSION_SUCCESS.json",
    ):
        (output / name).unlink(missing_ok=True)
    generation_path = output / "generation_config.json"
    generation = json.loads(generation_path.read_text()) if generation_path.exists() else {}
    generation.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=0, eos_token_id=[100257, 100265])
    generation_path.write_text(json.dumps(generation, indent=2))
    with (output / "model.safetensors").open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = {
        "source_parameters": params,
        "reference": str(reference),
        "exporter_source_commit": "allenai/open-instruct@268c124abd07c1f1addc83a47729b79ab53aed5c",
        "exporter_sha256": PINNED_SHA256,
        "tensor_count": len(hashes),
        "all_bf16_values_finite": True,
        "reference_only_audit_files_removed": True,
        "model_safetensors_sha256": model_hash,
        "independent_base_hashes_match": check_base,
        "changes": "Bound restore concurrency; use layout views until BF16 materialization; retain the registered atomic tokenizer",
        "stop_tokens": [100257, 100265],
    }
    (output / "export-receipt.json").write_text(json.dumps(receipt, indent=2))
    print("EXPORT-VERIFIED", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
