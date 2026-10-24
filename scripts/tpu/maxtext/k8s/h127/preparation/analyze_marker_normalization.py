"""Read-only CPU sensitivity diagnostic with frozen base normalization gains.

This isolates the embedding scale and first two RMSNorms. It is neither a
full-model replay nor evidence that normalization causes the cross-mesh gap.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def norm_and_jvp(x, tangent, gain, epsilon):
    radius = np.sqrt(np.mean(x * x) + epsilon)
    return gain * x / radius, gain * (tangent / radius - x * np.mean(x * tangent) / radius**3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads((args.base / "config.json").read_text())
    epsilon, scale = config["rms_norm_eps"], config["embed_scale"]
    names = ["model.embed_norm.weight", "model.layers.0.pre_attention_layernorm.weight"]
    with safe_open(args.base / "model.safetensors", framework="pt", device="cpu") as f:
        gains = [f.get_tensor(name).double().numpy() for name in names]
    arrays = np.load(args.audit, allow_pickle=False)
    key = "['token_embedder']['embedding'].value"
    # IDs are fixed by update_audit.py; avoid interpreting positional rows silently.
    ids = (29, 198, 524, 771, 14023, 27963, 33413, 36539, 100256, 100257, 100264, 100265, 100266, 100267, 100270)
    assert arrays["rows_before/" + key].shape == (len(ids), config["hidden_size"])

    def compose(x, tangent):
        y, dy = norm_and_jvp(x * scale, tangent * scale, gains[0], epsilon)
        z, dz = norm_and_jvp(y, dy, gains[1], epsilon)
        return y, z, dz

    rng = np.random.default_rng(731)
    directions = rng.normal(size=(16, config["hidden_size"]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    records = []
    max_fd_error = 0.0
    for stage in ("rows_before", "rows_after"):
        for token in (100256, 100264, 100265, 100270):
            x = arrays[stage + "/" + key][ids.index(token)].astype(np.float64)
            y, z, jv = compose(x, directions[0])
            # Check the analytic derivative on every recorded input, in FP64.
            h = 1e-12 if np.linalg.norm(x) < 1e-3 else 1e-6
            fd = (
                compose(x + h * directions[0], directions[0])[1] - compose(x - h * directions[0], directions[0])[1]
            ) / (2 * h)
            error = float(np.linalg.norm(fd - jv) / np.linalg.norm(jv))
            assert error < 1e-5, error
            max_fd_error = max(max_fd_error, error)
            records.append(
                {
                    "stage": stage,
                    "token_id": token,
                    "input_norm": float(np.linalg.norm(x)),
                    "after_embedding_norm": float(np.linalg.norm(y)),
                    "after_first_pre_attention_norm": float(np.linalg.norm(z)),
                    "median_unit_direction_jvp_norm": float(
                        np.median([np.linalg.norm(compose(x, v)[2]) for v in directions])
                    ),
                }
            )
    result = {
        "scope": "FP64 isolated normalization calculation with frozen base gains; no BF16 rounding, attention, routing or training. Does not establish TPU parity or causal attribution of the mesh gap.",
        "audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        "gain_tensors": names,
        "epsilon": epsilon,
        "embedding_scale": scale,
        "max_finite_difference_relative_error": max_fd_error,
        "rows": records,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
