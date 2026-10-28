"""Create a frozen ADS table from prompt-aligned vectors and answer NLLs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn import cluster, decomposition, preprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--components", type=int, default=128)
    args = parser.parse_args()
    vectors = np.load(args.directory / "embeddings.npy", allow_pickle=False)
    rows = [json.loads(line) for line in (args.directory / "rows.jsonl").read_text().splitlines()]
    if len(rows) != len(vectors) or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Invalid or misaligned embedding matrix")
    if [r["row_index"] for r in rows] != list(range(len(rows))) or len({r["prompt_id"] for r in rows}) != len(rows):
        raise ValueError("Embedding identities must be unique and positionally aligned")
    pca = decomposition.PCA(n_components=args.components, svd_solver="randomized", random_state=42)
    projected = pca.fit_transform(preprocessing.normalize(vectors))
    km = cluster.KMeans(n_clusters=args.clusters, random_state=42, n_init="auto")
    assignments = km.fit_predict(projected)
    for row, assignment in zip(rows, assignments, strict=True):
        row["cluster_id"] = int(assignment)
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    (args.directory / "ads-table.jsonl").write_bytes(raw)
    np.savez(
        args.directory / "clustering.npz",
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        centers=km.cluster_centers_,
        projected=projected,
    )
    summary = {
        "table_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(rows),
        "clusters": args.clusters,
        "components": args.components,
        "seed": 42,
        "normalization": "L2 before PCA",
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "cluster_sizes": np.bincount(assignments, minlength=args.clusters).tolist(),
    }
    (args.directory / "clustering.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
