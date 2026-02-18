#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "tqdm",
#     "click",
#     "huggingface-hub",
#     "luxical",
#     "sentence_transformers",
#     "tqdm",
#     "numpy",
#     "datasets",
#     "scikit-learn",
#     "hdbscan"
# ]
# ///

import dataclasses as dt
from pathlib import Path
from typing import List, Literal

import click
import hdbscan
import numpy as np
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer


@dt.dataclass(frozen=True)
class ThinkMessage:
    thinking: str
    message: str

    @classmethod
    def from_message(cls, message: str) -> "ThinkMessage":
        if "<think>" in message and "</think>" in message:
            start = message.index("<think>") + len("<think>")
            end = message.index("</think>")
            return cls(message[start:end].strip(), message[end + len("</think>") :])
        return cls("", message)

    def __len__(self):
        return len(self.thinking)


def cluster_sentences(embeddings: np.ndarray, min_cluster_size: int = 2, min_samples: int = 2) -> hdbscan.HDBSCAN:
    """Cluster sentence embeddings using HDBSCAN.

    HDBSCAN is chosen because it:
    - Automatically determines the number of clusters
    - Makes no assumptions about cluster shapes or data distribution
    - Handles varying density regions (works well when some paragraphs
      are repetitive while others are unique)
    - Labels outlier points as noise (-1), so unique paragraphs are not
      forced into clusters

    Args:
        embeddings: Array of shape (n_sentences, embedding_dim) from a sentence
            transformer model.
        min_cluster_size: Minimum number of sentences to form a cluster.
            Smaller values detect finer-grained repetition patterns.
        min_samples: Controls how conservative clustering is; higher values
            require denser regions to form clusters.

    Returns:
        Fitted HDBSCAN clusterer with labels_, probabilities_, and
        cluster hierarchy accessible as attributes.
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")
    clusterer.fit(embeddings)
    return clusterer


def compute_repetition_score(labels: np.ndarray, total_paragraphs: int) -> dict:
    """Compute repetition metrics from HDBSCAN cluster labels.

    Args:
        labels: Cluster labels from HDBSCAN (-1 = noise/unique).
        total_paragraphs: Total number of paragraphs in the thinking trace.

    Returns:
        Dictionary with repetition metrics:
        - is_repetitive: Whether the trace is flagged as repetitive.
        - clustered_ratio: Fraction of paragraphs assigned to a cluster (not noise).
        - largest_cluster_ratio: Fraction of paragraphs in the largest cluster.
        - num_clusters: Number of distinct clusters found.
        - cluster_sizes: Sorted list of cluster sizes (largest first).
    """
    non_noise_mask = labels != -1
    clustered_count = int(non_noise_mask.sum())
    clustered_ratio = clustered_count / total_paragraphs if total_paragraphs > 0 else 0.0

    cluster_ids = labels[non_noise_mask]
    if len(cluster_ids) == 0:
        return {
            "is_repetitive": False,
            "clustered_ratio": 0.0,
            "largest_cluster_ratio": 0.0,
            "num_clusters": 0,
            "cluster_sizes": [],
        }

    unique_labels, counts = np.unique(cluster_ids, return_counts=True)
    sorted_counts = sorted(counts.tolist(), reverse=True)
    largest_cluster_ratio = sorted_counts[0] / total_paragraphs

    # Flag as repetitive if a large fraction of paragraphs cluster together,
    # indicating the model is repeating similar sentences/thoughts.
    is_repetitive = clustered_ratio > 0.5 or largest_cluster_ratio > 0.3

    return {
        "is_repetitive": is_repetitive,
        "clustered_ratio": round(clustered_ratio, 3),
        "largest_cluster_ratio": round(largest_cluster_ratio, 3),
        "num_clusters": len(unique_labels),
        "cluster_sizes": sorted_counts,
    }


def embed_dataset(
    batch: dict[str, List], model_name_or_path: str = "DatologyAI/luxical-one", messages_field: str = "messages"
) -> dict[str, List]:
    luxical_one = SentenceTransformer(model_name_or_path=model_name_or_path, trust_remote_code=True)

    results: dict[str, List] = {
        "is_repetitive": [],
        "clustered_ratio": [],
        "largest_cluster_ratio": [],
        "num_clusters": [],
    }

    for idx in range(len(batch[messages_field])):
        assistant_messages = [msg for msg in batch[messages_field][idx] if msg["role"] == "assistant"]
        thinking_traces = [
            parsed for msg in assistant_messages if len(parsed := ThinkMessage.from_message(msg["content"])) > 0
        ]
        if not thinking_traces:
            results["is_repetitive"].append(False)
            results["clustered_ratio"].append(0.0)
            results["largest_cluster_ratio"].append(0.0)
            results["num_clusters"].append(0)
            continue

        # Aggregate repetition across all thinking traces in the example
        any_repetitive = False
        max_clustered_ratio = 0.0
        max_largest_cluster_ratio = 0.0
        total_clusters = 0

        for trace in thinking_traces:
            thinking_paragraphs = [p.strip() for p in trace.thinking.split("\n\n") if p.strip()]
            if len(thinking_paragraphs) < 3:
                continue

            thinking_embeddings = luxical_one.encode(thinking_paragraphs)
            clusterer = cluster_sentences(thinking_embeddings)
            score = compute_repetition_score(clusterer.labels_, len(thinking_paragraphs))

            any_repetitive = any_repetitive or score["is_repetitive"]
            max_clustered_ratio = max(max_clustered_ratio, score["clustered_ratio"])
            max_largest_cluster_ratio = max(max_largest_cluster_ratio, score["largest_cluster_ratio"])
            total_clusters += score["num_clusters"]

        results["is_repetitive"].append(any_repetitive)
        results["clustered_ratio"].append(max_clustered_ratio)
        results["largest_cluster_ratio"].append(max_largest_cluster_ratio)
        results["num_clusters"].append(total_clusters)

    return results


@click.command()
@click.option(
    "-i",
    "--input-dataset-name-or-path",
    required=True,
    help="Input dataset; can either be a local directory or a Hugging Face dataset name",
    default="allenai/Dolci-Think-SFT-32B",
)
@click.option(
    "-o",
    "--output-dataset-name-or-path",
    default=None,
    help="Output dataset; can either be a local directory or a Hugging Face dataset name. If not provided, the output will be input with suffix '_filtered'",
)
@click.option("-r", "--num-rows", type=int, default=0, help="Number of rows to filter; if 0, no filtering.")
@click.option("-p", "--num-proc", type=int, default=0, help="Number of processes to use for filtering.")
@click.option("-b", "--batch-size", type=int, default=1000, help="Batch size for processing the dataset.")
def main(
    input_dataset_name_or_path: str,
    output_dataset_name_or_path: str | None,
    num_rows: int,
    num_proc: int,
    batch_size: int,
):
    # load and filter dataset
    dataset = load_dataset(input_dataset_name_or_path, split="train")
    assert isinstance(dataset, Dataset)

    if num_rows > 0:
        dataset = dataset.take(num_rows)

    # get embeddings
    dataset = dataset.map(embed_dataset, num_proc=num_proc, batched=True, batch_size=batch_size)

    # save dataset
    if output_dataset_name_or_path is None:
        output_dataset_name_or_path = f"{input_dataset_name_or_path}_thinking_clusters"

    if Path(input_dataset_name_or_path).exists():
        dataset.save_to_disk(output_dataset_name_or_path)
    else:
        dataset.push_to_hub(output_dataset_name_or_path, private=True, num_proc=num_proc)


if __name__ == "__main__":
    main()
