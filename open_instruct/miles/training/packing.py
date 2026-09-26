"""Order-preserving, unpadded packs contained within one optimizer step."""

import torch


def plan(lengths, max_tokens):
    """Greedy consecutive packs; reject overflow instead of truncating samples."""
    if not lengths or max_tokens < 1 or any(n < 1 or n > max_tokens for n in lengths):
        raise ValueError("Packing requires positive lengths within the token budget")
    packs, current, used = [], [], 0
    for index, length in enumerate(lengths):
        if current and used + length > max_tokens:
            packs.append(current)
            current, used = [], 0
        current.append(index)
        used += length
    return packs + [current]


def equalize(packs, count):
    """Split packs to the largest rank's count so EP collectives stay aligned.

    Every rank has the same number of samples. Thus the common count never
    exceeds the number of samples and no dummy tokens or empty forwards are needed.
    """
    packs = [list(pack) for pack in packs]
    if not len(packs) <= count <= sum(map(len, packs)):
        raise ValueError("Cannot equalize packing schedule without empty batches")
    while len(packs) < count:
        index = max(range(len(packs)), key=lambda i: len(packs[i]))
        pack = packs[index]
        middle = len(pack) // 2
        packs[index : index + 1] = [pack[:middle], pack[middle:]]
    return packs


def combine(samples, indices):
    """Keep each sample's loss/reward/version records alongside concatenated tokens."""
    selected = [samples[i] for i in indices]
    fields = {key for key, value in selected[0].items() if isinstance(value, list)}
    fields.discard("max_seq_lens")
    batch = {key: [item for sample in selected for item in sample[key]] for key in fields}
    tokens = torch.cat([sample["tokens"] for sample in selected], dim=1)
    lengths = [int(sample["tokens"].numel()) for sample in selected]
    batch.update(
        tokens=tokens,
        max_seq_lens=None,
        doc_lens=torch.tensor([lengths], dtype=torch.int32, device=tokens.device),
        max_doc_lens=[max(lengths)],
    )
    if "dynamic_global_batch_size" in selected[0]:
        batch["dynamic_global_batch_size"] = selected[0]["dynamic_global_batch_size"]
    return batch


def measurements(batches, max_tokens):
    tokens = [batch["tokens"].numel() for batch in batches]
    samples = sum(len(batch["total_lengths"]) for batch in batches)
    return {
        "samples": samples,
        "packs": len(batches),
        "model_tokens": sum(tokens),
        "max_pack_tokens": max(tokens),
        "token_budget": max_tokens,
        "fill_fraction": sum(tokens) / (len(batches) * max_tokens),
        "samples_per_pack": samples / len(batches),
    }
