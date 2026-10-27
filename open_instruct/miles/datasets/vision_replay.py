"""Pack one document's image tensors for MILES' ragged-tensor transport.

Multiple images share a crop axis, so pooling indices must be rebased while -1
padding stays -1. This is not sequence packing: each document remains separate.
"""

import torch


def training_inputs(images: list, prompt_ids: list[int], patch_token_id: int) -> dict[str, torch.Tensor]:
    if not images:
        raise ValueError("Image replay requires at least one image")
    pixels, pooling, positions = [], [], []
    patch_base = 0
    for image in images:
        patches, pools = image.patches, image.pooling
        if patches.ndim != 4 or pools.ndim != 3 or patches.shape[0] != 1 or pools.shape[0] != 1:
            raise ValueError("Expected a single document: patches [1,crops,patches,width], pools [1,tokens,neighbors]")
        count = patches.shape[1] * patches.shape[2]
        if pools.dtype != torch.int64 or bool(((pools < -1) | (pools >= count)).any()):
            raise ValueError("Pooling index outside image crop range")
        current = [p for start, end in image.offsets for p in range(start, end + 1)]
        if len(current) != pools.shape[1] or not bool((pools >= 0).any(-1).all()):
            raise ValueError("Every patch token must have a valid pooling row")
        pixels.append(patches)
        pooling.append(torch.where(pools >= 0, pools + patch_base, pools))
        positions.extend(current)
        patch_base += count
    expected = [i for i, token in enumerate(prompt_ids) if token == patch_token_id]
    if positions != expected:
        raise ValueError("Image feature positions do not cover exactly the expanded patch tokens")
    return {
        "images": torch.cat(pixels, dim=1).contiguous(),
        "pooled_patches_idx": torch.cat(pooling, dim=1).contiguous(),
        "image_positions": torch.tensor(positions, dtype=torch.int64),
    }
