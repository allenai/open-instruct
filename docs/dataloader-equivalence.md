# DataLoader Equivalence: HF vs Numpy Paths

## Background

There are two paths for loading SFT training data in this codebase:

1. **HF path** (`olmo_core_finetune.py`): loads a HuggingFace Dataset, passes it through `HFDataLoader` with `PackedSFTCollator`, which bin-packs variable-length examples into fixed-length `[num_rows, sequence_length]` tensors with `input_ids` and `label_mask`.

2. **Numpy path** (external OLMo-core training scripts): an HF dataset is first converted to numpy mmap format via `scripts/data/convert_sft_data_for_olmocore.py`, producing `token_ids_part_*.npy`, `labels_mask_part_*.npy`, and `token_ids_part_*.csv.gz` metadata files. OLMo-core's `NumpyPackedFSLDataset` then packs these into fixed-length sequences.

Both paths must produce identical training data from the same source dataset.

## How conversion works

The conversion script (`scripts/data/convert_sft_data_for_olmocore.py`) iterates over a tokenized HF dataset and:

- Flattens all `input_ids` into a single continuous token stream
- Builds a parallel `labels_mask` boolean array (`True` where `label != -100`)
- Records document boundaries as `(start, end)` pairs in gzipped CSV metadata
- Writes token IDs as `np.uint32` and label masks as `np.bool_` memmap files

`NumpyPackedFSLDataset` then:

- Detects document boundaries by scanning for EOS tokens in the flat stream (matching the CSV metadata boundaries, since each SFT example ends with exactly one EOS token)
- Packs documents into fixed-length instances using the Optimized Best-Fit Decreasing (OBFD) algorithm via `InstancePacker`
- Pads each instance to `sequence_length` with the tokenizer's pad token

## How the test verifies equivalence

The integration test `TestHFNumpyEquivalence.test_batches_match` in `open_instruct/test_data_loader.py` verifies tensor-level equality between the two paths:

1. Loads 100 real examples from `allenai/tulu-3-sft-olmo-2-mixture-0225`, tokenized with the OLMo-2 tokenizer
2. Passes them through `HFDataLoader` with an `OBFDCollator` that uses the same `InstancePacker` algorithm as `NumpyPackedFSLDataset`
3. Converts the same data to numpy format and loads via `NumpyPackedFSLDatasetConfig`
4. Asserts `torch.equal` on both `input_ids` and `label_mask` tensors

### The OBFDCollator

The production `PackedSFTCollator` uses first-fit-decreasing bin packing, which differs from `NumpyPackedFSLDataset`'s OBFD algorithm. To enable batch-level comparison, the test uses a custom `OBFDCollator` that imports `InstancePacker` from `olmo_core.data.utils` and replicates the numpy path's packing logic on HF data.

### Shuffle alignment

`HFDataLoader._reshard` shuffles the dataset using `torch.randperm` with a deterministic seed. The test converts to numpy using the same shuffled order so both paths see documents in the same sequence.

## Running the test

```bash
uv run pytest open_instruct/test_data_loader.py::TestHFNumpyEquivalence -v
```

This downloads and tokenizes 100 examples from HuggingFace, so it requires network access and takes around 10-15 seconds.

## What this test does not cover

- **PackedSFTCollator equivalence**: The production HF path uses `PackedSFTCollator` (first-fit-decreasing), not OBFD. The two packers produce different row assignments from the same data. The test verifies data integrity through the numpy conversion, not packing strategy equivalence.
- **Multi-rank sharding**: The test uses `dp_world_size=1`. Distributed sharding is tested separately in `TestWorldAwarePacking`.
- **Long document handling**: All test examples fit within `sequence_length=4096`. The `LongDocStrategy.truncate` behavior is not exercised.
