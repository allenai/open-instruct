# Verifying SFT tokenization is byte-identical to `origin/main`

When changing code in `scripts/data/convert_sft_data_for_olmocore.py` or
`open_instruct/numpy_dataset_conversion.py`, you want to prove the new
output matches `origin/main` byte-for-byte before merging. The full
production mixer takes many hours. This page walks through a controlled
50k-example A/B that finishes in about 8 minutes of wall-clock.

## Approach

1. Build one Beaker image from `origin/main` and one from your branch.
2. Launch both tokenization jobs in parallel with identical args into
   distinct weka output dirs.
3. Run a tiny CPU compare job on Beaker that `sha256`s every output
   artifact.

The compare script (`scripts/train/olmo-hybrid/_compare_tokenization.sh`)
hashes `token_ids_part_*.npy`, `labels_mask_part_*.npy`,
`token_ids_part_*.csv.gz`, and `dataset_statistics.json` (with
`timestamp` and `output_directory` stripped). On mismatch it prints the
first diverging bytes.

## Step 1: build the `origin/main` image

```bash
git worktree add -b verify-main /tmp/oi-main-verify origin/main
cd /tmp/oi-main-verify
```

Two fixes you will likely need on the main worktree (both are harmless
no-ops if your branch already contains them):

- If the launcher on your branch calls `scripts/data/download_hf_repo.py`
  and that file doesn't exist on `origin/main`, drop the download step.
- `transformers v5` rejects tokenizer-only HF repos in
  `AutoConfig.from_pretrained`. Cherry-pick the path-substring fix if
  needed (this repo's commit `3f209e6e7`).

Then build the image:

```bash
./scripts/train/build_image_and_launch.sh \
    scripts/train/olmo-hybrid/7b_think_sft_tokenization_small.sh
```

Note the image ID printed at the end (looks like `01K...`).

## Step 2: build the HEAD image

Back on your branch worktree, same launcher:

```bash
OUTPUT_SUFFIX=head-50k NUM_EXAMPLES=50000 \
    ./scripts/train/build_image_and_launch.sh \
    scripts/train/olmo-hybrid/7b_think_sft_tokenization_small.sh
```

And on the main worktree:

```bash
OUTPUT_SUFFIX=main-50k NUM_EXAMPLES=50000 \
    ./scripts/train/build_image_and_launch.sh \
    scripts/train/olmo-hybrid/7b_think_sft_tokenization_small.sh
```

Outputs land at
`/weka/oe-adapt-default/$USER/dataset/olmo-hybrid-small-{head-50k,main-50k}`.
Each tokenize job takes ~7–8 minutes.

## Step 3: compare

weka isn't mounted locally, so run the compare on Beaker with a tiny CPU
job. Pin `--image` to the HEAD image ID so the container has the latest
`_compare_tokenization.sh`:

```bash
uv run python mason.py \
    --cluster ai2/jupiter \
    --budget ai2/oe-adapt \
    --workspace ai2/olmo-instruct \
    --image <HEAD_IMAGE_ID> \
    --pure_docker_mode --no-host-networking \
    --gpus 0 --priority urgent \
    --description "compare head-50k vs main-50k" \
    --no_auto_dataset_cache \
    -- bash scripts/train/olmo-hybrid/_compare_tokenization.sh \
        /weka/oe-adapt-default/$USER/dataset/olmo-hybrid-small-head-50k \
        /weka/oe-adapt-default/$USER/dataset/olmo-hybrid-small-main-50k
```

Runs in about 15 seconds. Success looks like:

```
=== PASSED: byte-for-byte match ===
```

On failure, the script prints which artifact diverged and the first
differing byte offsets via `cmp -l`.

## Full-scale repro (in progress)

Alongside the 50k A/B, a full-dataset tokenization job is running against
the `origin/main` code to produce a permanent byte-for-byte reference for
the entire production mixer:

- Beaker experiment: [01KPRDGYEM81EASNNSBZ2HA7KA](https://beaker.org/ex/01KPRDGYEM81EASNNSBZ2HA7KA)
- Output dir (on completion): `/weka/oe-adapt-default/finbarrt/dataset/olmo-hybrid-main-repro`

Once finalized, re-run the compare step against your branch's
`olmo-hybrid-fresh` output to verify byte-for-byte parity on the full
mixer.

## Gotchas

- The Beaker image tag `$USER/open-instruct-integration-test` is
  updated-in-place, so after a new push it no longer points at the
  image you ran your tokenize jobs from. Always pin the compare job's
  `--image` to the explicit HEAD image ID.
- `jq` is not in the Beaker container; the compare script uses
  `uv run python` for stats JSON hashing.
- If the ref vs new file *listings* differ (not just hashes), the cause
  is usually a code change in `scripts/data/convert_sft_data_for_olmocore.py`
  or upstream in `apply_chat_template`, not a refactor of the numpy path.
  Confirm by checking `git log origin/main..HEAD -- scripts/data/`.
