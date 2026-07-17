# Verifying SFT tokenization is byte-identical to `origin/main`

When you change code in `scripts/data/convert_sft_data_for_olmocore.py`
or `open_instruct/numpy_dataset_conversion.py`, you need to prove the
new output matches `origin/main` byte-for-byte before merging. This
page describes the procedure.

The full production mixer takes ~8 hours to run. For a tight feedback
loop during development, do a 50k-example A/B first (about 8 minutes of
wall-clock per side, plus a ~15s compare). Only fall back on a
full-scale run once the small A/B passes.

## How the compare works

The compare produces a `sha256` for every output artifact:

- `token_ids_part_*.npy`
- `labels_mask_part_*.npy`
- `token_ids_part_*.csv.gz` (decompress before hashing)
- `dataset_statistics.json` — hash after stripping `timestamp` and
  `output_directory` (these vary by run and aren't meaningful to compare)

Skip `dataset_statistics.txt` (human-readable dupe), the `tokenizer/`
dir, and any `_checkpoint*` files.

## Step 1: two image builds

Tokenization is driven from a Beaker image, so you need one image per
side of the A/B.

1. **Origin/main image.** Create a throwaway worktree at `origin/main`
   and build an image from it:

    ```bash
    git worktree add -b verify-main /tmp/oi-main-verify origin/main
    cd /tmp/oi-main-verify
    ./scripts/train/build_image_and_launch.sh \
        scripts/train/olmo-hybrid/7b_think_sft_tokenization.sh
    ```

    Note the image ID printed at the end.

2. **HEAD image.** From your branch:

    ```bash
    ./scripts/train/build_image_and_launch.sh \
        scripts/train/olmo-hybrid/7b_think_sft_tokenization.sh
    ```

## Step 2: parallel tokenize jobs

Launch both images against the full production mixer with identical
args (`Dolci-Think-SFT-32B 1.0` + 5 tool datasets at 3.0x,
`--chat_template_name olmo123`, `--max_seq_length 32768`). Write into
distinct weka output dirs, e.g.
`/weka/oe-adapt-default/$USER/dataset/olmo-hybrid-{main,head}`.

For a quick 50k A/B, add `--num_examples 50000` to both sides. At 50k
each job takes ~7–8 minutes.

## Step 3: compare on Beaker

weka isn't mounted locally, so run the compare as a tiny CPU job on
Beaker. Pin `--image` to an explicit image ID — the
`$USER/open-instruct-integration-test` tag moves in-place on every
push, so by the time the tokenize jobs finish the tag may no longer
match what you ran.

```bash
uv run python mason.py \
    --cluster ai2/jupiter \
    --budget ai2/oe-adapt \
    --workspace ai2/olmo-instruct \
    --image <HEAD_IMAGE_ID> \
    --pure_docker_mode --no-host-networking \
    --gpus 0 --priority urgent \
    --description "compare head vs main" \
    --no_auto_dataset_cache \
    -- bash -c 'uv run python <<PY
import hashlib, gzip, json, os, sys
new, ref = sys.argv[1], sys.argv[2]
fail = 0
for name in sorted(set(os.listdir(new)) | set(os.listdir(ref))):
    if name.startswith("_checkpoint") or name in ("dataset_statistics.txt", "tokenizer"):
        continue
    a, b = os.path.join(new, name), os.path.join(ref, name)
    if not (os.path.isfile(a) and os.path.isfile(b)):
        print(f"MISSING: {name}"); fail += 1; continue
    if name == "dataset_statistics.json":
        da, db = json.load(open(a)), json.load(open(b))
        for d in (da, db):
            d.pop("timestamp", None); d.pop("output_directory", None)
        ha = hashlib.sha256(json.dumps(da, sort_keys=True).encode()).hexdigest()
        hb = hashlib.sha256(json.dumps(db, sort_keys=True).encode()).hexdigest()
    elif name.endswith(".gz"):
        ha = hashlib.sha256(gzip.open(a).read()).hexdigest()
        hb = hashlib.sha256(gzip.open(b).read()).hexdigest()
    else:
        ha = hashlib.sha256(open(a, "rb").read()).hexdigest()
        hb = hashlib.sha256(open(b, "rb").read()).hexdigest()
    tag = "OK  " if ha == hb else "DIFF"
    if ha != hb: fail += 1
    print(f"{tag} {name} {ha} vs {hb}")
print("=== PASSED ===" if not fail else f"=== FAILED: {fail} mismatches ===")
sys.exit(1 if fail else 0)
PY
' -- /weka/oe-adapt-default/$USER/dataset/olmo-hybrid-head \
   /weka/oe-adapt-default/$USER/dataset/olmo-hybrid-main
```

Success looks like `=== PASSED ===`. On failure, use `cmp -l` on the
diverging artifact to find the first mismatching byte range.

## Gotchas

- Always pin `--image` to an explicit Beaker image ID, not a tag. Tags
  move when anyone rebuilds the image.
- The Beaker container doesn't have `jq` — use `uv run python` for JSON
  hashing.
- If the *file listings* differ (not just hashes), the culprit is
  usually a code change in `scripts/data/convert_sft_data_for_olmocore.py`
  or upstream in `apply_chat_template`. Check
  `git log origin/main..HEAD -- scripts/data/ open_instruct/dataset_transformation.py`.
- If you've already produced a full-scale `origin/main` reference on
  weka, point the compare's reference side at it to skip re-running the
  full mixer.

## Golden reference run

There is an existing full-scale `origin/main` tokenization on weka that
you can compare against without re-running the full mixer:

- Beaker experiment: <https://beaker.org/ex/01KPRDGYEM81EASNNSBZ2HA7KA>
- Output directory: `/weka/oe-adapt-default/finbarrt/dataset/olmo-hybrid-main-repro`

Point the compare's reference side at that directory.
