# Olmo 3 SFT on TPU with MaxText

SFT of Olmo 3 7B on the `gke-tpu-v4` cluster (`allenai/olmo-gcp-infra`) with MaxText + Tunix, tracked in
allenai/open-instruct#1933. It matches open-instruct's OLMo-core SFT:

- **Loss:** a 280-step run on single-turn Dolci rows, same recipe as the GPU runs. Steps 11–280 average 0.6101, against 0.6104 and 0.6090 for two GPU seeds.
- **Tokens and loss masks:** identical on 4,130 of 4,130 Dolci-Instruct rows, including tool and multi-turn rows.
- **HF export:** bit-exact.

| Path | What |
|---|---|
| `image/` | The training image: upstream MaxText at a pinned commit, plus our MaxText patch, Tunix and the post-training dependencies. `build.sh <maxtext commit>` builds it with Cloud Build. |
| `image-delretry/` | Layer on any published image: Orbax's delete of an old checkpoint step retries 5 times with backoff. If only the HNS folder cleanup still fails once every object is gone, it logs `ORBAX-DELETE-LEAKED-FOLDERS` and continues, so a transient GCS error costs storage, not the run; objects still left raise as before. The parallel deleter (`num_threads`, local paths) and the rename/todelete options bypass it. `build.sh <base tag>` builds `<base tag>-dlr<hash>`; the build runs the patch's check against the image's own Orbax. |
| `k8s/olmo3-sft/` | Kueue JobSet, launcher `train_sft.sh` and settings. Historical smoke base on a `2x2x4` slice (16 chips), which is no longer available; use the `4x4x8` overlay. |
| `k8s/instruct-sft-4x4x8/` | Overlay: the Olmo 3 7B Instruct SFT recipe on one reserved `4x4x8` slice (128 chips) at 32,768 tokens. |
| `k8s/olmo35-sft-4x4x8/` | Overlay: the Olmo 3.5 hero-small SFT recipe (16 rows of 65,536 tokens, 11,768 steps) on one `4x4x8` slice. Start here: it is the cheapest per chip-hour. |
| `k8s/olmo35-sft-8x8x8/` | Overlay: the same budget at 64 rows and 2,942 steps on one `8x8x8` (512 chips), for when no `4x4x8` is free. ~1.09x the chip-hours, ~3.6x sooner (projected); its data order differs, so compare it only with a same-slice control. |
| `k8s/watch_jobset.sh` | Waits for a JobSet and saves every pod's log. |
| `data/` | `build_full_dolci.py` writes Dolci-Instruct-SFT as globally shuffled parquet shards. `count_tokens.py` sizes a run. |
| `export_hf.sh` | MaxText checkpoint to HF, with the base model's `config.json` and the training tokenizer. |
| `parity/` | Token and loss-mask parity against open-instruct's SFT tokenization. |

## Configure your run

Copy an overlay, give the JobSet and ConfigMaps unique names, and set a unique `RUN_NAME` before applying it with
`kubectl apply -k <overlay>`. The templates refuse to launch with `RUN_NAME=CHANGE_ME`, so a new run cannot
accidentally reuse the historical reference run's checkpoint directory. It pins the restore-fix image for resumes.
Every `k8s/*/train_sft.sh` copy must stay identical; check with
`for f in k8s/*/train_sft.sh; do diff -q k8s/olmo3-sft/train_sft.sh $f; done` after editing one.

Data builders use the standard HF cache settings (`HF_HOME`, `HF_DATASETS_CACHE`) and load one dataset revision.
Pin it with `--revision <dataset commit>`, or pass explicit local parquet paths/globs with `--data-files`:

```bash
python data/build_full_dolci.py <new shard dir> 128 0 --revision <dataset commit>
python olmo35/build_full_dolci_think.py <new shard dir> 512 0 --data-files '/your/data/*.parquet'
python data/count_tokens.py <shard dir> 32768 <tokenizer dir> --output <lengths.npy>
python parity/tok_oi_local.py 100 <sample.jsonl> '/your/data/*.parquet'
```

Shard output directories must be new or empty, and token counts refuse to overwrite an existing lengths file.
Export destinations are chosen by the caller. For Olmo 3, set `MAXTEXT_DIR` to a MaxText checkout at the pinned
revision and optionally `PY` to its Python executable; temporary files use `TMPDIR`. Olmo 3.5 exports accept either
a single-file or sharded HF reference and copy its metadata without copying the reference weights.

Image builds use the base commit from `maxtext_source`; an optional commit argument must match that pin.
They use Cloud Build's default service account unless `BUILD_SERVICE_ACCOUNT` names another authorized account
(a full `projects/<project>/serviceAccounts/<email>` resource). No source changes are needed to switch accounts.

The watcher uses the caller's timezone and includes the UTC offset. Set `KUBE_CONTEXT` or `RUNS_DIR` to override
the cluster context or local log directory. It exits nonzero on a failed job, timeout, or Kubernetes API error.

## What the patch fixes

The patches live as commits on abhishekraok/maxtext; each image directory's `maxtext_source` names the fork, base and head commit, and `build.sh` writes `patches/` from them with `fetch_patches.sh`. The Olmo 3 patch is `e259ed61..55bc76ae` (branch `olmo3-sft-image`). The fixes are not upstream yet; diffs are posted on the linked issues.

- **YaRN on full-attention layers only** (AI-Hypercomputer/maxtext#5544), as Olmo 3 was trained. Unpatched, the starting loss is 0.105 nats/token too high.
- **Chat templating** (AI-Hypercomputer/maxtext#5522). The whole conversation is rendered once. Assistant spans are derived the way open-instruct derives them: common-prefix starts when the prefixes render stably, otherwise token counts with a content check. Messages after the last assistant turn are kept as untrained context. Unpatched, every multi-turn row differs.
- **HF export shape table for Olmo 3** (AI-Hypercomputer/maxtext#5587). MaxText's exported `config.json` is still wrong, so `export_hf.sh` replaces it with the base model's.

## Traps

- **Tokenizer:** `allenai/olmo-3-tokenizer-instruct-dev` loads as `GPT2Tokenizer` under transformers 5 and drops the Dolma2 pre-tokenizer. Train with a `PreTrainedTokenizerFast.from_pretrained(...).save_pretrained(dir)` copy.
- **Checkpointing:** multi-host `train_sft` needs `enable_checkpointing=true`. Otherwise `jax.distributed` is never initialized on TPU and Tunix's checkpoint manager fails.
- **Every run saves its last step:** Tunix's `_save_last_checkpoint` forces a save at the final step whatever `checkpoint_period` is, because `train_sft` always passes a checkpoint directory. A 20-step smoke of the Olmo 3.5 overlay leaves a ~137 GB checkpoint with optimizer state.
- **Loss lines are per pod:** each pod's `completed step: N, … loss:` line carries its own value; the 32 pods of one step differed by up to 0.4. Read the step loss from the `LOSS-PARTS step=N ce=… moe_lb=… z=… tokens=…` line, which one pod logs: its `tokens` is the step's loss-token count and the logged total is `ce + moe_lb + z`. Never deduplicate metric-logger lines across pods.
- **Vocabulary vs tensor parallelism:** the vocabulary (100,278) does not divide by tensor parallelism 4.
  - Do not zero-pad it to 100,352. The padded logits sit at 0 while real logits are often low, which costs +3.2 nats/token.
  - Replicate the vocab axis instead: `logical_axis_rules=[['vocab',[]],['activation_vocab',[]]] sharding_tolerance=0.05`. This matches single-device log-probs to 6e-6.
- **Shuffling:** MaxText streams parquet, and its shuffle only permutes shards plus a buffer. Shuffle globally first (`data/build_full_dolci.py`), because the hub shards are grouped by source.
- **Step guard:** `train_sft` exits 0 with zero steps when the data iterator fails (AI-Hypercomputer/maxtext#5523). The launcher fails unless the last step completed.
- **Document boundaries:** MaxText packs one segment per row, with per-row attention masking and positions. open-instruct's OLMo-core path splits documents at every EOS instead. They agree wherever a row ends in exactly one EOS, which held for 4,128 of 4,130 checked rows. The two exceptions were tool rows that end without EOS; OLMo-core merges such a row with the next one.
