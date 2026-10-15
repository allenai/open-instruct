# Olmo 3 SFT on TPU with MaxText

SFT of Olmo 3 7B on the `gke-tpu-v4` cluster (`allenai/olmo-gcp-infra`) with MaxText + Tunix, tracked in
allenai/open-instruct#1933. It matches open-instruct's OLMo-core SFT:

- **Loss:** a 280-step run on single-turn Dolci rows, same recipe as the GPU runs. Steps 11–280 average 0.6101, against 0.6104 and 0.6090 for two GPU seeds.
- **Tokens and loss masks:** identical on 4,130 of 4,130 Dolci-Instruct rows, including tool and multi-turn rows.
- **HF export:** bit-exact.

| Path | What |
|---|---|
| `image/` | The training image: upstream MaxText at a pinned commit, plus `patches/maxtext.patch`, Tunix and the post-training dependencies. `build.sh <full maxtext commit SHA>` builds it with Cloud Build; the base image is tagged with the full SHA. |
| `k8s/olmo3-sft/` | Kueue JobSet, launcher `train_sft.sh` and settings. The base is a smoke test on one spot `2x2x4` slice (16 chips) at 4,096 tokens; that spot pool no longer exists, so run the overlays. |
| `k8s/instruct-sft-4x4x8/` | Overlay: the Olmo 3 7B Instruct SFT recipe on one reserved `4x4x8` slice (128 chips) at 32,768 tokens. |
| `k8s/olmo3-dpo/`, `k8s/instruct-dpo-4x4x8/`, `k8s/dpo-smoke-4x4x8/` | DPO on pre-tokenized open-instruct pairs (`data/build_dpo_pretokenized.py`), dpo_norm: the 7B Instruct DPO recipe on one `4x4x8`, and its first 20 steps as a smoke test. |
| `evals/` | Released-vs-TPU eval specs for SFT and DPO, and `compare_pairs.py` for paired CIs. |
| `k8s/watch_jobset.sh` | Waits for a JobSet and saves every pod's log. |
| `k8s/resume_check.py` | Checks that a resumed run trains on the same batches as the original pass. |
| `data/` | `build_full_dolci.py` writes Dolci-Instruct-SFT as globally shuffled parquet shards. `count_tokens.py` sizes a run. |
| `export_hf.sh` | MaxText checkpoint to HF, with the base model's `config.json` and the training tokenizer. Tunix DPO checkpoints (policy under `base`) go through `tunix_to_maxtext.py` first. |
| `parity/` | Token and loss-mask parity against open-instruct's SFT tokenization. |

## What the patch fixes

The patch is against MaxText `e259ed61`. The fixes are not upstream yet; diffs are posted on the linked issues.

- **YaRN on full-attention layers only** (AI-Hypercomputer/maxtext#5544), as Olmo 3 was trained. Unpatched, the starting loss is 0.105 nats/token too high.
- **Chat templating** (AI-Hypercomputer/maxtext#5522). The whole conversation is rendered once. Assistant spans are derived the way open-instruct derives them: common-prefix starts when the prefixes render stably, otherwise token counts with a content check. Messages after the last assistant turn are kept as untrained context. Unpatched, every multi-turn row differs.
- **Resume skips trained batches.** Tunix skips the batches a restored checkpoint already trained on only when no data hooks are set, and MaxText always sets them. Unpatched, a run restarted after preemption or a slice failure replays the data from the start. `k8s/resume_check.py` validates a resume: `total_weights` depends only on the batch, so it must equal the original pass's at every step both ran.
- **Only trained rows are loaded.** With a fractional per-device batch (TP4 × FSDP32 at 0.25 sequences per chip) the HF pipeline packed 4 sequences per data host while the loss kept 2, so half of each pass was dropped and the stream ran dry at step 2,464 of 3,336. Two hosts then hit StopIteration, and libtpu reports `SLICE_FAILURE_SW_INJECT_ERROR`, which looks like a hardware fault. Each host now packs only its trained rows.
- **EOS stays in the loss.** MaxText ignored targets equal to `bos_token_id`, and Olmo's BOS is also its EOS (`<|endoftext|>`), so no SFT model learned to stop: every eval sample ran to `max_tokens`, while the training loss looked normal. Check a new SFT pipeline's P(EOS) after a finished answer before spending eval GPUs.
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
