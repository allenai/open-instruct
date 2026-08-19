# Running SFT: a known-good starting point

Olmo 3 7B dense, **one epoch of Dolci-Instruct-SFT, one 8×H100 node**.

Measured cost and outcome: ~1.5 h tokenization (CPU, 0 GPUs) + ~9.5 h training
(~77 GPU-hours) + ~15 min per checkpoint to evaluate. IFBench +6.7 and GSM8K +1.8
over the base model, which lands 2.4 points under the released
`Olmo-3-7B-Instruct-SFT` on IFBench.

The recipe is [`scripts/train/debug/oc_sft_olmo3_7b_1node.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/debug/oc_sft_olmo3_7b_1node.sh).
Read its header before editing it: several flags look wrong and are not, and the
reasons are recorded there rather than repeated here. For a flag-by-flag reference
see [Supervised finetuning](finetune.md).

## Example runs

| Report | What it is |
|---|---|
| [Olmo 3 7B SFT pipeline validation](https://wandb.ai/ai2-llm/open_instruct_internal/reports/Olmo-3-7B-SFT-pipeline-validation-and-eval-results--VmlldzoxNzcwOTIyMg) | This recipe, verbatim. [Beaker](https://beaker.org/ex/01KZHT32T30M2VHCWKRJS1G9P7) · [W&B run](https://wandb.ai/ai2-llm/open_instruct_internal/runs/ed453531) |
| [Olmo-Hybrid-7B SFT](https://wandb.ai/ai2-llm/open_instruct_internal/reports/Olmo-Hybrid-7B-SFT-through-open-instruct-it-works-on-Blackwell--VmlldzoxNzcyMjMzOQ) | The same recipe pointed at the hybrid base model. [Beaker](https://beaker.org/ex/01M013HTEWJYVHAKPHHJ762YJW) |

## Before you start

- Ai2-internal only: Beaker access, Docker, and `beaker account whoami` working.
- `build_image_and_launch.sh` refuses to run with uncommitted changes, so commit first.
- Steps 1–3 run from an open-instruct checkout; step 4 runs from an
  [olmo-eval](https://github.com/allenai/olmo-eval) checkout (`uv sync --frozen`).

## 1. Tokenize (CPU job, ~1.5 h)

Training hard-fails if the pre-tokenized cache is missing, so tokenization is a
separate job:

```bash
./scripts/train/build_image_and_launch.sh scripts/train/debug/oc_sft_olmo3_7b_1node.sh tokenize
```

The cache key hashes the tokenizer config, mixer, transform functions,
`max_seq_length` and seed. Both jobs read those from shared variables at the top of
the script so they cannot drift — if you change one, change it once, and re-tokenize.
Any divergence produces a different key and training fails as though tokenization
never ran.

Wait for this job to reach exit 0 before launching step 2.

## 2. Train (1×8 H100, ~9.5 h)

```bash
./scripts/train/build_image_and_launch.sh scripts/train/debug/oc_sft_olmo3_7b_1node.sh train
```

What a healthy run looks like: 1,723 steps for the epoch, CE from ~1.20 down to
~0.65, smooth, with no instability after the LR peaks around step 52. Step count is
a function of the mixture and chat template, not a constant — `olmo123` resolves to
the tokenizer's own template, which prepends a function-calling system block to every
conversation without a system message, so rows average 840 tokens rather than the
~643 you get under the `olmo_thinker` templates.

Checkpoints are written every 345 steps — five per epoch, which is what the eval
table below covers — into the directory the job logs as `CHECKPOINT_OUTPUT_DIR`.

If the job stays queued, run `beaker job events <job-id>` and read the scheduler's own
reason — the two common ones need opposite fixes, and both are spelled out in the
script header. Do not guess from cluster docs.

## 3. Convert checkpoints to HuggingFace

olmo-eval can serve a raw olmo-core checkpoint, but the verified path exports to HF
first. One 0-GPU job per checkpoint:

```bash
CKPT=<the checkpoint directory the training job wrote to>
STEP=1723
BEAKER_IMAGE=<the image build_image_and_launch.sh printed in step 1>

uv run python mason.py \
    --cluster ai2/saturn ai2/neptune ai2/ceres \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Convert Olmo-3-7B SFT step$STEP -> HF" \
    --pure_docker_mode \
    --num_nodes 1 \
    --gpus 0 \
    --no_auto_dataset_cache \
    -- uv run python scripts/train/convert_olmo_core_to_hf.py \
    --checkpoint-dir $CKPT/step$STEP/model_and_optim \
    --model-name allenai/Olmo-3-1025-7B \
    --tokenizer-name allenai/olmo-3-tokenizer-instruct-dev \
    --output-dir $CKPT/hf_step$STEP
```

Pass the same tokenizer you trained with. The exported directory carries its chat
template, which is what makes the model answer chat-formatted eval prompts at all.

## 4. Evaluate

From an olmo-eval checkout, one job per checkpoint on 1 GPU (~15 min each).
`$CKPT` and `$STEP` are the same as in step 3:

```bash
uv run olmo-eval beaker launch \
    -n olmo3-7b-sft-step$STEP \
    -m $CKPT/hf_step$STEP \
    -t ifeval_ood -t gsm8k \
    --harness default \
    -o provider.tokenizer=allenai/olmo-3-tokenizer-instruct-dev \
    -c ai2/ceres -c ai2/jupiter \
    -w ai2/open-instruct-dev \
    -B ai2/oe-other \
    -p urgent \
    --no-follow
```

Evaluate the base model the same way, with `-m allenai/Olmo-3-1025-7B`, so the delta
isolates the weights. `ifeval_ood` (IFBench, 300 prompts) moves early under
instruction tuning and `gsm8k` (1,319 problems) is the regression check; both are
cheap. Swap in a larger suite once you care about a specific claim.

Results print as a `Results Summary` table at the end of the job log and are written
to `/results/metrics.json`:

| checkpoint | epoch | `ifeval_ood` | `gsm8k` |
|---|---|---|---|
| base `Olmo-3-1025-7B` | 0.00 | 0.1833 | 0.7430 |
| step 345 | 0.20 | 0.2367 | 0.7210 |
| step 690 | 0.40 | 0.2200 | 0.7445 |
| step 1035 | 0.60 | 0.2133 | 0.7362 |
| step 1380 | 0.80 | 0.2633 | 0.7597 |
| step 1723 | 1.00 | 0.2500 | 0.7612 |

Treat differences of a few points as noise: the standard error on an IFBench
difference is ~3.5 points at 300 prompts, so the shape of that column is flat after
the first checkpoint, and the apparent peak at step 1380 is not separable from
sampling variation.

!!! warning "A metric of exactly 0.0000 is a broken run, not a bad model"
    olmo-eval reports the experiment as `Success` when every generation request
    fails. It is almost always a missing chat template — check the `-m` directory has
    one, and that `provider.tokenizer` points at a tokenizer that does.

## Changing the recipe

- **Other base models.** `--config_name` must resolve to an olmo-core
  `TransformerConfig` preset; there is no fallback if it does not. Architectures
  olmo-core has no preset or HF weight conversion for need both written first —
  see the Olmo-Hybrid report above for what that costs.
- **Other node counts.** Hold the global batch at 1,048,576 tokens:
  `per_device × grad_accum × (world_size / cp_degree) × seq_len`. Adjust
  `gradient_accumulation_steps`, not the sequence length.
- **Verify conversions numerically.** If you write or change olmo-core ↔ HF weight
  conversion, compare a forward pass against the HF reference before training on it.
  Structural checks — key and shape matching, config round trips, state-dict round
  trips — all pass cleanly on numerically broken code; in the Olmo-Hybrid work three
  green CPU checks missed a 38% logit error caused by a single hardcoded epsilon.
