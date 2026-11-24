# Olmo 3.5 multimodal SFT

This page covers Olmo 3.5 multimodal SFT end to end: the recommended recipe, launching it on GPU (Beaker) or TPU
(GKE), exporting checkpoints to HF, evaluation, and the known gotchas. Every step launches from open-instruct. The
training code itself is OLMo-core's multimodal SFT recipe
([allenai/OLMo-core#942](https://github.com/allenai/OLMo-core/pull/942)), which open-instruct pins by commit.

The model is **`olmo35-tiny`**, the 810M-total MoE (`olmoe3-810m-cx2`) with a SigLIP2 vision encoder. It was called
"small" before 2026-11, so older records and directory names still say `small`. Olmo 3.5 **Small** now refers to the
A4B MoE, which has no multimodal SFT recipe yet.

H-numbers (H230, H231, H271) are internal Ai2 post-training ledger IDs. Each hypothesis records the runs, the
exact commands and the results.

## Recommended recipe: `4n-emo256k`

The recommended recipe is arm `4n-emo256k` of H231:

| | |
|---|---|
| Base | **B2e**: the 256K causal-EMo multimodal midtrain, `olmoe3-810m-cx2-mm-hero-mt-lc-256k-emocausal-lm1e4` step 22240 (native OLMo-core checkpoint; model only, with routers and causal EMo taken from its config) |
| Data | C12 text SFT plus the 10% long-context replay (the text-only `c12r` recipe's text), plus the Molmo2 stage-2 image mixture v1 at 3.5e9 image tokens |
| Batch | 32 × 65,536 tokens (4 nodes × 8 GPUs, one packed sequence per GPU) |
| Steps | 10,962 (derived by the recipe from the text exposure and the image dose), warmup 3% |
| LR | peak 2e-4 for the LM, 4e-4 for the connector (×2), 2e-5 for the ViT (÷10); weight decay 0, grad clip 1.0 |
| EMo | causal pools on, as in midtraining: training samples pool sizes uniformly from 16 to 512 |
| Seeds | data seed 34521 (default) and 34522. Both ran, and the pair sets the run-to-run noise floor |
| Saves | permanent at 5,481 and 10,962; ephemeral every 250 steps for resuming |
| Cost | about 3.9 s/step on 4 × 8 B300 (`ai2/holmes`), about 12 h of training |

## GPU launch (Beaker)

```bash
uv run python scripts/train/sft_mm.py --recipe 4n-emo256k                 # data seed 34521
uv run python scripts/train/sft_mm.py --recipe 4n-emo256k --seed 34522    # the second seed
```

[`scripts/train/sft_mm.py`](https://github.com/allenai/open-instruct/blob/main/scripts/train/sft_mm.py) is the
multimodal sibling of [`sft.py`](sft_running_guide.md). It uses the same flags (`--checkpoint`, `--name`, `--seed`,
`--steps`, `--workspace`, `--priority`, `--dry_run`) and the same resume-by-rerun behavior. The launch itself differs:

- **Training code:** each replica fetches `allenai/OLMo-core` at `OLMO_CORE_SHA` (the #942 head the H230/H231 runs
  trained at), checks that it got that commit, installs it, and runs `src/scripts/train/Olmo35-Multimodal-SFT.py`.
  open-instruct's own `olmo-core` pin in `pyproject.toml` is not involved, so text SFT is unaffected.
- **Image:** the OLMo-core image the runs used (`olmo-core-tch2130cu130-fa4-rma-2026-09-11`: torch 2.13, CUDA 13,
  FA4), pinned by Beaker ID. Nothing needs building.
- **Spec:** the script writes the Beaker spec itself instead of going through `mason.py`, because the job is not
  an open-instruct entry point. The spec has 4 replicas with leader selection, host networking, failure and
  preemption propagation, and auto-resume. `--dry_run` prints the spec.
- **Output:** the recipe saves to `/weka/oe-adapt-default/allennlp/deletable_checkpoint/<you>/sft/<name>`, where
  `<name>` defaults to `olmo35-mmsft-<recipe>-s<seed>`. Running the same command again resumes from the latest
  checkpoint there. Copy out anything you want to keep.
- **`--steps N`** stops the run after N steps but keeps the full LR schedule. Use it for smoke tests. The run gets its
  own `-<N>st` directory.
- **`--min_runtime`** defaults to 8h. H231's runs with 1h were preempted repeatedly by allocation balancing.

The other recipes are the remaining arms of the same sweep. Each one's overrides are verbatim from its ledger
`launch:` field:

| Recipe | Base | Change from the base recipe | Steps | Ledger |
|---|---|---|---|---|
| `4n-emo256k` | B2e | (recommended) | 10,962 | H231 |
| `4n-emo256k-up5` | B2e | image tokens' loss weighted so that images carry 5% of the loss | 10,962 | H231 Amendment 1 |
| `4n-emo256k-tok5` | B2e | image dose of 12.675e9 tokens | 15,568 | H231 Amendment 1 |
| `4n-base` | B1 (64K multimodal midtrain, step 20000) | none | 10,962 | H230 |
| `4n-vis10` | B1 | images at 10% of the loss | 10,962 | H230 |
| `4n-vitlr` | B1 | ViT LR at LR/2 instead of LR/10 | 10,962 | H230 |
| `4n-hires` | B1 | Molmo2's 24-crop high-resolution draw on the document, chart and text sources | 10,962 | H230 |
| `4n-rebal` | B1 | pointing and counting cut back, their tokens moved to the multi-image and document sources | 10,962 | H230 |

To check a recipe's arguments without a GPU, run OLMo-core's own `dry_run` subcommand from a checkout of the pinned
commit, on a machine with the Weka buckets mounted. It prints the resolved config and the derived schedule
(`steps`, `save_interval`):

```bash
PYTHONPATH=src python src/scripts/train/Olmo35-Multimodal-SFT.py dry_run my-run ai2/holmes \
    --recipe.base=256k-emo-causal --recipe.save_fraction=0.5 --recipe.num_nodes=4
```

## TPU launch (GKE)

The same recipe runs in MaxText on one TPU v4 `4x4x8` slice (128 chips). It starts from the same B2e weights, and
the same OLMo-core data loader runs in-process, so the TPU run sees the same rows in the same order. The steps are in
[Olmo 3.5 multimodal SFT on TPU](olmo35_mm_sft_tpu.md). The general TPU setup is in the
[TPU SFT guide](tpu_sft_guide.md). H271 records the TPU replica of `4n-emo256k`. It matches H231's GPU run within the
GPU seed pair's spread.

## Export to HF

Two exports are made from a native OLMo-core checkpoint:

- **`hf_mm`**: the multimodal model (ViT, connector and LM) for the vision evals.
- **`hf_text`**: the language model alone, as a text-only HF checkpoint for vLLM text evals.

The export and eval scripts live in `scripts/eval/olmo35_mm/`. For the text-only MoE conversion conventions, see
`scripts/train/debug/convert_moe_checkpoint_to_hf.py`.

**Check that `generation_config.json` lists `eos_token_id: [100265, 100257]`** (`<|im_end|>` and `<|endoftext|>`).
`convert_moe_checkpoint_to_hf.py` writes that by default. The first multimodal exports listed only 100257, and vLLM
then never stopped at the end of a turn (see Gotchas).

TPU checkpoints are converted to native OLMo-core first (see the TPU page). After that they export the same way.

## Evaluation

Each checkpoint (5,481 and 10,962 for the recommended recipe) gets:

- **Vision suite:** 13 tasks, full splits, in three groups. `docs24` covers document/chart/text VQA at 24 crops,
  `core` covers VQAv2, AI2D, CountBench, RealWorldQA and similar, and `list` covers the multi-image sets (MMMU, MuirBench,
  MMIU, BLINK, MMMU-Pro). All run on `hf_mm`.
- **Text battery:** the same cells as the text-only GPU SFT: IFEval, IFBench, GPQA, MATH, RULER (64K/128K, plus the
  RULER-plus tasks) and BFCL. All run on `hf_text`.
- **Dev suite:** the post-training dev cells (BFCL, IF, LCB, math, MMLU, PopQA, RULER-64K).
- **EMo check:** HF logits against native OLMo-core logits at the eval pool (512).

Two runs only differ if they differ by more than the seed pair's spread. The two `4n-emo256k` seeds differ by a lot
on some text cells (BFCL in particular), so read single-run differences against that spread.

## Results

- **H231**: `4n-emo256k` (both seeds), `up5`, `tok5`. Descriptive, with seed-pair noise floors.
- **H230**: the B1 arms (`4n-base`, `vis10`, `vitlr`, `hires`, `rebal`), read beside H231 as "different base".
- **H271**: the TPU replica of `4n-emo256k`, compared with both GPU seeds.

## Gotchas

- **Stop tokens.** A text export whose `generation_config.json` has `eos_token_id: 100257` alone makes vLLM ignore
  `<|im_end|>`. Generations then run past the end of the turn, which distorted the generation evals (BFCL parallel
  in particular). Training labels were correct. Only the export was wrong. Re-export with
  `[100265, 100257]` before trusting any generation eval.
- **EMo pools.** Training samples causal pool sizes uniformly from 16 to 512. Evaluation uses pool 512, which is all
  experts. Exports and evals must run at 512. A TPU step-1 parity gate should also fix the pool at 512 (min = max),
  because otherwise the sampled pool differs between implementations.
- **Naming.** `olmo35-tiny` is this 810M model, which used to be called "small". "Small" means the A4B model. Run and
  checkpoint names from before 2026-11 (`olmo35-small-mm-…`) refer to tiny.
- **The pinned commit.** `OLMO_CORE_SHA` is on #942's branch, not on OLMo-core main. Do not delete that branch
  before #942 merges. Otherwise GitHub may stop serving the commit and the launch will fail at the fetch step.
- **Node faults.** H231's runs saw NVLink "invalid access of peer GPU memory" errors and NCCL watchdog timeouts on
  individual `ai2/holmes` nodes. Rerunning the same command resumes from the last ephemeral save, at most 250 steps
  back.
- **Text recipes.** Text-only Olmo 3.5 SFT (`c12r`, `a4b`) stays in [`sft.py`](sft_running_guide.md) on
  open-instruct's own trainer. `sft_mm.py` does not change it.
