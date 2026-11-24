# Evaluating Olmo 3.5 multimodal SFT checkpoints

One launcher for any multimodal SFT checkpoint, GPU-trained or TPU-trained and converted to OLMo-core: it exports
the checkpoint ([`scripts/export/olmo35_mm`](../../export/olmo35_mm/README.md)) and launches the same Beaker evals
H231 (GPU) and H271 (TPU replica) used, so new results line up with theirs. `compare.py` builds the tables
(run vs reference, with binomial SE; optionally against two GPU seeds).

## One checkpoint

```bash
# print every command / spec, launch nothing (olmo-eval also dry-runs each vision suite)
DRY_RUN=1 OI_JOB_DIR=/weka/.../open-instruct bash scripts/eval/olmo35_mm/run_eval.sh <label> <STEP_DIR>
# launch
OI_JOB_DIR=/weka/.../open-instruct bash scripts/eval/olmo35_mm/run_eval.sh olmo35-mmsft-4n-emo256k-s10962 \
    /weka/olmo-3p5-post-checkpoints/<user>/olmo35-mm-sft/checkpoints/olmo35-mmsft-4n-emo256k/step10962
```

`STEP_DIR` is an OLMo-core DCP step dir as Beaker jobs see it. `OI_JOB_DIR` is this open-instruct checkout at the same
commit, on Weka: the Beaker jobs run its `scripts/export/olmo35_mm/*_job.sh` and `scripts/eval/olmo35_mm/sanity_job.sh`.
Re-running the same command resumes (finished stages are recorded under `$EVAL_HOME/state/<label>`).

| Stage | Where | What |
|---|---|---|
| `export` | Beaker, 1 GPU | `export_all_job.sh`: verified `olmo-core-export`, then `export/hf_text` (+ smoke). Later stages wait for its `MMSFT-VISION-EXPORT-OK` / `TOKENIZER-CHECK OK` log markers |
| `sanity` | Beaker, 1 GPU | `sanity_job.sh` / `sanity_vlm.py`: a COCO image and three questions through `olmo_core_vlm` (`SANITY_REF=` adds a reference export side by side) |
| `vision` | Beaker | olmo-eval `scripts/olmo35_mmsft/eval_vision.sh`, 500 items/task, 2 GPUs: `docs24` (ChartQA, DocVQA, TextVQA, InfoVQA at 24 crops), `core` (VQAv2, AI2D, CountBenchQA, RealWorldQA), `list` (MMMU, MuirBench, MMIU, BLINK, MMMU-Pro) |
| `emocheck` | Beaker, 1 GPU | `emo_check_job.sh`: HF text export vs OLMo-core logits with and without the EMo router (SKIP for non-EMo runs) |
| `text` | Beaker | `launch_text_cells.py`: the recorded H182 text cells on `hf_text`. `TEXT_CELLS=full` (default): GPQA, GPQA-main, IFEval, IFBench, IFEval-MT, Multi-IF, MATH x7, HMMT26, RULER 64K/128K, RULER-plus (aggqa, mvmqvt, niah-smk), and BFCL (single + 4 multi-turn) on `BFCL_CLUSTER`/`BFCL_WORKSPACE`. Intermediate checkpoints in H231/H271 ran `TEXT_CELLS="ruler64k ifeval"` |

Everything else is an environment variable with the 2026-11 session's value as default; the header of
[`run_eval.sh`](run_eval.sh) lists them (OLMo-core / olmo-eval / ledger checkouts, workspace, budget, clusters,
priority with urgent -> high -> normal fallback, image).

External pieces the launcher drives, not copied here:

- **olmo-eval** (allenai/olmo-eval, branch `jasonr/olmo35-mmsft-eval`; H231/H271 used
  11dce79286f3787238e4d051e137b534e6b7f82e): `scripts/olmo35_mmsft/eval_vision.sh` (the `olmo_core_vlm` provider and its
  olmo35 prompt layout live in olmo-eval's source, and the vision job runs that checkout's pushed HEAD), and
  `text_cells.py`'s `build()`, which rewrites a recorded spec for a new model.
- **olmo-post-training-ledger** `experiments/h182/eval/{specs,holmes-cu130/specs}`: the recorded text cell specs
  (task, sampling, olmo-eval commit, vLLM, plugins).
- **OLMo-core #942** checkout on Weka for the 1-GPU jobs, and `OLMO_CORE_SHA` for the vision jobs.

## TPU checkpoints

Convert first with `scripts/tpu/maxtext/olmo35-mm/export_to_olmocore.sh` (MaxText -> OLMo-core DCP, then the
bitwise round-trip check; see `docs/algorithms/olmo35_mm_sft_tpu.md`), then run this launcher on the DCP. Name the label by the OLMo-core-equivalent step: MaxText's checkpoint
dir `S` is the state after `S + 1` updates (`checkpoints/5500` -> `s5501`, compared with GPU s5481;
`checkpoints/10961` -> `s10962`).

## Tables

```bash
# TPU vs GPU at the same step
python scripts/eval/olmo35_mm/compare.py --run GPU=$EVAL_HOME/jobs/<gpu label>.log --run TPU=$EVAL_HOME/jobs/<tpu label>.log
# seed control: TPU against two GPU data seeds; flags |TPU − Ḡ| > |seed Δ| + 2 SE
python scripts/eval/olmo35_mm/compare.py --run G1=... --run G2=... --run TPU=... --seed-pair G1 G2 --out TABLE.md
# the 2026-11 session's per-arm sweep logs: NAME=/root/mmsft-eval/sweep-<arm>.log@<step>
```

Scores come from each Beaker job's results table (`beaker experiment logs`); IFEval is prompt-level strict from the
job's results. SE is binomial per run, an upper bound for ANLS / VQA-score metrics. Run against H231/H271's logs,
`compare.py` reproduces the recorded s10962 seed control (one flag, `bfcl_parallel`, pre-stopfix).

## Provenance

Generalized from the scripts that ran H271 (olmo-post-training-ledger `experiments/mmsft/tpu/`: `run_tpu_eval.sh`,
`run_gpu_seed_eval.sh`, `compare_tpu_gpu.py`, `compare_seeds.py`, `sanity_job.sh`, `sanity_vlm.py`, `export/*`).
Changes: one launcher for GPU and TPU-converted checkpoints (the TPU-only stage/convert/verify stages moved to the
TPU tooling), per-label job logs, paths as environment variables, the stop-token fix in the export, EMo check skipped
for non-EMo runs, and the text cells' OLMoE3 plugins installed from this repo (`scripts/export/olmo35_mm`) at an open-instruct commit,
never from scaling-ladders.
