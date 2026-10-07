# o3i7b: Olmo 3 7B Instruct SFT, released vs TPU reproduction, on the H051 size-peer battery

Specs only; nothing has been launched. `mkspecs_o3i7b.py` writes `specs/` (22 specs: 11 cells x 2 models). Its
docstring lists every deviation (D1-D8). Both models' specs are byte-identical except `-m` and the names.

- `released`: `allenai/Olmo-3-7B-Instruct-SFT` @ `e1452fc5`, at `models/olmo3-7b-instruct-sft-released`. Every file
  was read and hashed after download, and all LFS sha256s match HF (`models/*.receipt.txt`, 2026-10-07 15:40 PDT).
- `tpu`: `models/olmo3-7b-instruct-sft-tpu-s0` (from `../export_hf.sh`; not exported yet).

Launch with `beaker experiment create specs/<name>.json --workspace ai2/olmo-instruct --name <name>`. The budget is
the workspace default, `ai2/oe-other`, as for the sources. Space launches about 4 min apart, because they share one uv
lock in the pypi cache. Run one cell per commit/harness first.

## Cells

All cells are on ai2/jupiter (H100), priority normal, thinking off, `max_model_len` 65536, bf16. Every cell uses the
released tokenizer and generation_config, with flat YaRN `hf_overrides`.

| cell | task | olmo-eval | harness | decoding | GPUs | timeout | est. GPU-h / model |
|---|---|---|---|---|---|---|---|
| aime25 | aime_2025:pass_at_32 | 25ae07ac | default | T1.0 p0.95 32K, k=16 | 1 | 24h | ~2.5 |
| aime26 | aime_2026:pass_at_32 | 25ae07ac | default | T1.0 p0.95 32K, k=16 | 1 | 24h | ~2.5 |
| hmmt26 | hmmt_feb_2026:pass_at_32 | 25ae07ac | default | T1.0 p0.95 32K, k=16 | 1 | 24h | ~2.5 |
| gpqa | gpqa_diamond:cot | 25ae07ac | default | T1.0 p0.95 32K, k=8 | 1 | 24h | ~4 |
| mmlupro | mmlu_pro:cot | 5ef9cee1 (main) | default, 4 instances | T1.0 p0.95 32K, k=1 | 4 | 24h | ~6 |
| lcb | livecodebench:lite (v3) | 0131141b | codex_python | T1.0 p0.95 (task) 32K, k=8 | 2 | 24h | ~8 |
| ifeval | ifeval | 25ae07ac | default | T1.0 p1.0 32K | 1 | 24h | ~0.3 |
| ifbench | ifeval_ood | 25ae07ac | default | T1.0, top_p unset (=1.0) 32K | 1 | 24h | ~0.3 |
| ifeval_mt_ood | ifeval_mt_ood_wildchat_unused_withRewrite | 25ae07ac | default | T1.0 p1.0 32K | 1 | 24h | ~1.5 |
| bfcl_single | bfcl + bfcl:categories | 9538453e | default, vllm_server, `tool_call_parser=olmo3` | task default (T0.001, 4096) | 1 | 24h | ~3 |
| ruler64k | ruler_all__65536 | 46faa7ee | default | battery default | 1 | 24h (minRuntime 0, backfill) | ~2 |

Estimated total: about 32 GPU-h per model, about 65 for both. The estimate scales H051's Cost table (which assumes a
thinking model) by the measured thinking-off/forced runtime ratio: 1.54/5.13 GPU-h on the H045 ifeval_mt_ood cells.
IFEval, IF-MT-OOD and BFCL use measured thinking-off source runtimes (0.23, 1.54 and 2.82 GPU-h). MMLU-Pro uses H051's
thinking-off row. If T=1.0 drives this instruct model into cap-length loops, costs approach H051's thinking-model
figures: about 110 per model, 220 for both. Refine after the first maths cell.

## Deviations from H051, and why

1. **Olmo-3.5 stack removed.** Removed: the olmoe3 plugins, ai2-olmo-core (hero-core), flash-linear-attention, the
   fa-stub, `mamba_ssm_cache_dtype`, and the plugin-only env vars. For BFCL, also the ladders tar, the hero-core clone
   and `reasoning_parser`. Stock vLLM 0.19.1 serves Olmo3ForCausalLM. The pins vllm 0.19.1 and transformers 5.14.1
   (5.16.1 for BFCL) are kept.
2. **`provider.kwargs.hf_overrides` = flat YaRN** (`hf_overrides.json`). This is required. transformers 5.x nests Olmo
   3's `rope_parameters` per layer type, and vLLM 0.19.1's olmo2.py then dies with `KeyError: 'rope_theta'`. With the
   override, the full-attention layers get YaRN x8 and the sliding layers get plain RoPE at theta 500000. Reproduced
   and fixed on CPU with transformers 5.14.1 and 5.16.1. The same override was verified on GPU for a dense 7B in
   xarch-evals (gsm8k 0.7604 vs 0.7612 in-loop).
3. **Released tokenizer and released `generation_config` for both models.** `export_hf.sh` takes
   `generation_config.json` from the *base* model, which has no `<|im_end|>` eos and no defaults. Served as-is, the TPU
   model would decode past `<|im_end|>`. The TPU training tokenizer encodes identically to the released one.
4. **BFCL `tool_call_parser` qwen3_xml -> olmo3.** vLLM's Olmo3PythonicToolParser parses the released template's
   `<function_calls>name(k=v)</function_calls>` format. olmo-eval's `patch_olmo3_tool_parser` is not used, because it
   also replaces the chat template.
5. **RULER at 65,536** (this model's window), not 131,072.
6. **New cell: ifeval_mt_ood** (H051 reused existing cells), at top_p 1.0. **OFF source for every 25ae07ac cell.** OFF
   differs from T25 only in the tokenizer, which item 3 replaces anyway. The gantry task name stays `main`.
7. **Unchanged from H051:** BFCL and RULER keep their own decoding (not T=1.0), and RULER keeps the source's minRuntime
   0 / autoResume (backfill). LCB v3 lite and BFCL v3 single-turn are not the bands' versions.

## Validation (`validate/`, CPU only, no GPU run yet)

- `check_specs.py`: passed. It checks that every spec parses, that keys and durations match source-accepted shapes,
  that the images and gantry datasets resolve, that every referenced `/weka` path exists (except the tpu dir), that
  the commits exist, and that the pairs are identical. `beaker experiment create` has no dry-run flag (CLI v1.5.333,
  beaker-docs).
- `check_released.py`: transformers loads the config, and the tokenizer renders a 2-turn chat and a tool prompt. vLLM's
  ModelConfig fails without the override and is correct with it (logs for transformers 5.14.1 and 5.16.1).
- `check_engine_args.py`: run on the released model and on a stand-in for the TPU export (export_hf.sh layout). Both
  resolve to the same RoPE, eos `[100265, 100257]` and defaults. Without item 3, the stand-in gets eos None.
- `check_bfcl_server.py`: olmo-eval 9538453e builds the vLLM server command, vLLM 0.19.1 parses it, and the olmo3
  parser extracts calls in the template format.

The checks used a scratch venv with vllm 0.19.1, transformers 5.14.1, huggingface-hub 1.16.1 and an olmo-eval
9538453e checkout installed `-e`. To rebuild it, use a uv venv on `/opt/scratch`, not WEKA.

## Risks

- No GPU run of these specs yet. Launch `o3i7b-released-ifeval-t10` first (~0.3 GPU-h). It exercises the YaRN
  override, the stack removal and the install command.
- Evaluate the TPU export within about 10 days of writing it, or re-copy it first (WEKA cold tier). The released
  snapshot was written 2026-10-07.
- The olmo3 parser fails on single-quoted strings that contain JSON (vllm#32534). The released template writes JSON
  (double-quoted) values, so this should be rare.
- RULER cells are unallocated (minRuntime 0) and may sit in jupiter's FIFO. Setting minRuntime 8h, like the other
  cells, would make them allocated.
- `enforce_eager=true` is kept from the sources. Dropping it would speed up the dense model but would no longer match
  the sources.

**Edited after generation (2026-10-07):** both `ruler64k` specs were set to `minRuntime 8h`, `autoResume false`, like every other cell, instead of the source's backfill-only `0s` / `autoResume true`. Scheduling only; decoding and task are unchanged. Re-running `mkspecs_o3i7b.py` undoes this.
