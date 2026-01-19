# Bash commands
- `uv run pytest`: Run the tests.
- `make style && make quality` run the linter + formatter.
- `uv run mkdocs serve`: View the documentation locally at http://127.0.0.1:8000/
- `uv run mkdocs build`: Build the documentation to the `site/` directory.



# Workflow
- Always run the linter and make sure the tests pass before finishing a task.
- Prefer running single tests, not the whole suite, when developing.
- To run the `./scripts/train/build_image_and_launch.sh` script, you must commit the current changes.
- To launch experiment scripts, use the `build_image_and_launch.sh` script, like this: `./scripts/train/build_image_and_launch.sh $SOME_SCRIPT`.
- For GRPO, we have three test scripts:
  - `scripts/train/debug/single_gpu_on_beaker.sh`: single GPU, no tools (~8 minutes).
  - `scripts/train/debug/tool_grpo_fast.sh`: single GPU, with tools (~15 minutes).
  - `scripts/train/debug/large_test_script.sh`: two 8x GPU nodes, no tools (~32 minutes).
- For DPO, we have two test scripts:
  - `scripts/train/debug/dpo.sh`: single GPU.
  - `scripts/train/debug/large_dpo.sh`: four 8x GPU nodes.
- To run the `./scripts/train/build_image_and_launch.sh` script, you must commit the current changes.
- Launch tool use experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/tool_grpo_fast.sh`.
- Launch multi-node non-tool experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/large_test_script.sh`.
- Launch DPO experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/dpo.sh`.
- Launch multi-node DPO experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/medium_dpo.sh`.
- Launch the GPU tests with `./scripts/train/build_image_and_launch.sh scripts/train/debug/run_gpu_tests.sh`.
- If you are given a Beaker URL (beaker\.allen\.ai.*) use the Beaker CLI tool to interact with it.

# Coding conventions
- Always use `logger = logger_utils.setup_logger(__name__)` for logging.

# Documentation
To verify that documentation changes don't alter the generated output:
1. Build docs on your branch: `uv run mkdocs build && cp -r site site-branch`
2. Switch to main branch and build: `cd /path/to/main && uv run mkdocs build`
3. Compare the builds: `diff -rq site-branch /path/to/main/site`
4. If no output, the docs are identical. If differences exist, review with: `diff -r site-branch /path/to/main/site`

# TEMP: FP32 LM head rollout notes (ScaleRL alignment)
- Goal: reduce generator↔trainer logprob mismatch; apply fp32 at LM head on both sides.
- References: TRL GRPO `cast_lm_head_to_fp32` (PR #4303, #4446); vLLM patch point is `LogitsProcessor._get_logits`; vLLM plugin system can host the patch; LLaMA-Factory upcast_lmhead_output.
- Implementation in repo:
  - Trainer: `enable_fp32_lm_head()` patches output embeddings to run in fp32 without re-allocating weights (tied embeddings preserved).
  - Reference policy: same fp32 head toggle to keep KL alignment.
  - Generator: vLLM LogitsProcessor `_get_logits` patched to upcast hidden_states (+ bias) before projection.
- Flag: `--fp32_lm_head` (GRPO). Ensure vLLM engines receive the flag too.
- Sanity check (step 0): compare trainer logprobs vs vLLM logprobs on identical sequences; expect tighter deltas with fp32 head.
- OOM safety (DGX Spark): check `free -h`, kill leftover Ray/vLLM if needed, set `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"`.
- Suggested local debug: `scripts/train/dgx-spark/grpo_qwen_gsm8k.sh` (Qwen2.5-0.5B GSM8K), add `--fp32_lm_head` when testing.
- Long comparison runs: `FP32_LM_HEAD=0 ./scripts/train/dgx-spark/grpo_qwen_gsm8k.sh` then `FP32_LM_HEAD=1 ./scripts/train/dgx-spark/grpo_qwen_gsm8k.sh`; monitor `debug/vllm_vs_local_logprob_diff_*`.
- Plugin vs monkey patch: plugin is just a startup hook that can apply the same patch; in-process patch is used unless a vLLM plugin package is added.
