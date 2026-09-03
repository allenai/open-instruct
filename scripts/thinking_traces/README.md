# Thinking-trace length measurement

Measures the **mean and variance of thinking-trace length** produced by open-source
reasoning models on a slice of our post-training data, and compares models to each
other.

The default experiment is `Qwen/Qwen3-8B` vs `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
over `allenai/Dolci-Think-SFT-7B`.

## Layout

| File | Role |
| --- | --- |
| `../beaker/launch_thinking_traces.sh` | Gantry launcher; one Beaker job per model |
| `../beaker/run_thinking_traces_in_job.sh` | In-job: `vllm serve` → poll `/v1/models` → run the client → write `/results` |
| `generate_traces.py` | Samples prompts, calls the endpoint, records per-trace token counts |
| `analyze_traces.py` | Moments, variance decomposition, clustered bootstrap CIs, model comparison |
| `test_traces.py` | Unit tests for trace parsing and the statistics |

The Beaker scripts are adapted from `tmax`'s `launch_gen_solutions.sh` /
`run_gen_solutions_in_job.sh`. The submit-a-SHA-to-Gantry structure and the
env-var contract between the two halves are the same; the podman base-image
machinery that pipeline needed for agentic rollouts is not carried over.

## Running it

```bash
# Both models, 200 prompts x 4 samples, on 8xL40S each
./scripts/beaker/launch_thinking_traces.sh --both

# Smoke test first
./scripts/beaker/launch_thinking_traces.sh --num-prompts 8 --num-samples 2 --gpus 2

# Compare once both jobs' results are downloaded
PYTHONPATH=. uv run python scripts/thinking_traces/analyze_traces.py \
    --traces qwen3-8b=qwen.jsonl \
    --traces deepseek-r1-distill-llama-8b=deepseek.jsonl
```

The launcher submits the current git SHA, so **push your branch first**; local
dirty changes do not reach the job.

## Things that decide whether the numbers mean anything

**Prompt selection is deterministic**, a pure function of
`(dataset, revision, seed, num_prompts, max_prompt_tokens)`. That is what lets two
jobs on two different models be compared at all. `--both` reuses one set of values
by construction, and every record stores a `prompt_sha` so `analyze_traces.py`
*verifies* the two runs saw the same prompts instead of assuming it.

**Truncation censors the metric.** A trace that hits `--max-tokens` has no `</think>`;
its true length is only a lower bound, so the reported mean is a lower bound too.
The summary always prints the truncation rate alongside a completed-traces-only mean.
If that rate is more than a few percent, raise `--max-tokens` before believing the mean.

**The two chat templates differ, deliberately.** Qwen3 lets the model emit its own
`<think>`; DeepSeek-R1-Distill prefills `<think>\n` in the assistant prefix, so its
completions begin *inside* the trace with no opening tag. The parser keys off the
**closing** tag for exactly this reason, and `test_traces.py` pins both shapes.
No `--reasoning-parser` is passed to vLLM, so the literal tags survive into the
response and one parser handles both.

**vLLM version is pinned to a CUDA 12 build.** vLLM 0.20+ pins torch 2.11, built
against CUDA 13, which needs driver >= 580. `ai2/neptune` and `ai2/jupiter` run
570.x (CUDA 12.8) and fail at engine start with *"The NVIDIA driver on your system
is too old (found version 12080)"*. 0.19.1 is the newest release still on torch
2.10 / CUDA 12. Raise `--vllm-version` only for a cluster with a new enough driver.

**Samples within a prompt are correlated**, so the bootstrap resamples whole prompts.
Per-trace CIs would be too narrow. The reported ICC says how much of the total
variance is "which question was asked" versus "how long this particular rollout ran".
