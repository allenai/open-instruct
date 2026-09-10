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

## Serving configuration and measured throughput

Getting these models to serve at all, and then to serve *fast*, is a deliverable
of this work in its own right -- not a prerequisite to it. What follows is
measured on `ai2/holmes` (8x B300 per node, 288 GB/GPU, driver 590 / CUDA 13.1),
vLLM 0.28.0, 1000 prompts x 8 samples, 128K context, concurrency 256.

### Measured output throughput

| Model | Active | Weights | TP | tok/s | traces/min | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-397B-A17B-FP8 | 17B | 406 GB FP8 | 4 | **4687** | 36.6 | 8000 traces in 4.27 h |
| DeepSeek-V3.2-Exp | 37B | 689 GB FP8 | 4 | **2124** | 24.0 | needs `thinking: true` |
| Kimi-K2.6 | 32B | 595 GB INT4 | 4 | **814** | 6.6 | see quantization note |
| GLM-5.2-FP8 | ~40B | 756 GB FP8 | 4 | *pending* | | `--kv-cache-dtype fp8` |

**FP8 throughput tracks active parameters; INT4 does not.** Qwen at 17B active is
2.2x faster than DeepSeek at 37B -- almost exactly the ratio of their active
sizes. Kimi has nearly the same active size as DeepSeek (32B vs 37B) but runs
**2.6x slower**, which puts the gap on the weight format rather than model scale.
Blackwell's tensor cores consume FP4/FP6/FP8 natively; a 4-bit *integer* format
has no such path. NVIDIA publishes `nvidia/Kimi-K2.6-NVFP4` precisely for this,
and Kimi's own recipe pairs the INT4 weights with 8 GPUs while reserving TP=4 for
the NVFP4 repack.

### Settings that turned out to be load-bearing

| Setting | Why |
| --- | --- |
| `--safetensors-load-strategy=prefetch` | vLLM does not recognise WEKAFS as a network FS, so it falls back to lazy mmap and reads shards serially: **321 s/shard vs 23 s/shard**, a 4-hour load instead of 25 minutes. Nodes have 2.95 TiB RAM, so prefetching a 595 GB checkpoint fits easily. |
| Full CUDA toolkit installed in-job | The `ai2/cuda13.*` images ship the runtime without `nvcc`, and there is no `-dev` variant. Every recent vLLM JIT-builds sm_103 kernels for FP8 MoE, so serving is impossible without a compiler. ~2 GB, cached on weka. |
| `VLLM_USE_DEEP_GEMM=0`, `VLLM_MOE_USE_DEEP_GEMM=0` | Both the Qwen and DeepSeek recipes specify this; DeepGEMM's JIT is the first thing to fail without a toolchain. |
| `uvx --python 3.12` | On 3.11, flashinfer's `fd_exchange` fails to import (`array.array` is not subscriptable), which breaks **every TP>1 serve** while TP=1 is unaffected. |
| FlashInfer cache on weka | Kernels are JIT-built per (version, arch); a container-local cache makes every model repeat a multi-minute sm_103 build. |
| `chat_template_kwargs={"thinking": true}` | DeepSeek-V3.2 is hybrid and defaults to **non-thinking** -- it prefills a closing `</think>`. Without this every trace is empty. Qwen's `enable_thinking` is silently ignored; the kwarg is `thinking`. |
| `--min-runtime 8h` + auto-resume | 8h is Beaker's maximum, and these jobs run 5-38h, so preemption is the expected ending. Trace-level resume plus auto-resume makes it survivable. |

### Load times (all with prefetch, warm weka caches)

| Model | Weights | Shards | Load | To `vllm ready` |
| --- | --- | --- | --- | --- |
| GLM-5.2-FP8 | 756 GB | 141 | **7 min** (1.8 s/shard) | pending |
| DeepSeek-V3.2-Exp | 689 GB | 163 | ~32 min | 68-89 min (own DSA kernels) |
| Kimi-K2.6 | 595 GB | 64 | ~24 min | 34-49 min |
| Qwen3.5-397B-A17B-FP8 | 406 GB | - | - | 52 min (cold: toolkit + kernels) |

Startup is dominated by kernel JIT, not weight I/O, once the caches are warm.

### Caveat on the Kimi dataset

Kimi's traces span two tensor-parallel widths: the first 3,641 were generated at
TP=4 and the remainder at TP=8, after TP=4 proved too slow (1.9 traces/min once
resume had drained the short prompts). Same weights, same sampling parameters,
and each sample is an independent draw at temperature 0.6, so this should not
bias the length distribution -- but TP width changes matmul reduction order, so
the traces are not bit-identical in provenance.

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

A completion can also hit the cap *after* closing its thinking block, cutting off
only the final answer. That leaves the trace length exact, so it is reported
separately (`answer cut, trace ok`) rather than counted as censoring — but it is a
warning that the cap is close to binding.

These defaults were set from measurement, not guesswork. The first real run put
Qwen3-8B at a mean of ~9.6K thinking tokens with a maximum of 30,560 against a
30,720 cap, so the cap moved to 39,000 within a 40,960 context. 40,960 is Qwen3-8B's
native `max_position_embeddings` and the binding limit across the pair
(R1-Distill-Llama-8B allows 131,072); both models get the same budget on purpose,
since extra room to think on one side would confound the comparison.

**The two chat templates differ, deliberately.** Qwen3 lets the model emit its own
`<think>`; DeepSeek-R1-Distill prefills `<think>\n` in the assistant prefix, so its
completions begin *inside* the trace with no opening tag. The parser keys off the
**closing** tag for exactly this reason, and `test_traces.py` pins both shapes.
No `--reasoning-parser` is passed to vLLM, so the literal tags survive into the
response and one parser handles both.

**Preemption safety comes from `--min-runtime`, not from priority.** These
clusters use strict priority with unallocated-only backfill, so a normal-priority
job with no guaranteed runtime runs as backfill and can lose its nodes partway
through. `--min-runtime 8h` tells the scheduler the job may not be preempted for
that window, while the job stays at normal priority. Set it to at least the
expected run length. The older `--preemptible` / `--not-preemptible` switch is
deprecated and gantry warns on it; `--no-auto-resume` is the companion flag when
a restart-from-scratch would be worse than stopping.

**vLLM version is pinned to a CUDA 12 build.** vLLM 0.20+ pins torch 2.11, built
against CUDA 13, which needs driver >= 580. `ai2/neptune` and `ai2/jupiter` run
570.x (CUDA 12.8) and fail at engine start with *"The NVIDIA driver on your system
is too old (found version 12080)"*. 0.19.1 is the newest release still on torch
2.10 / CUDA 12. Raise `--vllm-version` only for a cluster with a new enough driver.

**The serve port is called `SERVE_PORT`, not `VLLM_PORT`.** vLLM reserves
`VLLM_PORT` as the *base* of its internal port range ("if VLLM_PORT is set ... the
rest will be generated by incrementing"). Exporting it makes every data-parallel
rank derive the same rendezvous port, and startup dies with `EADDRINUSE` on
`port: 8009`. The HTTP port is passed with `--port`, and the in-job script
explicitly unsets `VLLM_PORT` in case one leaks in from the environment.

**Samples within a prompt are correlated**, so the bootstrap resamples whole prompts.
Per-trace CIs would be too narrow. The reported ICC says how much of the total
variance is "which question was asked" versus "how long this particular rollout ran".

**"Which variance?" is the whole question.** Total variance across a mixed prompt
set is dominated by the prompt mix -- a math proof and a safety refusal differ by
20x -- so it mostly measures the dataset, not the model. The comparison therefore
reports the **within-prompt** variance ratio: re-ask the *same* prompt and see how
much the length moves. That is the model property. The spread of prompt *means* is
reported too, but it largely reflects the corpus.

**Compare medians when censoring is asymmetric.** If one model hits the token cap
far more often than the other, their means are lower bounds by different amounts and
the mean difference is confounded. The median is unaffected at any censoring rate
below 50%, so it is reported with its own interval.
