# Serving large MoE reasoning models on B300

Answers one question: **to maximize throughput, how many B300s should a model get,
what exact vLLM configuration, what memory does it consume, and what throughput
should we expect from memory bandwidth?**

It also answers it *reproducibly* — `plan_vllm_config.py` derives the answer for
any HuggingFace repo id, so a new model is one command rather than a research
project.

Consolidated from three independent research passes plus our own measured runs on
`ai2/holmes` (8x B300, 288 GB/GPU, 8 TB/s HBM, driver 590 / CUDA 13.1, vLLM 0.28).

---

## Bottom line: one node, 8x B300, 128K context

| Model | Weights | GB/GPU | KV/GPU | Concurrent seqs | Est. output tok/s |
| --- | --- | --- | --- | --- | --- |
| **Qwen3.5-397B-A17B-FP8** | 406 GB | 51 | 209 | ~759 | **11,200-20,200** |
| **DeepSeek-V3.2-Exp** | 689 GB | 86 | 173 | ~264 | **4,500-8,100** |
| **GLM-5.2-FP8** | 755 GB | 94 | 165 | ~197 | **3,600-6,500** |
| **Kimi-K2.6** (INT4) | 532 GB | 66 | 193 | ~295 | **3,000-5,300** |

**All four fit on a single node.** None needs more than 8 GPUs, and none should
get fewer: at 4 GPUs the three MLA models drop to 11-37 concurrent sequences,
which is why our TP=4 runs crawled.

### Exact commands

```bash
# Qwen3.5-397B-A17B-FP8  — GQA, so DCP buys nothing; plain TP is optimal
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 --max-model-len 131072 --async-scheduling \
  --language-model-only --enable-prefix-caching

# DeepSeek-V3.2-Exp / GLM-5.2-FP8 / Kimi-K2.6 — MLA, so DCP is the dominant lever
vllm serve <repo> --tensor-parallel-size 8 --decode-context-parallel-size 8 \
  --kv-cache-dtype fp8 --max-model-len 131072 --async-scheduling \
  --enable-prefix-caching --trust-remote-code
```

Environment, on every model:

```bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0   # see caveat below
uvx --python 3.12 vllm==0.28.0 serve ...                # 3.11 breaks every TP>1 serve
```

Per model, additionally: DeepSeek needs `--tokenizer-mode deepseek_v32` **and**
`chat_template_kwargs={"thinking": true}` (it is hybrid and defaults to
non-thinking — without it every trace is empty). Qwen needs
`--language-model-only` (it is a multimodal checkpoint we use as text-only).

---

## The five findings that decide the configuration

**1. MLA KV cannot be sharded by tensor parallelism — use DCP.** All three
analyses verified in vLLM source that `get_num_kv_heads()` returns 1 for MLA
regardless of TP, so TP *replicates* the latent cache N times.
`--decode-context-parallel-size` shards it along the sequence axis instead. For
GLM on 8x B300 this is 25 vs 197 concurrent sequences at 128K — an 8x difference
from one flag, on identical hardware. This is the single largest lever.

**2. INT4 gets no compute on any current NVIDIA datacenter GPU.** Kimi-K2.6 is an
INT4 `compressed-tensors` W4A16 checkpoint (only the 384 routed experts;
attention, shared experts and lm_head stay BF16) — the name says nothing about
it. Native INT4 tensor cores were dropped after Ampere and did not return:
`tcgen05.mma` has no INT4 kind, and vLLM's Marlin kernels issue
`mma...f16.f16.f32`. So it is a storage and bandwidth win with **no FLOPs win**,
and Kimi is the only one of the four that becomes compute-bound. One pass also
found Marlin has no SM100 SASS target, so on Blackwell it JITs Ampere PTX. On
B300 prefer `nvidia/Kimi-K2.6-NVFP4`, which is what the vendor recipe uses.

**3. Expert parallelism is the right decomposition for MoE — but DCP matters
more here.** `--data-parallel-size N --enable-expert-parallel` shards experts
(97-98% of parameters in all four models) and replicates only the dense stack.
Vendor recipes lead with it, and DeepSeek's states plainly that *"the kernels are
mainly optimized for TP=1... simple TP works and is more robust, but the
performance is not optimal."* On a single 8-GPU node our numbers put TP+DCP
slightly ahead of DP+EP because DP replicates the dense stack 8 times; across
multiple nodes DP+EP wins. `--enable-ep-weight-filter` additionally lets each
rank read only its own expert shard from disk — 116 GB instead of 755 GB per rank
for GLM.

**4. FP8 KV is worth ~1.7x concurrency, and costs more than half.** vLLM's FP8
MLA KV is **656 B/layer/token, not 576**: the latent is FP8, per-block FP32
scales ride along, and the RoPE half stays BF16. Using 576 overstates
concurrency by ~14%. DeepSeek's recipe recommends fp8 for long requests and
bf16 for short — ours are long.

**5. B300's advantage over B200 is capacity, not speed.** Identical 8 TB/s
bandwidth and identical FP8 rates; decode is memory-bound for all four models, so
B300 buys single-node deployment rather than throughput. Its INT8 rate is *cut*
~24-27x versus B200, which is a trap for W8A8 checkpoints (harmless for these
four). NVLink domain size matters more than GPU generation: the same B200 silicon
differs 1.4-3.7x per GPU between HGX 8-GPU and NVL72.

---

## Startup: install the prebuilt FlashInfer wheels

`uvx vllm` pulls in `flashinfer-python` only. `flashinfer-cubin` is excluded from
vLLM's wheel dependencies and `flashinfer-jit-cache` ships solely in the official
Docker image -- so by default FlashInfer compiles **every kernel from source with
nvcc** during vLLM's warmup run.

Measured across five sequential boots on a single node, which removes the 8x
node-to-node variance by construction (Beaker `01M28P2590XZZDYT2SQ4G8DGX0`,
Qwen3.5-35B-A3B-FP8, 1 GPU):

| | time to ready | warmup run | peak nvcc procs | NVIDIA CDN fetches |
| --- | --- | --- | --- | --- |
| without the wheels | 1790 s | 1450 s | 52 | 64 |
| **with the wheels** | **290 s** | **11.5 s** | **0** | **0** |

**~25 minutes per server start, an 8x speedup.** Weight loading, torch.compile and
CUDA-graph capture are identical between arms -- the entire difference is the
warmup run. The same backends are selected either way (FLASHINFER attention,
FLASHINFER_TRTLLM FP8 MoE, trtllm-gen decode on sm_103a), so nothing is traded
away.

```bash
uvx --python 3.12 \
  --with flashinfer-cubin==0.6.16.post3 --with flashinfer-jit-cache==0.6.16.post3 \
  --index-strategy unsafe-best-match \
  --extra-index-url https://flashinfer.ai/whl/flashinfer-cubin/ \
  --extra-index-url https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/ \
  vllm==0.28.0 serve ...
```

Neither wheel is on PyPI at this version; cu130 matches vLLM 0.28's own
Dockerfile. About 2.4 GB, ~40 s incremental env build.

Two consequences. `VLLM_USE_FLASHINFER_SAMPLER=0` is no longer needed -- it
existed only because that sampler's JIT wanted nvcc. And the default path makes
**64 runtime HTTPS calls to `edge.urm.nvidia.com`** mid-boot to fetch cubins,
which the wheels remove: a reliability win on restricted-egress nodes,
independent of speed. Relatedly, `has_nvidia_artifactory()` probes that host with
a 5 s timeout and silently falls back to a slower attention path on failure.

**Keep the in-job CUDA toolkit as a fallback.** Two boots served in under five
minutes with no nvcc on PATH at all, so it is droppable in principle -- but the
jit-cache wheel ships sm_103 builds of `fused_moe_103` and `fp4_quantization_103`
while carrying only sm_100 of `gemm`, `fused_moe_trtllm` and `fmha_cutlass`.
Qwen3.5 was fully covered; DeepSeek-V3.2 (MLA+DSA), Kimi-K2.6 (INT4/Marlin) and
GLM-5.2 (MLA, FP8 KV) request different specialisations and were not tested. Drop
the toolkit per model, and only once that model boots with zero compiler
activity. The test also ran at TP=1; the saving should carry to TP>1 since the
kernel set depends on architecture rather than parameter count, but that is
untested.

## Correction: `VLLM_USE_DEEP_GEMM=0` does not disable DeepGEMM

For DeepSeek-V3.2 and GLM-5.2 it disables only the *warmup*. The DSA indexer
raises without the DeepGEMM package present, and `has_deep_gemm()` is an import
check rather than an env-var check, so the kernels are still built -- the JIT
moves to the first real request, where it stops appearing in time-to-ready and
instead shows up as a slow first batch. An earlier version of this document
implied the flag switched DeepGEMM off; it does not.

## Reproducing this for a new model

```bash
uv run --no-project --with huggingface_hub python \
    scripts/serving/plan_vllm_config.py <hf-repo-id> [--gpu b300|b200|h100]
```

It derives everything from the model's own `config.json` geometry and
<!-- cross-checks against the parameter census the HF API reports for the safetensors -->
shards. What it encodes, each of which cost us real time to learn:

| Trap | What it does |
| --- | --- |
| Hybrid attention | Qwen3.5 has 60 layers but only **15 cache KV**; the rest are Gated DeltaNet. Counting all layers overstates KV/token 4x. |
| Recurrent state | Those linear layers carry a **180 MiB/sequence** state independent of context, invisible to KV-only accounting. |
| MLA replication | TP multiplies MLA cache cost by the TP degree unless DCP is used. |
| FP8 MLA KV | 656 B/layer/token, not 576. |
| INT4 | No tensor-core path; costed at BF16 compute despite ~0.5 B/param storage. |
| Non-uniform checkpoints | An "FP8" repo can hold hundreds of BF16 modules; bytes/param comes from the real dtype mix, not the repo name. |
| MTP heads / vision towers | A geometry-vs-census check flags them instead of letting them inflate active-parameter counts. Vendor param counts are typically **HF total minus the MTP layer**. |

---

## Measured: DCP validated, and a correction the roofline missed

GLM-5.2 was re-run on identical prompts with the only change being
`--decode-context-parallel-size 8` alongside `--tensor-parallel-size 8`. vLLM's
own counters, sampled during generation:

```
Avg generation throughput: 4706.9 tok/s   Running: 162 reqs  Waiting:  94  KV: 99.9%
Avg generation throughput: 3042.7 tok/s   Running:  90 reqs  Waiting: 166  KV: 99.8%
Avg generation throughput: 2127.0 tok/s   Running:  54 reqs  Waiting: 202  KV: 99.1%
Avg generation throughput: 1428.3 tok/s   Running:  30 reqs  Waiting: 226  KV: 96.9%
```

**Prediction: ~197 concurrent sequences, 3,594-6,468 tok/s. Measured: 162
concurrent, 4,707 tok/s.** Both inside the predicted band, against a pre-DCP rate
of roughly one completed trace every three minutes. DCP is confirmed as the
dominant lever for MLA models.

**The correction: this workload is KV-capacity-bound, and it degrades over
time.** `Running` falls from 162 to 30 while `Waiting` climbs from 94 to 226 and
KV utilisation stays pinned at 97-100%. The planner assumes a fixed average
context per sequence; in a long-reasoning workload each surviving sequence's
cache keeps growing, so concurrency decays as the short traces retire and the
long ones accumulate. Peak throughput is therefore a poor predictor of
end-to-end wall clock -- GLM sustained 4,707 tok/s early and 1,428 tok/s later
in the same run.

Two consequences worth acting on:

* **Client concurrency above what KV can hold buys nothing.** At `Waiting: 226`
  of 256 in flight, the extra requests are queued, not served. Concurrency should
  be sized to measured KV capacity, not set optimistically.
* **`--max-model-len` is a throughput knob, not just a correctness one.** Qwen's
  p99 trace was 45,437 tokens and DeepSeek's 37,113 against a 131,072 cap. Sizing
  the context to the measured p99 rather than the model maximum would roughly
  double concurrency for the MLA models. Verify the truncation rate stays near
  zero before doing this.

## Calibration and honesty

The throughput model is a memory-bandwidth roofline: per decode step each GPU
streams the weights it holds — for MoE only the experts actually routed to, which
approaches all of them as batch grows — plus the KV it reads for its own
sequences. It is reported as a **band, not a point**.

Against our own measured runs (4x B300, TP=4, 128K ctx, concurrency 256):

| Model | Predicted | Measured | Ratio |
| --- | --- | --- | --- |
| DeepSeek-V3.2-Exp | 2,092 | **2,124** | 0.98x |
| Kimi-K2.6 (INT4) | 1,310 | **814** | 1.61x |
| Qwen3.5-397B-FP8 | 8,295 | **4,687** | 1.77x |

No single constant absorbs a 0.98-1.77x spread. The model is most accurate when
KV traffic dominates the decode step and over-predicts when weight traffic does —
the regime where expert-routing locality, kernel launch overhead and all-reduce
cost are least well captured. **Ratios between configurations are far more
reliable than absolute numbers**, since the fudge factors cancel; treat the
absolutes as ±40% and the comparisons as sound.

Independent published anchors disagree in both directions: one pass matched
LMSYS's large-scale-EP DeepSeek decode measurement to within 0.4% (2,775
predicted vs 2,785 measured), while vLLM's reported 25K tok/s/GPU for Qwen3.5 on
GB200 NVL72 is ~6x higher than any of our figures — but at short context, high
batch, NVFP4 weights and P/D disaggregation. Beware the widely-cited "Kimi K2.6
6,091 tok/s/GPU on 8x B200" number: it is prefill-dominated (~67k in / ~400 out)
and measures the NVFP4 checkpoint, not the INT4 one.

### Biggest remaining uncertainties

1. Speculative decoding is excluded; MTP heads plausibly add 40-100%.
2. DSA (DeepSeek/GLM sparse attention) *enlarges* the KV cache — one pass measured
   +23% for DeepSeek, +6% for GLM via an indexer cache of ~132 B/token/layer. The
   planner does not yet model this, so its MLA concurrency figures are optimistic.
3. B300 capacity is ambiguous in NVIDIA's own materials (262.5 / 277.8 / 288 GiB).
   The planner uses what the device reports on our nodes: 275,040 MiB.
4. The planner assumes a single NVLink domain; multi-node changes the answer
   toward DP+EP.
