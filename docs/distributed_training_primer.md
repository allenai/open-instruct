# Distributed Training Primer

A beginner's guide to the infrastructure stack that shows up across open-instruct: **Accelerate, DeepSpeed, FSDP/FSDP2, Ray, mixed precision, quantization**. Written for someone who knows ML/NLP/DL theory but isn't deeply familiar with the practical training infrastructure.

If you've just read the [OLMo-core reference](algorithms/olmo_core_reference.md) and felt some of the acronyms whoosh past, start here.

## The starting problem

The whole distributed-training stack exists to solve two problems:

1. **Compute is too slow.** Training even a 7B model on hundreds of billions of tokens takes years on a single GPU. We want N GPUs to give us ~N× speedup.
2. **The model doesn't fit.** Fine-tuning a 7B model in mixed precision needs roughly:
   - 14 GB for weights (7B × 2 bytes for bf16)
   - 14 GB for gradients
   - 56 GB for AdamW optimizer states (2 states × 4 bytes fp32 × 7B params)
   - + activations for backprop
   
   Total ≫ 80 GB, even on an H100. A 32B model is ~4× worse. You **must** split it.

The key insight: **parallelism strategies are independent and composable.** A real training run uses several at once — typically data-parallel sharding *plus* mixed precision, sometimes plus tensor parallelism, often coordinated by Ray.

## Building blocks

### Data parallelism (DDP)

The simplest pattern: replicate the entire model on every GPU, hand each GPU a different microbatch, all-reduce gradients across GPUs before each optimizer step.

- Solves *compute* (near-linear speedup).
- Does **nothing** for memory — the model still has to fit on one GPU.
- In PyTorch this is `DistributedDataParallel` (DDP).
- Good fit: small models that already fit on one GPU.

### ZeRO and sharding

If the model itself doesn't fit, you can't replicate it. Observation: most per-GPU memory in DDP is *redundant* — every replica holds an identical copy of optimizer states, gradients, parameters. ZeRO ("Zero Redundancy Optimizer", from the DeepSpeed paper) shards that redundant state across GPUs:

- **ZeRO-1**: shard optimizer states only. ~4× memory savings on AdamW.
- **ZeRO-2**: also shard gradients. ~8× savings.
- **ZeRO-3**: also shard the parameters themselves. Memory scales as O(1/N) with N GPUs.

At ZeRO-3, no GPU ever holds the full model. When a forward/backward pass reaches a layer, GPUs all-gather that layer's weights *just in time*, compute, then free the gather and move on. Communication cost goes up; memory cost plummets.

### FSDP (PyTorch's ZeRO-3) and FSDP2

**FSDP** = `FullyShardedDataParallel`, PyTorch's native implementation of ZeRO-3. Same idea, different code. FSDP and DeepSpeed ZeRO-3 are *alternatives*, not stacked.

FSDP (a.k.a. FSDP1) shipped in PyTorch 1.12. It works, but has rough edges: awkward to compose with tensor parallelism, fiddly state-dict format, hard to keep `torch.compile` happy.

**FSDP2** is the 2024 rewrite, built on **DTensor** (Distributed Tensor) — a new PyTorch primitive that describes a tensor as "logically a 7B-param matrix, physically sharded across these 64 GPUs along this axis". This makes parallelism strategies *composable*: you can say "this tensor is sharded along dim 0 by data-parallel rank AND along dim 1 by tensor-parallel rank" cleanly in one type.

OLMo-core builds on FSDP2. That's why its checkpoints are per-rank distributed-checkpoint (DCP) format rather than flat `.pt` files — DTensor knows how to stitch them back together.

### HSDP (Hybrid Sharded Data Parallelism)

Pure FSDP at >16 GPUs hits a wall: all-gather and reduce-scatter happen on *every* layer and need to involve *every* GPU. Cross-node bandwidth (InfiniBand, ~50–400 GB/s) is ~10× slower than intra-node NVLink (~600–900 GB/s). Cross-node communication starts dominating step time.

HSDP groups GPUs into:

- **Shard groups** — usually = one node (8 GPUs). Sharding happens within the group, over fast NVLink.
- **Replica groups** — across nodes. Each shard group holds a full copy of the model. Only a cheap gradient all-reduce crosses node boundaries.

This is open-instruct's default. See [OLMo-core sharding](algorithms/olmo_core_sharding.md) for the exact knobs (`--fsdp_shard_degree`, `--fsdp_num_replicas`).

### Tensor / pipeline / context parallelism

When a single layer or sequence is too big, or you want to scale further, three orthogonal axes:

- **Tensor parallelism (TP)** — split a single matmul across GPUs. The Megatron pattern: divide a linear layer's weight matrix column-wise, each GPU computes part of the output, all-gather. Communication-heavy → only works well within a node, so TP-degree ≤ 8 typically.
- **Pipeline parallelism (PP)** — split *layers* across GPUs. GPU 0 holds layers 1–8, GPU 1 holds 9–16, etc. Forward pass flows down the pipeline; needs micro-batching to keep all stages busy. Used at >70B.
- **Context parallelism (CP)** — split *sequence length* across GPUs. Useful for very long contexts (>32k tokens).

You compose them. A real 16-node run might be `DP=128 × TP=8 × CP=1` — each GPU holds 1/128 of params and 1/1 of the sequence, with TP and DP both contributing to the 1024-GPU world size.

Open-instruct's DPO trainer supports `DP + TP`. CP and PP are not currently used.

## Frameworks: who does what

The names you'll see — and why they're not all the same kind of thing.

### DeepSpeed

A Microsoft library that bundles:
- ZeRO sharding (stages 1/2/3).
- A fused optimizer.
- Its own training engine (`deepspeed.initialize` returns a wrapped model).
- Optional CPU/NVMe offload (push optimizer state to RAM/SSD for hobby setups).

Predates FSDP. Still widely used because it's mature, has good HuggingFace Transformers integration, and works well out of the box.

**In open-instruct**: the *legacy* paths (`grpo_fast.py`, `finetune.py`) use DeepSpeed. The new OLMo-core paths (`dpo.py`, `grpo.py`) replaced it with FSDP2, mainly to get `torch.compile`, TP, and DTensor composition.

### FSDP / FSDP2

PyTorch's native sharding (see above). **What OLMo-core uses.**

### HuggingFace Accelerate

Important: Accelerate is **not** a parallelism strategy — it's a thin **wrapper** that lets you write training code once and switch the underlying backend (DDP / DeepSpeed / FSDP) via config. It handles:

- Initialising `torch.distributed`.
- Gradient accumulation, mixed precision, device placement.
- Forwarding your config to DeepSpeed or FSDP under the hood.

`finetune.py` uses Accelerate + DeepSpeed. Accelerate is the convenience layer; DeepSpeed is doing the sharding.

You'll see Accelerate everywhere in HF examples. It's optional — OLMo-core skips it entirely and talks to `torch.distributed` directly.

### Ray

A different beast. Ray is **not** a parallelism strategy — it's a **distributed actor framework**. Mental model: "asyncio across machines". You define a Python class, decorate it with `@ray.remote`, and Ray spawns it as a process (possibly on another machine) and lets you call its methods over the network.

Why open-instruct uses Ray for GRPO: RL fine-tuning has *multiple distinct components* running concurrently:

- Trainer process(es) holding the policy weights (FSDP-sharded).
- Inference process(es) running vLLM to generate rollouts.
- Reward model process(es).
- A driver coordinating them.

These can't share one Python process — different GPU requirements, different lifecycles, vLLM has its own CUDA context. Ray gives you a clean way to spawn them as separate processes, pin them to specific GPUs, and ship tensors between them.

`PolicyTrainerOLMoCoreProcess` in [grpo_olmo_core_actor.py](../open_instruct/grpo_olmo_core_actor.py) is a Ray actor — each instance owns one FSDP shard of the policy and exposes methods like `train_step()` and `sync_weights_to_vllm()` over Ray's RPC.

DPO doesn't use Ray: it's a single-process training job with no separate rollout / reward model.

**Important**: Ray sits *above* FSDP. Each Ray actor internally runs FSDP across its GPUs. Ray doesn't shard models; FSDP shards models. Ray *coordinates* processes.

## Number representation

Orthogonal to all the parallelism above — these are about how many bits each weight takes.

### The dtypes

| Dtype | Bytes | Range | Use |
|---|---|---|---|
| fp32 | 4 | ±1e38, very precise | Master weights, gradient reductions |
| fp16 | 2 | ±6e4, easy to overflow/underflow | Older mixed precision (V100 era) |
| bf16 | 2 | ±3e38 (same as fp32), less precise | **Current training default on A100/H100** |
| fp8 | 1 | small, two flavours (E4M3, E5M2) | Leading-edge training on H100, inference |
| int8 | 1 | -128..127 | Inference quantization (GPTQ, AWQ) |
| int4 | 0.5 | -8..7 | Inference + QLoRA fine-tuning |

bf16 dominates because it has fp32's dynamic range (so gradients don't underflow) with half the memory.

### Mixed-precision training

You don't usually train *purely* in bf16. The standard recipe:

1. Keep a master copy of weights in **fp32**.
2. Cast to **bf16** for forward + backward pass (compute is 2× faster, memory 2× smaller).
3. Compute gradients, all-reduce them in **fp32** (avoids precision loss when summing across many GPUs).
4. Update the fp32 master weights.

OLMo-core configures this via `param_dtype=bfloat16, reduce_dtype=float32` on the data-parallel config. That's the canonical setting; you almost never change it.

### Quantization

Distinct from mixed precision:

- Mixed precision: 16-bit *and* 32-bit, switching contextually during training.
- Quantization: 8-bit or 4-bit *everywhere*, with explicit calibration / dequantization at compute boundaries.

Two main use cases:

1. **Inference quantization** (most common) — take a trained bf16 model and post-train-quantize to int8 or int4 for cheaper inference. AWQ, GPTQ, GGUF are quantization formats / methods.
2. **QLoRA fine-tuning** — load the base model in 4-bit (frozen), train small LoRA adapters on top in bf16. Lets you fine-tune a 70B model on one 48 GB GPU.

Open-instruct does **not** quantize during training — it's full bf16. Quantization shows up only at inference time, when serving trained models.

### `torch.compile`

Different category, mentioned because OLMo-core depends on it. `torch.compile` is a JIT that lowers PyTorch model code into fused, optimized kernels (via Inductor / Triton). Usually 1.3–2× speedup with one line.

Catch: it's finicky, especially with FSDP and dynamic shapes. FSDP2 was designed to play nicely with `torch.compile`; FSDP1 wasn't. That's a major reason OLMo-core picked FSDP2.

## Kernels and memory tricks

Parallelism splits work across GPUs. Kernel-level optimizations make each GPU *itself* faster and lighter. These are arguably the biggest source of practical speed-ups in the last few years.

### Flash Attention

The textbook attention computation materializes the full `(seq_len × seq_len)` attention matrix in HBM (the GPU's main memory). For an 8k-token sequence that's 64M entries *per head* — both the memory cost and the HBM↔SRAM data movement dominate runtime. Standard attention is bandwidth-bound, not compute-bound.

**FlashAttention** (Tri Dao, 2022) is a fused attention kernel that:

- Computes attention in **tiles** that fit in on-chip SRAM (~100 KB).
- Never materializes the full attention matrix in HBM.
- Streams Q, K, V tiles through SRAM, accumulating the softmax-normalized output on-the-fly using the **online softmax** trick (rescale running sums incrementally).

Result: **O(n) memory** instead of O(n²), and 2–4× faster, with identical numerics. It is the single largest practical reason we can train 8k+ context models on commodity hardware.

Versions:

- **FlashAttention-1** (2022): the original tile-based algorithm.
- **FlashAttention-2** (2023): better warp specialization, ~2× over FA-1.
- **FlashAttention-3** (2024): H100-specific, exploits TMA (tensor memory accelerator) and FP8 paths.

In OLMo-core you pick via `AttentionBackendName`: `flash_2`, `flash_3`, or `torch` (the fallback PyTorch SDPA). Open-instruct auto-detects the best available in `model_utils.detect_attn_implementation()`.

There's also a family of variants: **FlexAttention** (PyTorch's compilable attention DSL — lets you write custom masking and have it lowered to a Flash-like kernel), **xFormers** (Meta's memory-efficient attention, predates FlashAttention but similar idea), **SDPA** (PyTorch's built-in `scaled_dot_product_attention` which dispatches to flash / memory-efficient / math backends).

### Activation checkpointing

The forward pass stores intermediate tensors needed by the backward pass — the **activations**. For a transformer this scales as `batch × seq × hidden × num_layers`, often the dominant memory cost during training (more than weights or optimizer state at long context).

**Activation checkpointing** (a.k.a. gradient checkpointing): during forward, *don't* save all activations — only save the inputs at block boundaries. During backward, **recompute** the activations on the fly from those saved inputs.

Trade: ~30% more compute, but activation memory drops from O(num_layers) to O(√num_layers) or even O(1) depending on the policy.

OLMo-core supports several modes:

- **Full**: checkpoint every transformer block.
- **Selective**: checkpoint specific layer types (e.g., attention only).
- **Budget mode** (default): you set `activation_memory_budget ∈ [0.0, 1.0]` and PyTorch's compiler chooses what to recompute to fit the budget. `1.0` = no checkpointing, `0.5` = roughly half the activations recomputed, `0.0` = recompute everything.

Budget mode only works when `compile_model=True` — it's part of the inductor pipeline. See [PyTorch's blog on AC techniques](https://pytorch.org/blog/activation-checkpointing-techniques/).

### Fused kernels and Triton

Many GPU operations spend most of their time in memory traffic, not in arithmetic. **Fused kernels** combine multiple operations (e.g., bias-add + activation + dropout) into one CUDA kernel so intermediate tensors never leave registers / SRAM.

Examples you'll see:

- **Fused AdamW** — the optimizer update is one kernel instead of ~5 separate elementwise ops. Enabled by `fused_optimizer: true` in OLMo-core.
- **Fused softmax**, **fused RMSNorm**, **fused cross-entropy + z-loss** — common in modern training stacks.
- **FlashAttention** is itself a giant fused kernel.

These are often written in **Triton** (OpenAI's Python-like GPU kernel DSL), which is what `torch.compile`'s Inductor backend uses to autogenerate fused kernels. You don't write Triton normally; you benefit from it through `torch.compile`.

## Collective communication

Every parallelism strategy ultimately boils down to a handful of GPU-to-GPU communication primitives, implemented by **NCCL** (NVIDIA's Collective Communication Library):

- **All-reduce** — every GPU contributes a tensor; after the call, every GPU has the *sum* of all contributions. Used in DDP to average gradients across replicas. Cost: ~`2 × tensor_size × (N-1)/N` bytes per GPU.
- **All-gather** — every GPU has a *shard*; after the call, every GPU has the full concatenated tensor. Used by FSDP to materialize a layer's full weights before forward / backward.
- **Reduce-scatter** — inverse of all-gather. Every GPU has a full tensor; after the call, every GPU has its shard of the sum. Used by FSDP to scatter gradients.
- **Broadcast** — one GPU's tensor is copied to all others.
- **All-to-all** — every GPU sends a different shard to every other GPU. Used by MoE routing.

The key fact: `all-reduce ≈ reduce-scatter + all-gather`. FSDP's per-step communication is exactly that pair, split so the all-gather can be overlapped with compute.

Why it matters for performance: cross-node bandwidth is much slower than NVLink (intra-node), so the design goal of HSDP is to put the heavy all-gather/reduce-scatter inside a node and only do the lighter all-reduce across nodes.

## Optimizers

The optimizer choice barely matters academically (most options converge to similar quality) but matters a lot for **memory and step time**.

| Optimizer | States per param | Notes |
|---|---|---|
| **AdamW** | 2 (m, v) | The workhorse. ~8 bytes/param in fp32. What 95% of LLMs use. |
| **Adafactor** | ~0 (factored) | Memory-efficient. Used in T5. Slightly worse convergence. |
| **Lion** | 1 (sign-based) | 2024-ish. Half the memory of AdamW, competitive. |
| **Muon** | 2 + Newton-Schulz on momentum | 2024. Orthogonalizes the momentum update via a small iterative routine. Reports faster convergence than AdamW per token. Used in OLMo-core via `MuonConfig`. |
| **Shampoo / SOAP** | Full / approximate second-order | Higher memory but better convergence; emerging. |

Open-instruct uses fused AdamW by default. Muon support landed recently (PR #1533) — it's an option for new runs.

**Learning rate schedules** in OLMo-core: `LinearWithWarmup`, `CosWithWarmup`. The standard recipe is linear warmup over ~10% of steps then cosine decay to 10% of peak LR. Nothing fancy.

## Architectural patterns that matter for infra

A few model-design choices are infra-relevant — they change what parallelism strategies are feasible.

### GQA / MQA — KV cache size

The KV cache (saved K, V tensors for autoregressive generation) scales as `batch × seq × num_kv_heads × head_dim`. At long context, this dominates inference memory.

- **MHA** (multi-head attention): one KV per Q head. Big cache.
- **MQA** (multi-query): all heads share one KV. Tiny cache, slight quality loss.
- **GQA** (grouped-query, ~2023): K/V shared within groups of Q heads. The middle ground. Llama-2 70B, Llama-3, Qwen3, OLMo-2 all use GQA.

This is invisible to training infra (the kernels handle it) but determines whether you can serve long context — vLLM's paged attention is much more useful with GQA than MHA.

### Mixture of Experts (MoE)

Replace each dense FFN with `N` parallel "experts" (each an FFN). A small **router** (a linear layer) scores experts per token and sends each token to its **top-k** (usually 2) experts. Only those experts compute → you get the parameter count of a much larger model at the compute cost of a smaller one.

Why MoE is hard for infrastructure:

- Experts live on different GPUs. Routing tokens to their experts means an **all-to-all** exchange per MoE layer — fundamentally heavier than the all-gather/reduce-scatter used by dense FSDP.
- Load balancing: if some experts are picked far more than others, throughput collapses. Routing losses + capacity factors are added to encourage balance.
- Sharding policies need expert-aware logic; not every FSDP wrapper handles MoE cleanly.

This is a major reason OLMo-core exists as a separate framework: its custom routing kernels and FSDP-MoE integration are tuned for OLMoE specifically. HuggingFace + DeepSpeed MoE works but is slower per token.

## The inference side: vLLM

Inference is its own world. The pivotal piece in the modern stack — and the one open-instruct's RL pipeline depends on most — is **vLLM**.

vLLM is an LLM serving engine. Three innovations matter for our usage:

### Paged attention

Naive KV cache: allocate one contiguous block per request, sized for max sequence length. Massive memory fragmentation.

vLLM treats the KV cache like virtual memory: a request's KV is stored in **pages** (small fixed-size blocks, e.g. 16 tokens each), with a per-request **block table** mapping logical token positions to physical pages. This:

- Eliminates fragmentation — you can pack arbitrary mixes of request lengths.
- Enables sharing (e.g., prefix caching, beam search) — multiple requests can reference the same physical pages.
- Lets the engine kill requests cleanly without fragmenting memory.

### Continuous batching

Naive batched generation: form a batch, generate until *all* sequences in the batch finish, then form the next batch. Short sequences sit idle waiting for long ones.

Continuous batching: at every decode step, **slot in new requests** into the batch as old ones finish. The batch composition changes mid-generation. Combined with paged attention, this gives 5–20× higher throughput than HF `generate()`.

### Why this matters for RL

In RL fine-tuning, **rollout** (generation) is often the bottleneck — it's autoregressive and CPU-bound on tokenization. vLLM gets generation throughput up to where the gradient step actually has work to do.

In open-instruct GRPO, vLLM runs as separate Ray actors. After each policy update, the trainer pushes new weights into vLLM via NCCL (see `VLLMWeightSyncCallback` and `olmo_core_to_hf_name` in [grpo_callbacks.py](../open_instruct/grpo_callbacks.py)).

### Speculative decoding (briefly)

Use a small "draft" model to propose K tokens, verify them with the big model in parallel. 1.5–3× decode speedup, no quality loss. Not currently used in open-instruct's RL loop, but standard in production serving (vLLM supports it).

## Training-time data optimizations

### Gradient accumulation

`effective_batch_size = per_device_batch × num_devices × gradient_accumulation_steps`.

Set `gradient_accumulation_steps=k` and the trainer runs k forward+backward passes before each optimizer step, accumulating gradients. Memory cost is one microbatch; effective batch size is k× larger. Used everywhere when you want a bigger effective batch than fits.

### Sequence packing / padding-free

Naive batching pads every sequence to the batch's max length, then masks. If your data has length variance (chat, code, mixed corpora), you can waste 30–70% of FLOPs on padding tokens.

**Sequence packing**: concatenate multiple short sequences into one long packed sequence, with an attention mask that prevents cross-sequence attention. **Padding-free attention** kernels (FlashAttention's `cu_seqlens` interface) handle this natively, with zero padding waste.

Open-instruct uses padding-free collators (`padding_free_collator` module) in its OLMo-core paths — that's where this trick is wired in.

## Parameter-efficient fine-tuning (briefly)

We don't use these for the main DPO/GRPO paths, but you'll see them mentioned often:

- **LoRA** — freeze the base weights, insert small low-rank `A @ B` matrices alongside each linear layer, train only `A` and `B`. Often <1% of full params. Fast, cheap, good for adapting an instruction-tuned model.
- **QLoRA** — same but load the base in 4-bit (frozen) while LoRA stays in bf16. Lets you fine-tune a 70B model on a single 48 GB GPU.
- **DoRA**, **Adapters**, **Prompt Tuning** — variants in the same space.

Open-instruct does full fine-tuning for the OLMo-core paths. The `finetune.py` deprecated path supports LoRA/QLoRA via PEFT.

## Cheat sheet — mapping back to open-instruct

| You see | What it means | Where in this repo |
|---|---|---|
| `DDP` | Replicate model, parallel data, all-reduce grads | (not used; we always shard) |
| `FSDP1` | PyTorch's pre-DTensor ZeRO-3 | (not used) |
| `FSDP2` | DTensor-based ZeRO-3 | OLMo-core paths (`dpo.py`, `grpo.py`) |
| `HSDP` | Shard within node, replicate across | Default for both OLMo-core paths |
| `TP` / `--tensor_parallel_degree` | Split matmuls across ≤8 GPUs | DPO only |
| `DeepSpeed` | Microsoft's ZeRO. Alternative to FSDP | `grpo_fast.py`, `finetune.py` (legacy) |
| `Accelerate` | HF wrapper picking between DDP/DS/FSDP | `finetune.py` (with DeepSpeed underneath) |
| `Ray` | Process orchestration, not parallelism | GRPO (training + vLLM + driver as actors) |
| `bf16` + fp32 reduce | Standard mixed precision | All OLMo-core training |
| `int4` / QLoRA | 4-bit base + LoRA adapters | Not used here |
| `torch.compile` | JIT compilation of model graph | Enabled by default in OLMo-core paths |
| `flash_2` / `flash_3` | FlashAttention-2/3 kernel | OLMo-core `attn_backend`; auto-detected |
| **AC** / activation checkpointing | Recompute activations in backward to save memory | OLMo-core "budget mode" via `activation_memory_budget` |
| **SDPA** | PyTorch's built-in attention dispatcher | Used as fallback when flash isn't available |
| **NCCL** | NVIDIA's GPU collective comm library | Under the hood of every all-reduce / all-gather |
| **All-reduce / all-gather / reduce-scatter** | The three workhorse collectives | DDP uses all-reduce; FSDP uses gather + scatter |
| **All-to-all** | Each GPU sends a different shard to every other GPU | MoE expert routing |
| **AdamW / Muon** | Optimizers; Muon adds Newton-Schulz orthogonalization | `AdamWConfig`, `MuonConfig` in OLMo-core |
| **MoE** | Mixture-of-Experts (sparse FFN with router + top-k) | OLMoE; supported in OLMo-core |
| **GQA / MQA** | Grouped/multi-query attention → smaller KV cache | All recent supported models (Qwen3, OLMo-2/3) |
| **vLLM** | Inference engine with paged attention + continuous batching | Used as Ray actors for GRPO rollouts |
| **Paged attention** | KV cache stored in fixed-size pages, like virtual memory | vLLM internals |
| **Continuous batching** | Slot new requests into a running batch mid-decode | vLLM throughput trick |
| **Sequence packing / `cu_seqlens`** | Pack multiple short sequences into one, zero padding | `padding_free_collator` |
| **Gradient accumulation** | Multiple forward+backward per optimizer step → larger effective batch | `gradient_accumulation_steps` |
| **LoRA / QLoRA** | Low-rank adapters / 4-bit base | Not used for main training; legacy `finetune.py` only |
| **Triton** | GPU kernel DSL `torch.compile` lowers to | Indirect; you benefit via `compile_model=True` |

## How to read an open-instruct training command

Putting it together — a typical multi-node GRPO command involves all of these layers:

- **Ray** orchestrates: spawns a driver, N trainer actors, M vLLM actors, places them on GPUs.
- Each **trainer actor** runs OLMo-core's `Trainer`, which uses **FSDP2 / HSDP** to shard the policy across its GPUs. Sharding is implemented as **all-gather** (before each block's forward) and **reduce-scatter** (gradients), run by **NCCL** over NVLink within a node and InfiniBand across nodes.
- Inside FSDP2, parameters are stored in **bf16**, gradients reduced in **fp32**.
- Each transformer block is **`torch.compile`d** (with **fused kernels** including fused AdamW and FlashAttention), with **activation checkpointing** in budget mode trading recompute for memory.
- Attention runs through **FlashAttention-2 or -3** kernels, which compute attention in SRAM tiles instead of materializing the full attention matrix.
- Multiple short training examples are **packed** into one long sequence with `cu_seqlens` to avoid padding waste.
- **vLLM** actors run the policy in bf16 for generation, using **paged attention** + **continuous batching** to keep throughput high.
- After each optimizer step, weights are **synced** from trainer → vLLM via NCCL.
- A **driver process** pulls rollouts from vLLM, computes advantages, and ships them back to the trainers via Ray.

That's the whole stack. Every term in this primer maps to one box in that picture.

## Further reading

- ZeRO paper (read this first if you want the foundation): https://arxiv.org/abs/1910.02054
- PyTorch FSDP overview: https://pytorch.org/docs/stable/fsdp.html
- FSDP2 + DTensor design: https://pytorch.org/blog/pytorch-2-4-released/ and the [torchtitan](https://github.com/pytorch/torchtitan) repo
- DeepSpeed docs: https://www.deepspeed.ai/
- HuggingFace Accelerate: https://huggingface.co/docs/accelerate
- Ray docs: https://docs.ray.io/en/latest/
- FlashAttention papers: [FA-1](https://arxiv.org/abs/2205.14135), [FA-2](https://arxiv.org/abs/2307.08691), [FA-3](https://arxiv.org/abs/2407.08608)
- Activation checkpointing techniques: https://pytorch.org/blog/activation-checkpointing-techniques/
- vLLM paged attention paper: https://arxiv.org/abs/2309.06180
- Mixture-of-Experts survey: https://arxiv.org/abs/2209.01667
- Muon optimizer: https://kellerjordan.github.io/posts/muon/
- LoRA paper: https://arxiv.org/abs/2106.09685, QLoRA: https://arxiv.org/abs/2305.14314
- NCCL docs: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
