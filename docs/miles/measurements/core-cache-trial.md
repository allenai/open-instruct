# Actual Core cold/restored compiler-cache qualification

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

This opt-in trial extends the [compiler-cache lifecycle](../compiler-cache.md) from a single Triton kernel to one actual Core optimizer update. It has not been launched or GPU-qualified yet.

After committing the source, use the normal build wrapper:

```bash
MILES_BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks \
  ./scripts/train/build_image_and_launch.sh --miles \
  scripts/train/debug/miles_core_cache.sh
```

The launcher resolves image aliases to an immutable Beaker image ID before fingerprinting. Render an already-created image without launching:

```bash
python -m scripts.miles.launch_core_cache_trial IMMUTABLE_IMAGE_ID --render-only
```

Placement is fixed at one Holmes GPU, `ai2/open-instruct-dev`, urgent priority, 30-minute minimum runtime, and 45-minute timeout. Published immutable caches live under `/weka/oe-training-default/robertb/tmp-30d/open-instruct-core-compiler-cache`; mutable caches and fixture copies are node-local. The trial does not modify current SFT launchers or configs.

## Workload and controls

1. Bootstrap the existing `tests/miles/ep_contract.py` random tiny KDA/full-attention latent-MoE HF fixture once, then copy identical bytes into independent cold/restored directories. Each directory initially lacks `routes.pt`.
2. Run `ep_contract.run(combined, EP1, checkpointing=False, capture_gradients=True)` once in each of two fresh `torchrun` process trees. Both perform the same route capture, replay scoring, actor scoring and optimizer update. The common fingerprint includes the actual runtime/source trees, HF architecture and full expected Core/MILES configuration. An observation hook checks the configuration constructed by the production contract against that expected configuration before training starts.
3. Use the real Core trainer, MILES loss, auxiliary coefficient 0.01, z-loss coefficient 1e-5, routing replay, and production Adam optimizer. The observer records tensors and compiler activity; it does not replace probabilities, advantages, gradients or optimizer arithmetic. Existing native gradient/Adam self-checks run in each arm.
4. Compare initial/final model tensors, all three scoring calls and their four response tensors, routing IDs, pre/post-clip gradients, FP32 masters and both Adam moments. **Exact equality is required initially.** A difference fails with retained arrays and maximum absolute errors; no EP2 or cross-backend tolerance is borrowed.
5. Require an actual Triton restore hit, positive observed Triton group-cache reads, and fewer real `FileCacheManager.put` calls in the restored process. An archive restore alone is insufficient. Other families' hits/misses/rejections are reported individually; they are not presumed warm.

The learning workload is four unequal fixed sequences with masked response tokens and one combined policy/auxiliary update. It uses Torch attention and native Core kernels; there is no SGLang, Ray rollout, reward verifier or W&B session. The inactive clipping threshold matches the original combined EP contract, while captured gradients and Adam states still check arithmetic and cache invariance.

## Results

The experiment result retains `/output/core-cache-trial/` on success or failure:

- `cold-cache.json`, `restored-cache.json`: immutable identity, cache generations, family outcomes and lifecycle timings.
- `cold.log`, `restored.log`: full independent child logs.
- Each arm's `observation.json`/`.pt`, `stress-ep1-rank0.pt`, final optimizer state, routing IDs, actual training-contract JSONL, fixture and expected TOML.
- `comparison.json`: strict numerical/cache gate, real compiler read/write counts, scoring/optimizer/whole-command timing evidence.

Instrumented timings include extra tensor copies and diagnostics. Compare restore/publication cost alongside cold/restored scoring and optimizer times. This trial can qualify native Core artifact reuse; it cannot establish full-SFT throughput, multi-node propagation, or complete portability of all compiler families. In particular, a rejected TileLang/Inductor artifact remains an explicit remaining issue even if the native Core update and Triton reuse gate pass.

CPU validation covers launcher placement/bounds/TTL, identical fixture preparation, fresh-process commands, full configuration round-trip, observer invariance, missing or mismatched evidence, numerical corruption at each captured boundary, and rejection of cache-hit claims without real read/write evidence. GPU execution remains the next gate.
