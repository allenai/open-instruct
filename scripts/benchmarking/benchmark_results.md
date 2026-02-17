# Hybrid vs Non-Hybrid Benchmark Results

## Setup

- **GPUs**: 8x (H100/H200 on Beaker)
- **vLLM engines**: 4 (TP=2 each)
- **Prefix caching**: enabled
- **GPU memory utilization**: 0.9
- **enforce_eager**: True
- **Dataset**: `hamishivi/hamishivi_rlvr_orz_math_57k_collected_all_filtered_hamishivi_qwen2_5_openthoughts2`
- **Num unique prompts**: 16, **Samples per prompt**: 4 (64 total generations per batch)
- **Max prompt token length**: 2048
- **Temperature**: 1.0, **Top-p**: 0.9, **Seed**: 42

## Generation Throughput Benchmarks

### OLMo 3.1 7B Hybrid (`OLMo3.1-7B-6T-30h-long-context-drope/step23842-hf`)

| Gen Length | Time per Batch (s) | MFU (%) | MBU (%) | Tokens/s | Beaker URL |
|-----------|-------------------|---------|---------|----------|------------|
| 1024 | 48.15 | 0.73 | 7.00 | 1361.13 | [01KH9AX8](https://beaker.allen.ai/ex/01KH9AX80QH3CWD2JPFPFA6E7A) |
| 4096 | 183.98 | 0.79 | 9.00 | 1424.87 | [01KH9BR6](https://beaker.allen.ai/ex/01KH9BR66CSDRP76GNYSAKY63H) |
| 8192 | 384.23 | 0.81 | 10.77 | 1364.51 | [01KH9CV0](https://beaker.allen.ai/ex/01KH9CV0PBG4F793C5CNX3MN1Z) |
| 16000 | 855.17 | 0.80 | 13.04 | 1197.42 | [01KH9EG0](https://beaker.allen.ai/ex/01KH9EG02CVBGMAB3M2RDYXJEE) |
| 32000 | 3133.38 | 0.54 | 11.18 | 653.61 | [01KH9GN2](https://beaker.allen.ai/ex/01KH9GN2H5VC9HG6H5KYDYFBYN) |

### Nemotron-H 8B (`nvidia/Nemotron-H-8B-Base-8K`)

| Gen Length | Time per Batch (s) | MFU (%) | MBU (%) | Tokens/s | Beaker URL |
|-----------|-------------------|---------|---------|----------|------------|
| 1024 | 39.95 | 2.44 | 22.34 | 1640.31 | [01KH9QFS](https://beaker.allen.ai/ex/01KH9QFSX04TFM2EYQ7EJ2HPHG) |
| 4096 | 162.20 | 2.45 | 22.96 | 1616.21 | [01KH9RVE](https://beaker.allen.ai/ex/01KH9RVEK78P8BVZ9FA4J9B97K) |
| 8192 | 319.97 | 2.56 | 24.12 | 1638.53 | [01KH9STX](https://beaker.allen.ai/ex/01KH9STXZAG6ZVFVYNFARDJHTC) |
| 16000 | 674.21 | 2.52 | 23.85 | 1518.81 | [01KHCJ22](https://beaker.allen.ai/ex/01KHCJ223D28C0TKVWWSFXTMBQ) |
| 32000 | 1654.42 | 2.56 | 24.34 | 1237.90 | [01KHCM0H](https://beaker.allen.ai/ex/01KHCM0H11V070P1CDY02ETJ83) |

### Falcon-H1 7B (`tiiuae/Falcon-H1-7B-Base`)

| Gen Length | Time per Batch (s) | MFU (%) | MBU (%) | Tokens/s | Beaker URL |
|-----------|-------------------|---------|---------|----------|------------|
| 1024 | 80.40 | 0.40 | 3.52 | 815.08 | [01KH9ZGF](https://beaker.allen.ai/ex/01KH9ZGFNJN3XAJJBWYWEXT8NN) |
| 4096 | 349.67 | 0.37 | 3.32 | 749.69 | [01KHAG35](https://beaker.allen.ai/ex/01KHAG35VRZGZJ8QHHAMXRXP4G) |
| 8192 | 829.52 | 0.33 | 2.90 | 632.04 | [01KHBTTY](https://beaker.allen.ai/ex/01KHBTTY5Y8DKZN93184RZ26DZ) |
| 16000 | 1863.11 | 0.31 | 2.68 | 549.62 | [01KHBWSN](https://beaker.allen.ai/ex/01KHBWSNK4AR1S1SR2P6CY9947) |
| 32000 | 4713.07 | 0.29 | 2.41 | 434.54 | [01KHC0KC](https://beaker.allen.ai/ex/01KHC0KCZXRZSQAZCM7GYAZ5KW) |

### falcon-mamba 7B (`tiiuae/falcon-mamba-7b`)

Failed: vLLM prefix caching is incompatible with pure Mamba models (`AssertionError: UnitaryKVCacheCoordinator assumes hash_block_size == block_size`). Beaker: [01KHCAF9](https://beaker.allen.ai/ex/01KHCAF9YBAVYFB1X0TCC94DGK)

## KV Cache Concurrency Analysis

Max number of concurrent sequences that fit in KV cache at each sequence length (8x GPU, TP=2, 0.9 GPU memory utilization).

### Nemotron-H 8B (`nvidia/Nemotron-H-8B-Base-8K`)

- Available KV cache memory: 62.50 GB
- KV cache blocks: 14449
- KV cache groups: 7 (6x MambaSpec @ 4 layers each, 1x FullAttentionSpec @ 4 layers)

| Sequence Length | Max Concurrency |
|----------------|----------------|
| 1024 | 1806.12x |
| 4096 | 1032.07x |
| 8192 | 656.77x |
| 16384 | 380.24x |
| 32768 | 209.41x |

Beaker: [01KHD60X](https://beaker.allen.ai/ex/01KHD60X5EJVKZWGNCF7F2WMB5)

### Falcon-H1 7B (`tiiuae/Falcon-H1-7B-Base`)

- Available KV cache memory: 62.89 GB
- KV cache blocks: 1780
- KV cache groups: 2 (1x FullAttentionSpec @ 44 layers, 1x MambaSpec @ 44 layers)

| Sequence Length | Max Concurrency |
|----------------|----------------|
| 1024 | 890.00x |
| 4096 | 445.00x |
| 8192 | 254.29x |
| 16384 | 148.33x |
| 32768 | 80.91x |

Beaker: [01KHD61B](https://beaker.allen.ai/ex/01KHD61BZ9AAZV3H4QT5GVX57E)

### OLMo 3.1 7B Hybrid

Failed: `olmo3_2_hybrid` model type not recognized by the Transformers version in the Beaker image. Needs updated transformers with `trust_remote_code` support for this architecture. Beaker: [01KHD60D](https://beaker.allen.ai/ex/01KHD60DYNRWS7KEA0FPM2RH9S)
