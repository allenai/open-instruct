# Experiments

## Five-scale SGLang midtrain qualification (2026-10-27)

- User requested GPU serving tests at all five hybrid scales, midtrain first,
  and selected Rhea A6000 instead of Saturn A100 for this qualification.
- [Beaker](https://beaker.org/ex/01M5ZA7ZXSFH1T18AQFMX8P9RQ), image
  `01M5ZA7AP531QVWGSAGSC8M5B8`, source MILES `ce1d9564`, release plugin
  scaling-ladders `0fb485ce38733103f7da73973cdf0e74b0fc75db`.
- Cluster ai2/rhea; 1 RTX A6000 48GB, normal priority, minRuntime 0,
  zero retries, timeout 2h; budget ai2/oe-other; workspace ai2/open-instruct-dev.
  Verified Rhea driver 580.173.02 supports the CUDA 13 base runtime.
- BF16, TP1/PP1, Triton full attention, release FLA GDN, no graphs or radix cache,
  no chunked prefill. max_running_requests=4, mem_fraction_static=0.35.
- Final midtrain steps: 275m/450m 0038147, 810m 0023842,
  1.4b/2.7b 0011921, all on the general hybridish-local HF tree.
- Prompt lengths 1/3/31/127/511/2047/3071; mixed batches; 32 greedy cached
  decode steps per request; slot reuse; full disk weight reload followed by repeat.
  HF probabilities on the served continuation must differ by <=0.08 logprob;
  >=95% of generated tokens must be within 0.01 logprob of the HF argmax.
- Every scale runs separately (20-minute cap each), with per-size JSON/logs and
  aggregate failures. Durable output:
  `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/qualification/rhea-ce1d9564`;
  reports also copied into Beaker `/results/hybrid` after each model.
- Status at submission: scheduled, pulling image. No GPU pass claimed.
