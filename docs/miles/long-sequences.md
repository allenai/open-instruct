# Choosing sequence lengths and memory controls

Start from a [structured example](../../configs/miles/examples/README.md), then
change the length and capacity controls together. The SFT MoE checkpoint used in
our readiness exercises advertises 65,536 positions. That is an architecture
limit, not evidence that every training topology or serving concurrency fits.

## Choose a prompt/response budget first

`inference.max_context_length` sets the Core, MILES and SGLang total sequence
limits. `inference.max_response_length` limits newly generated tokens.
`miles.rollout_max_prompt_len` bounds the **rendered and tokenized** prompt,
including its chat template. Reserve room for both; changing the response cap
without considering the input budget can reject otherwise useful data.

Examples of budget arithmetic (not capacity qualifications):

| Context | Long-input, short-answer task: prompt + response | Short-input reasoning task: prompt + response |
|---|---|---|
| 16,384 | 15,872 + 512 | 2,048 + 14,336 |
| 32,768 | 32,256 + 512 | 2,048 + 30,720 |
| 65,536 | 65,024 + 512 | 2,048 + 63,488 |

The prompt limit defaults to context minus response. Prefer explicit budgets for
long-sequence experiments. A dataset's “64K” label may use another tokenizer;
measure with the actual model and template. Imported immutable data is checked
against the prompt limit, not silently truncated. The packer likewise never
splits an individual sample to make it fit.

## Set capacity independently of optimizer batch size

| Control | What to change or watch as length grows |
|---|---|
| `trainer.micro_batch_size` | Keep at one on the currently supported Core path. |
| `trainer.sequence_packing`, `trainer.packing_max_tokens` | Packing helps combine short samples; it does not shrink one long sample. The pack budget must cover `max_context_length`. Start with equal budgets. |
| `trainer.activation_recompute` | Enable for the long-response probes; trades additional compute for activation memory. |
| `miles.log_probs_chunk_size` | The probes use 1,024 to limit temporary log-probability processing. This does **not** eliminate the model's full token-by-vocabulary logits allocation. |
| `inference.sglang_server_concurrency` | Controls client requests. Lower it deliberately for initial long-sequence tests. |
| `inference.sglang_max_running_requests` | Engine admission ceiling. Set alongside client concurrency; neither is an optimizer batch-size control. |
| `inference.sglang_max_total_tokens` | Explicit full-attention KV token budget. Inspect the allocated pool in startup logs; do not assume the requested ceiling was allocated. |
| `inference.sglang_max_mamba_cache_size` | KDA recurrent state capacity, separate from KV tokens. With radix off, cover running requests. Retained prefix caching requires substantially more slots; see the configuration validator and topology guide. |
| `miles.sglang_chunked_prefill_size` | Bounds work in each prefill chunk. The initial sweep uses 2,048; this does not bound the completed request's KV footprint. |
| `inference.sglang_cuda_graph_max_bs_decode` | Match the admission range being tested; large graph captures consume memory. |
| `inference.sglang_mem_fraction_static` | Leaves headroom for transient allocations. The disaggregated probes use 0.6; this is not a colocated recommendation. |

An explicit 131,072-token pool has a theoretical upper bound of eight full 16K,
four full 32K, or two full 64K sequences. Treat this as **budget arithmetic**, not
an achieved throughput target: allocation rounding, reserved capacity, other
requests and transient memory matter. The default pool request can be much larger
because it scales with configured admission and context. Set it explicitly during
qualification instead of retaining a short-sequence concurrency of 64 blindly.

Keep collection and optimizer batch sizes fixed while studying engine admission.
Reducing admission queues requests; reducing samples per prompt changes the GRPO
recipe. EP distributes expert parameters/work, not the token sequence itself.
The current adapter does not offer trainer TP/PP/CP greater than one as a remedy
for a single sequence that does not fit.

## Measured long-input serving settings

On one B300 with the readiness SFT MoE checkpoint, both serving sweeps completed
with no logged OOMs or retractions. The larger pool supported these submitted
request groups, each producing 128 tokens:

| Context | Actual prompt tokens | Concurrent requests | Group completion time |
|---|---|---|---|
| 16,384 | 16,128 | 32 | 21.52 s |
| 32,768 | 32,512 | 16 | 21.91 s |
| 65,536 | 65,280 | 8 | 22.65 s |

Common server settings were KV tokens 524,288, maximum running requests 32,
recurrent slots 64, decode graph cap 32, prefill chunk 2,048, static fraction 0.6
and radix off. Peak sampled device memory was 46.1–46.3 GiB. These are useful
starting points for this model's **long-input, short-output serving**, with client
concurrency reduced as context grows to keep aggregate tokens within the pool.
They are not training or long-decode throughput qualifications. The standalone
probe did not return router traces. Production RL additionally exercises that
path and must keep its own trainer/serving memory budget.

The [full record](measurements/length-guidance-20260913/README.md) distinguishes
submitted requests from periodically logged engine occupancy, includes the
smaller-pool comparison, and retains runtime pins and generations. The reasoning
model hit the 128-token output cap throughout: finite execution passed, retrieval
accuracy was not established.

## What has actually been exercised

See the [length exercise record](measurements/length-guidance-20260913/README.md)
for exact commands, configurations, runtime pins and outcomes. Until a result is
recorded there, a candidate configuration is not qualified.

The prior Core long-context run completed four updates with packing and replay,
reaching **12,742 total tokens**. It does not establish 32K/64K training support.
The new sweep separates synthetic long-input serving from real math RL with long
response budgets. Synthetic inputs establish execution capacity; their repeated
filler and short outputs do not represent long reasoning throughput or task
accuracy. Even long-input, short-answer training must run the model over the prompt; the
serving-only memory trace is not a backward-memory estimate. Router trace return
and replay also add work that the standalone serving sweep does not measure.

The subsequent two-update 16K and 32K-context RL jobs and independent audits
passed. Maximum actual total lengths were **14,466** and **30,850** tokens;
response medians were 14,336 and 30,720, with 11/16 and 9/16 cap hits. Each
retained 24 replay observations with zero mismatches. See the
[final length audit](measurements/length-guidance-20260913/README.md#final-long-response-rl-audit).
This qualifies bounded long-response execution with EP2, packing and recomputation;
it does not establish 64K backward or high-concurrency training capacity.

Earlier olmo-miles/Megatron exercises are useful starting evidence: a 32K run
reached its response cap, while the nominal 64K run actually reached about 41K
response tokens. Both used low admission and microbatch one. Their padding,
DeepEP allocation and trainer memory differ from Core. Do not transfer their
capacity claims to this adapter. One earlier failure was WEKA exhaustion before
training, which illustrates why the failure stage matters.

## Recognize the failure before changing knobs

- **Prompt validation failure:** inspect rendered token counts and the reserved
  response budget. Changing GPU memory will not fix a prompt-budget mismatch.
- **Serving OOM or KV retractions:** inspect cache allocation and the memory trace;
  lower admission, graph size or prefill chunk size according to the allocation
  that failed. Repeated retractions can finish successfully with poor throughput.
- **Trainer forward/backward OOM:** inspect actual sample length, pack size,
  recomputation and logits/scoring temporaries. More inference GPUs do not help.
- **Slow first batch:** separate compilation/startup from warm steps. Record cache
  provenance before treating a cold run as steady throughput.
- **Long generation tails/timeouts:** inspect cap hits and actual completions.
  Async buffering cannot guarantee that a queue stays full when long generations
  are cancelled at publication. Do not infer a scheduling fix from a larger cap.
- **Verifier failure:** code timeouts and judge context limits are independent of
  policy context. A judge must fit prompt, response and rubric together.

Use a small synchronous run to establish length correctness and memory first,
then add the production async and service configuration. A useful result records
finite rewards/gradients, nonzero advantages, publication, replay mismatches,
actual sequence lengths, memory and stage times. Report a lifecycle pass
separately when the sampled rewards do not exercise a policy gradient.
