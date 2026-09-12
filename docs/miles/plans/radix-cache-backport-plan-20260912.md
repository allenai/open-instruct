# Backporting the olmo-miles KDA radix cache policy

> Historical proposal. For current operating instructions, start at the [MILES guide](../index.md).

Reviewed 2026-09-12 against olmo-miles `07887b7` (six commits: `6c23c18`, `0be3552`,
`ef7d9db`, `10c1b23`, `ba5da13`, `ca03a25`), our olmo-sglang branch
`robertb/miles-serving` (`02ccb5dcf`), our MILES branch `robertb/olmo-core-backend`
(`d29c04a94`), and the two-node run
[record](../measurements/two-node-async-gsm8k-20260912.md).

## What lives where

| Layer | What olmo-miles relies on | Our stack today |
| --- | --- | --- |
| olmo-sglang | `RadixLinearAttention` in the KDA layer, `uses_mamba_radix_cache=True` in the model registration, `validation/radix.py` (commits `fb41e05`, `1a5bd6f`) | **Already present** on `robertb/miles-serving`; both commits are ancestors of our base `81a312ee`. Nothing to port. |
| MILES | `--sglang-disable-radix-cache`, `--sglang-mamba-radix-cache-strategy`, `--sglang-max-mamba-cache-size`, `--sglang-router-policy` (`cache_aware` among the choices), `--router-cache-threshold`, `--router-balance-{abs,rel}-threshold`, `--sglang-enable-mixed-chunk` | **All exposed** by the pinned parser (see `options.json`). Nothing to port. |
| olmo-miles (`config.py`, `launcher.py`, `preflight.py`, `compat/sglang_compat.py`) | Defaults, capacity validation, a worker-side registration guard, launch flags, tests, and the measurements that chose the defaults | **This is the backport.** Our structured run file only knows `inference.radix_cache`; everything else is reachable only through `[miles]` pass-through, and our examples set `radix_cache = false`, strategy `auto`, 128 state slots. |

So the port goes into **open-instruct** (`open_instruct/miles/run_spec.py`,
`validation.py`, the profiles and examples, docs), not into the MILES fork and not
into olmo-sglang.

## What olmo-miles decided, and why

- **Radix cache on, strategy `extra_buffer`.** The KDA blocks carry a recurrent state
  per request, so prefix reuse needs that state cached alongside the full-attention
  keys and values. `extra_buffer` is the branch-capable strategy that was validated;
  the config rejects any other strategy, requires the `triton` attention backend,
  `page_size = 1`, and overlap scheduling on.
- **State-slot sizing.** Each running request can hold up to 5 KDA state slots under
  `extra_buffer` (overlap allowance at chunk boundaries and branches), so the pool must
  exceed `5 × max_running_requests`; the remainder holds retained prefix states.
  olmo-miles' default is 332 slots for 64 running requests, i.e. 320 active plus 12
  retained. With the cache off the rule is simply `slots ≥ running requests`, which is
  our current 128 for 64.
- **Cache-aware routing.** With eight engines, a request only benefits from a cached
  prefix if it lands on the engine that holds it. The sglang-router `cache_aware`
  policy with thresholds `0.8 / 4 / 1.5` (cache-hit ratio, absolute and relative
  load imbalance) sends siblings of a prompt group to the same engine unless that
  engine is overloaded.
- **Mixed chunk: screened and not adopted.** The
  [knob sweep](/home/robert/proj/olmo-miles/docs/measurements/trainer-inference-knob-sweep.md)
  measured `--enable-mixed-chunk` at −13% / −7% / +9% / −11% across its four cells and
  left it off. The same sweep found radix on versus off within noise on speed
  (+1% / −2% / +4% / −2%): "`extra_buffer_lazy` and radix-off buy state slots, not
  speed." Radix is the olmo-miles default for generality and for the long-prompt
  workloads it was built for, not because it sped up short-prompt math.

On our two-node GSM8K shape (94-token prompts, 1,100–2,100-token responses), the
cacheable share is about 7 of every 8 prompt prefixes, which is well under 10% of
generated tokens, so the expected speed change is inside noise. The value is in
matching the validated olmo-miles serving path and in being ready for long system
prompts, multi-turn and tool loops, where prefix reuse is most of the prefill.

## Plan

1. **Serving check, no code.** Run olmo-sglang's own `validation/radix.py` in the
   pinned image against the SFT-65536 checkpoint: prefix-state reuse on versus off must
   give identical log-probabilities for the same continuations. One GPU, about 15
   minutes. This is the proof that the serving side we already ship is sound for this
   model in this image; olmo-miles ran it on their image, not ours.
2. **Structured keys.** Add to `[inference]` in `run_spec.py`, mapped to the native
   names: `mamba_radix_cache_strategy`, `router_policy`, `router_cache_threshold`,
   `router_balance_abs_threshold`, `router_balance_rel_threshold`, `enable_mixed_chunk`
   (`radix_cache` already exists). Reject duplicates against `[miles]` as the rest of
   the schema does.
3. **Validation, ported from olmo-miles `validate_inference_capacity`.** When the
   cache is on: strategy must be `extra_buffer`; attention backend `triton`; page size 1;
   overlap scheduling on; `max_mamba_cache_size > 5 × max_running_requests`. When off:
   `max_mamba_cache_size ≥ max_running_requests`. Router thresholds in range. Port the
   four olmo-miles tests (`test_radix_cache_capacity_reserves_active_and_prefix_slots`,
   `test_radix_cache_rejects_unvalidated_runtime_controls`,
   `test_train_args_can_disable_radix_for_matched_control`,
   `test_round_robin_router_is_an_explicit_threshold_free_control`) into
   `tests/miles/test_options.py` or a new `test_inference_capacity.py`.
4. **Registration guard.** olmo-miles asserts in every Ray worker that the olmo-sglang
   registration exposes `uses_mamba_radix_cache` before `ServerArgs` is built. We rely
   on `SGLANG_EXTERNAL_MODEL_PACKAGE` alone. Add the same assertion to our engine
   startup path when `radix_cache` is on; it is ten lines and turns a silent
   misconfiguration into an immediate error.
5. **Defaults.** Switch the four examples and the two full-SFT profiles to
   `radix_cache = true`, `mamba_radix_cache_strategy = "extra_buffer"`,
   `sglang_max_mamba_cache_size = 332`, `router_policy = "cache_aware"` with
   `0.8 / 4 / 1.5`. Leave `enable_mixed_chunk = false` but present in the examples so it
   is one edit to try. Record the slot memory: read the engine's KDA pool allocation
   from the log of step 6 and note bytes per slot in the README table.
6. **A/B on the two-node config.** Twenty updates each, same seed and prompts: radix
   off (today's r2 settings) versus on with the step-5 defaults, and a third arm with
   mixed chunk on. Compare warm cadence and generation wait from `driver_timing.jsonl`,
   `prefix_cache_hit_rate` and `avg_cached_tokens_per_sample` from the perf records,
   the trainer-versus-behavior log-probability gap and TIS clip fraction (prefix reuse
   changes the arithmetic of the shared prefix, so the gap is the correctness signal),
   and the scoring checks. Predicted: hit rate near 0.06, cadence within noise, gap
   unchanged. Adopt if the gap and checks hold; the speed result is informational.
7. **Docs and changelog.** Run-controls table, profile README admission table, knob
   inventory rows, and a measurement record for step 6.

Steps 2–5 are about a day of work with tests; step 1 and step 6 are two GPU jobs. The
`ba5da13` commit (Megatron checkpoint loader) and the `ca03a25` recipe flag fix are
Megatron-side and do not apply.
