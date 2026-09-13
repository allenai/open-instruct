# Readiness continuation, September 13 UTC

This is qualification evidence, not recommended defaults. Work is on
`robertb/miles-colleague-exercises`; primary remains `robertb/miles-olmo-core`.
Exact submitted specifications and source/image identities are in
[launches.json](launches.json). GPU jobs use urgent Holmes, open-instruct-dev,
one-hour minimum runtime. CPU preparation/audits use Saturn.

## Current work

| Exercise | Evidence/status | Remaining acceptance |
| --- | --- | --- |
| Dense Olmo 3, two colocated FSDP GPUs | First process [passed](https://beaker.org/ex/01M2CD5M6R6WB7M8NYGZ243B8Y): updates 1–2, two committed saves, stopped intentionally, publications 0/1/2 | Fresh allocation resumes updates 3–4, final export, fresh serving reload |
| Dense resume | [Submitted](https://beaker.org/ex/01M2CEKDXM84SBDCNFEK3FP509), identical TOML/root/horizon | Restore and export audit |
| Packed MoE EP2, replay, async/TIS, checkpoint/restart | [Corrected attempt](https://beaker.org/ex/01M2CDZ17RQWTE2Q5W8P56BQNK) starting | Two updates, intentional stop, same-config fresh-process resume for two more |
| Radix + packing/replay + code + named judge | [Starting](https://beaker.org/ex/01M2CDQGG7REP67CCCMPS6BN8B), five GPUs | Four updates, actual cache hits and reward observations, policy/replay checks |
| Natural long contexts + mixed chunks | Fixture prepared; four-GPU run submitted (ID in launches.json) | Actual long training tokens, chunked prefill, packing/replay checks and final eval |
| EP8 + seven engines + judge | Config 06 staged | Two full nodes available; distributed startup/rewards/ownership |
| Corrected 8 versus 16 engine sizing | Configs staged; previous 16-engine attempt failed rendezvous before training | Matching runs with producer budget raised and periodic weight audits off |

Do not conflate the named-judge large-topology test with the unjudged sizing
comparison. Do not retry multi-node jobs blindly while no whole nodes are free.
Engine-level rolling drain is a separate implementation handoff; cancellation
still limits useful asynchronous throughput in these images.

## Fix found and exercised

Intentional debug stopping previously triggered final HF export, occupying the
exclusive export directory before a resumed process reached the run horizon.
Commit `f958d4b6d452` defers final export until the configured horizon completes.
Image `01M2CD5CH7MBVHZK50AXAPYGFB` contains that fix. The first dense process
stopped correctly; resumed final export remains to be checked. Its workflow
result records the intended export path even when export is deferred.

Local checks: 67 pinned-image lifecycle tests passed, including early-stop export
assertions; focused host workflow/config tests and lint passed. This does not
replace the GPU resume/export/reload exercise.

The original MoE resume attempt
[failed before training](https://beaker.org/ex/01M2CCWWPGXG3W2C6SEYNZR704): its
single-node bridge container inherited `NCCL_SOCKET_IFNAME=ib` from a multi-node
profile. NCCL found no interface. Retry `09-moe-async-resume-r2.toml` removes the
physical-interface overrides. Multi-node and managed-judge profiles use host
networking; their interface settings are a separate case.

## Retained reward audit

[Saturn audit](https://beaker.org/ex/01M2CCWVM8J963SFSRNPDWVAQR) passed on 256
training plus 12 evaluation samples per cold/warm run. It checked immutable
prompts, labels, verifier targets, prompt-token hashes, group multiplicity,
single-policy groups, policy age, component weights and reward accounting.
Named-judge replies were reparsed and their named model binding checked.

| Mixed-reward training groups | Cold | Warm |
| --- | --- | --- |
| Math | 1 | 0 |
| Function code | 1 | 0 |
| Stdio code | 0 | 0 |
| Instruction following | 6 | 6 |
| General judge | 7 | 8 |
| Reference judge | 8 | 8 |

All 32 stdio training requests in each run executed with service status `ok` but
scored zero. Known-answer service canaries passed earlier; this fixture has not
demonstrated positive stdio training rewards or a mixed-reward stdio group.
Warm function-code rewards included four positives in one all-correct group,
which also provides no within-group policy advantage.

Reports: [cold](reward-audit-0.json), [warm](reward-audit-1.json). They retain dump
paths/hashes so generations can be recovered without checking large response
payloads into Git. This was an accounting audit, **not independent re-execution**
of deterministic verifiers, nor an optimizer-equivalence proof.

## Long-context fixture

[Inventory](prompt-inventory.json) found all prepared prompts at or below 2,048
tokens. Unfiltered pinned Dolci also lacked eligible 4K–8K prompts. The pinned
Olmo 3 code mixture yielded two additional distinct natural long prompts after
deduplication. [Preparation](long-context-preparation.json) passed on
[Saturn](https://beaker.org/ex/01M2CER6M9HASYW2HZZZZ4674J): 4,550 training prompt
tokens and 4,556 held-out prompt tokens, with short math controls. No synthetic
padding. Limits are 8,192 prompt / 8,192 response / 16,384 context tokens.

This is a small boundary exercise with two distinct long problems, not a
long-context learning benchmark. Configured limits do not prove actual coverage;
inspect retained generation lengths and trainer packing/replay records.

Preparation attempts also caught a source-schema mistake: a plain `code_stdio`
target is a JSON list of tests, whereas Dolci wraps aligned verifier targets.
The preparer now preserves the former as one target; regression tests cover both
representations. Earlier CPU attempts failed closed and created no fixture.

## Still open

Fresh-process restore/export/reload qualification, independent verifier rescoring,
positive stdio learning coverage, controlled service-failure recovery, the full
judged topology, corrected engine-pool sizing, and the hero numerical gate remain
open. No H100 or new hero-training claim follows from these B300 exercises.
