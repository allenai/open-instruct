# Readiness continuation, September 13 UTC

> Point-in-time measurement. Recorded defaults, branch names and then-pending work are historical; consult the [current support matrix](../../feature-parity.md) and current examples before launching.

This is qualification evidence, not recommended defaults. Work is on
`robertb/miles-colleague-exercises`; primary remains `robertb/miles-olmo-core`.
Exact submitted specifications and source/image identities are in
[launches.json](launches.json). GPU jobs use urgent Holmes, open-instruct-dev,
one-hour minimum runtime. CPU preparation/audits use Saturn.

## Current work (06:19 UTC)

| Exercise | Evidence/status | Remaining acceptance |
| --- | --- | --- |
| Dense Olmo 3, two colocated FSDP GPUs | Four updates across [first process](https://beaker.org/ex/01M2CD5M6R6WB7M8NYGZ243B8Y) and [fresh resume](https://beaker.org/ex/01M2CEKDXM84SBDCNFEK3FP509), both exit 0; final HF export and fresh reload passed | Qualified lifecycle on this B300 configuration; no bit-exact continuation claim |
| Packed MoE EP2, replay, async/TIS, checkpoint/restart | [First process](https://beaker.org/ex/01M2CDZ17RQWTE2Q5W8P56BQNK) passed updates 1–2; [fresh resume](https://beaker.org/ex/01M2CG9Q3K3GBT8PCX7EBFMY6T) finished updates 3–4; checkpoint/cursor and saved-parameter audits passed | Qualified same-topology lifecycle; no exact-continuation claim |
| Radix + packing/replay + code + named judge | [Four updates passed](https://beaker.org/ex/01M2CDQGG7REP67CCCMPS6BN8B); retained reward audit passed | Policy responses had zero cache hits; combined cache-use qualification remains open |
| Forced cached-prefix replay | [Three-GPU follow-up passed](https://beaker.org/ex/01M2CH5AWXM02NF31RBA55RTXH), two updates; [retained feature audit passed](https://beaker.org/ex/01M2CMEKT2EFY46WV2BRRAJBJ8) | Qualified cached-prefix replay in this sequential-request configuration |
| Natural long contexts + mixed chunks | [Four updates passed](https://beaker.org/ex/01M2CEX1M5BB4Q98GH8P9FHQ4Y); [feature audit passed](https://beaker.org/ex/01M2CGNGJTWPSDENXW9C55KDE1) | Qualified bounded long-token exercise, not a learning benchmark |
| HTTP verifier recovery | [Six cases passed](https://beaker.org/ex/01M2CNHG2KMGY0M93G1RRC3ACD): transient recovery, retry exhaustion, subsequent healthy request, rate limit, sample rejection | Distributed engine/trainer failure recovery remains open |
| Stdio learning signal | [Reviewed fixture submitted](https://beaker.org/ex/01M2CPQ4331ZP52PWN8CCBCESC), three GPUs, two updates | Positive natural rewards and mixed groups; optimizer/replay checks |
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
Image `01M2CD5CH7MBVHZK50AXAPYGFB` contains that fix. Both dense processes finished cleanly and the final export reloaded successfully. Its workflow
result records the intended export path even when export is deferred.

Local checks: 67 pinned-image lifecycle tests passed, including early-stop export
assertions; focused host workflow/config tests and lint passed. GPU resume/export/reload evidence is recorded below.

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
payloads into Git. The full-dump audit checked accounting. A separate
[Saturn rescoring job](https://beaker.org/ex/01M2CFFZDJMXNP3Q3ET6AVW0J5)
then re-executed two retained responses per deterministic domain per run, selected
at the reward extremes: all 16 matched exactly. Reports: [cold](rescore-cold.json),
[warm](rescore-warm.json). No new stochastic judge requests were made.

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

## Completed feature evidence

The long run consumed 64 training responses: eight had prompts over 4,096 tokens,
63 had responses over 4,096 tokens, and 61 exceeded 8,192 total tokens. Maximum
individual sequence length was 12,742. Eight packing records and 78 replay
observations passed sample/token alignment and routed-layer recomputation checks.
[Full feature report](long-features.json). Only one distinct long training problem
was used; short math controls supplied one mixed-reward group. Do not infer
long-context learning quality from this exercise.

Dense [lifecycle audit](dense-lifecycle.json) verified all four committed native
checkpoints/cursor hashes, optimizer descriptors and rank state, steps 1–4 on both
ranks, and the scoring check again at resume. Publication versions were
`0, 1, 2, 2, 2, 3, 4`: the extra restored-version publications include the startup
snapshot/reset/republish equality check. The scoring checks were bit-identical.
[Bounded CPU probes](dense-drift.json) found movement in five of eight sampled
parameter tensors between saved updates one and four. This checks actual parameter
movement without adding periodic full-weight audits to training.

The [fresh SGLang reload](https://beaker.org/ex/01M2CFXEF3RY9ZP6YS17AAWD9C)
loaded the committed HF export in about 136 seconds and generated finite
log-probabilities on all four held-out prompts. Three of four greedy prefixes
matched the retained final evaluation exactly; one differed. [Report](dense-reload.json).
This is a load/generation check, not bit-exact inference or restart equivalence.

The four-update radix/judge run passed its [reward audit](radix-rewards.json),
including mixed rewards from math, IF, function code, and both named judge rubrics.
Stdio scores remained zero. The stricter feature audit failed because retained
policy samples reported **zero cached tokens**. Judge-server cache hits do not
establish policy-cache reuse. The forced-prefix follow-up uses one policy engine,
one request at a time and 512-token responses to make reuse observable; it is a
smaller unjudged replay probe, not another throughput comparison.

Two early dense lifecycle audit jobs failed because the new auditor assumed a
scalar LR and omitted the startup reset/republish round trip. Those were auditor
schema errors, not training failures; corrected audit
[01M2CGBVYNEY3XWWWXHE2RQNVM](https://beaker.org/ex/01M2CGBVYNEY3XWWWXHE2RQNVM)
passed. Keep this distinction when counting failed GPU exercises.

The MoE async restart also passed its [lifecycle audit](moe-lifecycle.json) and
[bounded parameter drift check](moe-drift.json). Its [352 replay observations](moe-resume-replay.json)
covered scoring/training/recomputation across both processes with zero route or
boundary mismatches. All three compiler workers restored matching caches.
This is same-topology durable recovery, not automatic recovery from a killed
engine/trainer or an exact comparison with an uninterrupted stochastic run.

The forced-prefix probe completed both updates. Its first collection reported a
0.5314 prompt cache hit rate (148 cached tokens per response). The retained-sample
feature audit passed, including actual policy cached tokens, 12 replay observations
and four packing records, with zero replay route/boundary mismatches.
[Report](cache-replay-features.json). Publication took 0.46 and 0.45 seconds.
The first training call took 362 seconds with cold compilation; the second took
2.40 seconds. These tiny sequential-request measurements are not a pool-sizing benchmark.

Local follow-up: 83 focused preparation, judging, RunSpec and readiness tests
passed; all 183 Open Instruct Python files passed formatting and Ruff checks.
Repository type checking also passed with the actual Core source selected explicitly;
the unconditional missing-source-directory requirement was removed from global
configuration and the developer override documented. All four researcher example configs rendered with consistent bridge/host-network
and NCCL interface settings.

## Still open

Establish positive stdio learning coverage; exercise controlled service-failure recovery and the
full judged topology; finish corrected engine-pool sizing and the hero numerical
gate. No H100 or new hero-training claim follows from these B300 exercises.


## Follow-ups submitted at 06:04 UTC

[Stdio-only exercise](https://beaker.org/ex/01M2CNVBQERPXAMW332DNYCB8E):
three GPUs (EP2 trainers plus one engine), two collections of four prompts times
four responses, packing/replay and radix enabled. This selects the shortest
natural stdio prompts from the original immutable manifest, preserving targets,
chat template and train/eval identities. Eight training and two held-out problems;
6,144 response tokens and 8,192 total context. Shortness is not a difficulty claim.
[Preparation](stdio-preparation.json) and all six verifier canaries passed.
Acceptance requires positive natural rewards and a mixed-reward group, alongside
optimizer/replay evidence; a clean process exit alone does not establish it.

[Service recovery probe](https://beaker.org/ex/01M2CNHG2KMGY0M93G1RRC3ACD)
runs on Saturn. A private loopback proxy injects transient 503/502/504 errors,
exhausts the production HTTP retry budget, then permits healthy requests again.
Only successful proxy requests reach the real code service. Also checks 429 with
Retry-After and the established HTTP-500 sample-rejection semantics. This covers
HTTP verifier recovery/exhaustion, not distributed engine replacement or recovery
inside a live training loop. All six cases passed with the production retry settings: transient recovery
took about 6.3 seconds; nine failed attempts raised after 246 seconds. The same
session then completed a healthy request. [Report](service-recovery.json).


The first stdio GPU submission was stopped while queued, before startup. Manual
inspection of the prepared prompts found that several source entries retained only
input/output examples, without a problem statement. The corrected `10-stdio.toml` uses a
fresh immutable fixture and excludes those entries before sorting. The preparer
has a regression test for example-only prompts. Existing source snapshots remain
unchanged; neither this filtering nor shortness guarantees an easy problem.


Preparation revisions r2 and r3 also failed manual prompt review (they were never
submitted to GPUs): remaining examples included resource-limit headers without a
statement, generic coding instructions, and image-only statements. Revision r4
uses a conservative statement/action-word and input/output screen, followed by
manual review. This screen is confined to the qualification fixture; it is not a
production dataset filter or a proof of data quality. Only the final exercise TOML
is kept; submitted run specifications and source history preserve earlier attempts.


Final stdio revision r4 passed [CPU preparation](https://beaker.org/ex/01M2CPHWQW9CJ9ZHXBXS55WKP9)
and manual prompt review. [Retained preparation report](stdio-r4-preparation.json).
The [GPU run](https://beaker.org/ex/01M2CPQ4331ZP52PWN8CCBCESC) is submitted using
image C, with exact launch receipt in `launches.json`. Training examples include
addition of integer pairs, a digital root and sequence modes, alongside harder
string problems. It is still a short learning-path check, not an easy-task score
benchmark. All 68 focused host readiness/config/retry tests passed; changed scripts
passed Ruff/type checks and generated documentation remained current.


## Combined asynchronous judged exercise

[Submitted run](https://beaker.org/ex/01M2CRTVVAAAM85D0E8RFQ7NJ3), config
`11-combined-async-judged.toml`: one node with two EP2 trainers, three TP1 policy
engines and one managed Qwen3-32B judge. Four collections of 8 prompts x 8
responses, global batch 64, asynchronous sample submission with TIS and maximum
policy lag two, packing/replay, radix extra-buffer caching, cache-aware routing,
and native saves after updates two and four. Startup-only full-weight audits.

The existing six-domain fixture and pinned runtime image C are reused. The KDA
cache holds 384 states per engine to satisfy the >5x admission guard for 64
running requests when radix is enabled. CPU planning verified six physical GPUs,
one-node host networking for the judge bootstrap, and all resolved controls;
64 config/cache tests and lint passed before submission. This combines previously
separate checks; it does not establish EP8, restart, failure recovery, or actual
prefix-cache use merely by enabling the flags. Acceptance requires retained
per-domain reward/service outcomes, lag/publication/packing/replay checks, native
checkpoint integrity, and observed cache hits for the cache-use claim. The full
EP8 + seven-engine + judge case remains separately pending whole-node capacity.


## Overnight outcomes, checked at 16:31 UTC

Both GPU runs exited zero. Stdio completed two updates in 24.8 minutes including
startup and evaluation; the combined six-GPU run completed four updates in 34.9
minutes. [Stdio completion/replay/timings](stdio-completion.json) and
[combined completion/replay/timings](combined-completion.json) retain the workflow,
rank-contract hashes, per-stage timings and rollout metrics.

Stdio logged mean reward 0.125 in each 16-response collection, with one mixed-reward
group of four in each collection. The execution service rejected no samples.
Both optimizer steps ran with finite nonzero total gradient norms. Forty-two
replay observations and four packing records passed with zero route/boundary
mismatches. Twenty-three of 32 responses reached the 6,144-token cap. This
establishes positive rewards and mixed groups on natural stdio tasks, not a
learning-quality comparison or an isolated policy-gradient norm measurement.

The combined run passed 196 replay observations and eight packing records across
four optimizer steps. Actual cache hits occurred in the first three collections;
maximum consumed policy lag was two, within the configured bound. Both scheduled
saves completed and committed manifests report completed steps two and four.
After update one, generation waits were 4.84, 4.10 and 1.82 seconds; publication
was 0.70–1.29 seconds across all four updates. Training took 590.54 seconds on the
first call and 25.29–70.64 seconds on later calls. Saves took 109.04 and 120.01
seconds. These are short combined-feature measurements, not a calibrated engine
pool comparison or a fresh-process restore of these checkpoints.

Independent [retained-sample reward audits](https://beaker.org/ex/01M2DSEHTMJHBE7AZZEMA24H4A)
and a [combined cached-prefix feature audit](https://beaker.org/ex/01M2DSGSSZ5GWJFZYDQ31DR3ZX)
were submitted on Saturn after downloading the logs. At this timestamp they are
pulling the runtime image; no audit pass is claimed yet. Full EP8/judged topology
still awaits two whole Holmes nodes (current free GPUs are fragmented 7 + 1).


At 16:32 UTC both independent audit jobs passed, exit zero. The
[stdio reward audit](stdio-reward-audit.json) confirms four fully correct and 28
zero-score training responses, two mixed-reward groups and 32 successful code
service outcomes. The [combined reward audit](combined-reward-audit.json) confirms
mixed groups in math, IF, function code and both named judge rubrics; its original
stdio fixture still scored zero. The [combined feature audit](combined-feature-audit.json)
confirms 1,792 cached prompt tokens out of 50,480, 196 replay observations and
eight packing records. Reward identity/accounting, prompt/split hashes, group and
policy-version checks passed. These audits reparse retained grades; they do not
independently re-execute every code test or re-query stochastic judges.
