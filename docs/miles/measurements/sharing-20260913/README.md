# Sharing consolidation and documentation review, September 13

This record covers the candidate preparation. The current [candidate page](../../sharing-candidate.md)
records image qualification and promotion; the [support matrix](../../feature-parity.md)
is the current boundary for colleagues.

## Documentation review

The [inventory](documentation-inventory.json) accounts for 115 Markdown entry
points, current guides, generated references, compatibility redirects and historical
records. Current model/topology claims were checked against the retained qualification
reports and configuration/driver implementations. Generated reference checks cover
every structured field and native parser action; parser acceptance is explicitly
separate from backend support.

Corrections:

- One preferred MILES entry point in README, documentation home, installation,
  navigation and agent instructions; deprecated paths remain historical references.
- Replaced mixed historical/current Core, dense, cache and parity pages with current
  guides. Original chronology is archived under implementation-history with explicit
  supersession links, not left as a competing procedure.
- Corrected dense resume/export/reload, full-model replay/async restart, combined
  judged workload, engine-drain follow-ups and final long-response status.
- Removed stale cache source paths, old shutdown-cost claims presented as current,
  pending radix reports and duplicated config guidance. Kept long-prefix cache benefit
  conditional on actual reuse and measured end-to-end time.
- Kept every historical run/config/result immutable in meaning. Historical plans and
  measurements identify their scope; then-pending jobs do not define live status.
- Moved runtime/validation.json's early gdn2 prototype evidence into the measurements
  directory. It is not a qualification certificate for today's runtime.

The host suite passed 324 tests with one skip; all 388 checked Python files passed
Ruff formatting/lint and type checks passed against the pinned Core source.
MkDocs built successfully. Four pre-existing missing-target warnings outside MILES
remain (documentation home data script, Tulu human_eval and two legacy screenshots);
there are no MILES missing-file or missing-anchor warnings after this review.

## Consolidation

The current [itemization](../../sharing-candidate.md#consolidation-decisions-september-13)
distinguishes merged, already incorporated, superseded and held work. The qualified
engine-drain mode remains opt-in. Experimental mixed-policy refresh and its inherited
throughput templates stay on their feature branches while their training gate is
being repaired/qualified. No default baseline inherits that mode.


## Immutable image and full-SFT qualification

Beaker image `01M2E5QR5C60WF7H0TDEF4CD3S` contains application revision
`2de5c5ba421d653a95981fa1bb35dcb64231aeb3`. The [provenance manifest](image-provenance.json)
records the binary foundation, source pins, image digest and integration file hashes.
Subsequent consolidation commits add documentation, evidence and a host-only CPU
audit launcher; they do not change the image's runtime Python, lock or run configuration.

The [full-SFT run](https://beaker.org/ex/01M2E5VRB83X9TFKFAVYMNMJXN) and
[independent Saturn audit](https://beaker.org/ex/01M2E7CGNMDWPQCMMG1A7VS9BP)
both exited zero. The submitted [configuration](../../../../configs/miles/qualification/sharing-sft-20260913.toml)
uses EP2 plus one serving engine, async TIS with lag one, 8 prompts × 8 responses,
packing, dynamic rows, recomputation, router replay, and initial/final GSM8K evaluation.

The [sample audit](sft-sample-audit.json) checked 128 training samples and 16 evaluation
responses, with seven mixed-reward training groups, four packing observations and
70 replay observations across ranks. Training rewards were 111/128 correct; the two
evaluations combined were 13/16 correct. This tiny screen is not a learning comparison.
It checks retained rewards/accounting; it does not independently rerun the math verifier.

Both optimizer steps completed with finite nonzero total gradient norms (0.0864 and
0.2897). The first standalone scoring check was bit-identical to the training forward
over 85,463 active tokens; the second update skipped the redundant pass. Both batches
were generated with version zero, so the second update actually exercised allowed
lag one. TIS clipped no tokens. This is learning-path evidence (mixed groups and
nonzero gradients), not evidence of improved model quality. Full-SFT saving/export
was disabled here; the separate synthetic gate exercises that lifecycle.

| Driver interval | First | Second |
|---|---:|---:|
| Generation wait | 64.90 s | 1.41 s |
| Training call, including scoring | 566.03 s | 96.52 s |
| Inner optimizer batch (nested in training) | 350.23 s | 96.00 s |
| Weight publication after update | 2.493 s | 2.585 s |

Serving startup took 351.25 s, trainer startup 127.38 s, initial publication 11.66 s,
and initial/final evaluation 24.87/50.36 s. These were cold caches. The GPU gate ran
on other GPUs of the same physical node, so concurrent compilation could contend
for CPU; these are diagnostic timings, **not an isolated throughput benchmark**.
Rank-zero peak allocated memory was about 166.2 GiB.

[Compiler-cache finalization](sft-compiler-cache.json) took 11.224 s total, published
all three workers, and did not time out. About 61 MB of serving artifacts compressed
to 12.6 MB; each trainer's roughly 238 MB compressed to about 51 MB. Local staging
and two concurrent publishers were effective in this actual WEKA run.

Retained structured evidence includes [training metrics](sft-training-metrics.json),
[submitted identity](sft-submitted-run.json), [workflow completion](sft-workflow.json),
and the driver, publication and rank contract JSONL files alongside this note.
The sample report includes representative generations and hashes of all rollout dumps.


## Combined image gate — passed

The [gated B300 experiment](https://beaker.org/ex/01M2E5VPRJWQKD0H4J186M828M)
exited zero after every stage, with the final [completion marker](gate-complete.json):

- Runtime suite: **847 passed, 10 skipped, zero failures/errors**, retained in
  [JUnit](gate-runtime-tests.xml). One explicitly excluded optional cross-backend
  comparison module requires an external olmo-miles development checkout.
- Synthetic dense Olmo 3: two updates, native save and HF export, then a fresh-process
  restore and third update. Scoring decisions were checked/skipped/checked across
  that process boundary.
- Two-rank FSDP diagnostics: three tests passed on each rank.
- EP2 packing with recomputation: both rank reports passed. Packed/unpacked score
  checks and sequence-isolation checks were exact; gradients and optimizer states
  passed the existing numerical tolerances, not a bit-exact gradient claim.
- Live synthetic MoE: EP2 with replay enabled, two updates then a fresh-process third
  update. The [audit](gate-moe-audit.json) verified sample versions 0/1/2, continued
  prompt cursor and parameter changes. All task rewards were zero, so these changes
  were driven by auxiliary objectives; this synthetic case proves lifecycle, not
  policy learning. Its audit does not retain route tensors; the full-SFT audit above
  supplies actual route-replay observations.

The final documentation regression suite passed nine tests and MkDocs rebuilt with
no MILES link warnings. The primary Open Instruct branch is promoted to this
consolidation; MILES includes its companion engine-drain hook. Core and serving
remain at the already-qualified pins in the provenance manifest.
