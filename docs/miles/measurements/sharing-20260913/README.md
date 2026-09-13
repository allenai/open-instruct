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
