---
name: research-status
description: Summarize the user's current research focus and ongoing experiment runs for this project. Use when the user asks for a research status, research summary, progress update, "what am I working on", or "what's currently running".
allowed-tools: Read, Bash(beaker:*)
---

# Research status summary

## Instructions

1. Read `research.md`'s `## Current focus` section — this has the main goal,
   where things stand, and the next step. That's the backbone of the summary.
2. Read the most recently appended section(s) of `experiment.md` (the bottom
   of the file — it's append-only) to find the runs tied to the current
   focus: names, configs, and Beaker links. Prefer runs with no recorded
   outcome yet (no `TBD`/no wandb link/no verdict written up) — those are
   the "ongoing" ones worth surfacing.
3. Skim `research.md` for any other `[ACTIVE]` entries that are *not* part
   of the current-focus thread (e.g. infra/cluster ports, tooling work) and
   call them out separately as a side track — don't blend them into the main
   thread.
4. Do not query Beaker for live status by default (that's a live check, not
   a doc summary) — just report what's recorded in the docs. Only run
   `beaker experiment get <id>` / `beaker experiment logs <id>` if the user
   asks for live status or you're asked to resolve a `TBD`.
5. Output in this shape:
   - `## Current focus` — 2-4 sentences: goal, current state/finding, working
     hypothesis.
   - `## Ongoing runs` — a table (config → Beaker link), plus a one-line flag
     for any known risk (e.g. "not yet rerun with the async_steps fix").
   - `## Side track(s)` — one line each for any other `[ACTIVE]` research.md
     entries outside the main thread, if there are any.

Keep the whole thing tight — this is a status check, not a re-read of the
full log. Link out to `research.md`/`experiment.md` rather than restating
their detail.

## Example output shape

```
## Current focus
Goal: <one line>. State: <2-3 sentences on the latest finding/blocker>.
Hypothesis being tested: <one line>.

## Ongoing runs
| Config | Beaker |
| --- | --- |
| ... | [...](...) |

Known risk: <one line, if any>.

## Side track(s)
- <topic>: <one line status>.
```
