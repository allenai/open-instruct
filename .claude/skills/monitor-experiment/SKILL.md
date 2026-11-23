---
name: monitor-experiment
description: Monitor Beaker experiments until completion. Use when the user asks to monitor, check, or track a Beaker experiment.
allowed-tools: Bash(beaker:*), Bash(jq:*)
---

# Monitor Beaker Experiment

## Instructions

When monitoring a Beaker experiment:

1. Status is per job, and a retried experiment holds several jobs. For each task and replica rank, read the most recently created job, never `jobs[0]`:
   ```bash
   beaker experiment get <experiment-id> --format json | jq -c '.[0].jobs | group_by([.execution.task, .execution.replicaRank]) | map(max_by(.status.created) | {id, rank: .execution.replicaRank, status: (.status | {finalized, exitCode, canceledFor})})'
   ```
2. A job is done once `status.finalized` is set. A cancelled job has no `exitCode`: read `canceledFor`. Beaker can replace a cancelled job (a healthcheck swap, or a preemption when the spec sets `autoResume`), so re-run step 1 a few minutes later before calling a cancellation final: a newer attempt for that rank means the experiment is still running. `exitCode` 0 is success; otherwise show the failing rank's `beaker job logs <job-id> --tail 200`.
3. If a job is not done, check again in about a minute. Claude Code blocks a foreground `sleep`: use Monitor or a background loop. Codex can `sleep`.
4. Continue until every job is done or the user asks you to stop.

Stream a running job's logs with `beaker job logs --follow <job-id>`, using a job id from step 1.
