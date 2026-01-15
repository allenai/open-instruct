---
name: monitor-experiment
description: Monitor Beaker experiments until completion. Use when the user asks to monitor, check, or track a Beaker experiment.
allowed-tools: Bash(beaker:*)
---

# Monitor Beaker Experiment

## Instructions

When monitoring a Beaker experiment:

1. Get the experiment status using `beaker experiment get <experiment-id>`
2. Check if the experiment has completed by looking at `status.exited`
3. If still running, wait 30 seconds and check again
4. When complete:
   - If exitCode is 0: Report success
   - If exitCode is non-zero: Fetch and display logs with `beaker experiment logs <experiment-id>`
5. Continue monitoring until the experiment finishes or the user asks you to stop

## Examples

Check experiment status:
```bash
beaker experiment get 01KCW39T5JBZTYV69BXHWJJ83P
```

Get experiment logs on failure:
```bash
beaker experiment logs 01KCW39T5JBZTYV69BXHWJJ83P
```

Stream logs in real-time for running experiments:
```bash
beaker experiment logs --follow 01KCW39T5JBZTYV69BXHWJJ83P
```
