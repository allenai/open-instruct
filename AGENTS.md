# Bash commands
- `uv run pytest`: Run the tests.
- `make style && make quality` run the linter + formatter.
- `uv run mkdocs serve`: View the documentation locally at http://127.0.0.1:8000/
- `uv run mkdocs build`: Build the documentation to the `site/` directory.

# Workflow
- Always run the linter and make sure the tests pass before finishing a task.
- Prefer running single tests, not the whole suite, when developing.
- To run the `./scripts/train/build_image_and_launch.sh` script, you must commit the current changes.
- Launch tool use experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/tool_grpo_fast.sh`.
- Launch multi-node non-tool experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/large_test_script.sh`.

# Monitoring Beaker Experiments

After launching an experiment with `./scripts/train/build_image_and_launch.sh`, the script outputs a Beaker URL like:
```
https://beaker.org/ex/01KCHBGMH5NTX77PC17EZ5HJTB
```

The experiment ID is the last part of the URL (e.g., `01KCHBGMH5NTX77PC17EZ5HJTB`).

## Using the monitor script

```bash
# Check status once
./scripts/train/monitor_experiment.sh <experiment_id>

# Wait for completion (polls every 30s)
./scripts/train/monitor_experiment.sh <experiment_id> --wait

# Wait and show logs when done
./scripts/train/monitor_experiment.sh <experiment_id> --wait --logs
```

## Manual monitoring commands

```bash
# Get experiment status
beaker experiment get <experiment_id> --format json | jq '.[0].jobs[0].status'

# Get exit code (0 = success, non-zero = failure, null = still running)
beaker experiment get <experiment_id> --format json | jq '.[0].jobs[0].status.exitCode'

# Get job ID for logs
beaker experiment get <experiment_id> --format json | jq -r '.[0].jobs[0].id'

# View job logs (replace <job_id> with actual job ID)
beaker job logs <job_id> | tail -100
```

## Interpreting status

- `scheduled`: Job is queued and waiting for resources
- `started`: Job is running
- `exited` with `exitCode: 0`: Job completed successfully
- `exited` with `exitCode: 1` (or other non-zero): Job failed - check logs

# Documentation
To verify that documentation changes don't alter the generated output:
1. Build docs on your branch: `uv run mkdocs build && cp -r site site-branch`
2. Switch to main branch and build: `cd /path/to/main && uv run mkdocs build`
3. Compare the builds: `diff -rq site-branch /path/to/main/site`
4. If no output, the docs are identical. If differences exist, review with: `diff -r site-branch /path/to/main/site`
