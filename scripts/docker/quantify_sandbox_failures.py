#!/usr/bin/env python3
"""Quantify Podman/Docker sandbox failures in a training run's logs.

Training on sandbox environments frequently thrashes on Podman concurrency
limits (locks, exec contention, host rotation) and sandbox crashes (OOM,
container death). Today these are only emitted as plain ``WARNING`` /
``echo`` log lines with no counters and no wandb metric -- and some (OOM in
particular) are silently scored as tool-call *successes* in the wandb
``tools/*/failure_rate`` metric. See
``docs/algorithms/monitoring_and_debugging_runs.md`` and
``docs/sandbox_modal_vs_podman.md``.

This script scrapes those log lines and tallies them into counts + rates so
you can size how badly a run is affected. It is post-hoc grep archaeology --
the proper fix is to emit these as wandb metrics (see EnvStatistics in
``open_instruct/environments/tools/utils.py``).

Sources for a Beaker experiment (default) or one/more local log files:

    # From a Beaker experiment (fetches logs of all jobs of all tasks):
    python scripts/docker/quantify_sandbox_failures.py --experiment 01ABC...

    # From a local file, or several, or stdin:
    python scripts/docker/quantify_sandbox_failures.py --log-file run.log
    beaker experiment logs 01ABC... --all-jobs | python scripts/docker/quantify_sandbox_failures.py -

Each signature below cites the code site that emits it (file:line at time of
writing) so the mapping stays auditable.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Signature:
    key: str
    group: str
    description: str
    pattern: re.Pattern
    # Denominators are counted but excluded from the "incident total".
    is_denominator: bool = False
    # If True, a recovered/retried condition (the run often survives these,
    # but a high count means Podman is thrashing under concurrency).
    recovered: bool = False


def _p(regex: str) -> re.Pattern:
    return re.compile(regex)


# Ordered catalog. Each raw log line contains at most one of these signatures,
# except "Docker exec APIError" which is a substring of the transient variant --
# handled with a negative lookbehind so the fatal bucket excludes transient ones.
SIGNATURES: list[Signature] = [
    # -- Denominators (proxy for number of sandbox lifecycles ~= rollouts) ----
    Signature(
        "container_started",
        "denominator",
        "Sandbox containers started (~ rollouts)",
        _p(r"Docker container started:"),
        is_denominator=True,
    ),
    # -- FATAL: Ray actor / node death ('running hot' regime -> job crash) ----
    # These are ABSENT when the client semaphore throttles load (load queues
    # safely). They APPEAR when concurrency/daemon count is raised without node
    # headroom: the node runs hot, processes get OOM/thread-killed, and actor or
    # node death propagates to the driver and crashes the whole job.
    Signature(
        "ray_actor_death",
        "fatal",
        "Ray actor/worker died (ActorDiedError/RayActorError)",
        _p(r"ActorDiedError|RayActorError|The actor died unexpectedly|actor is dead because its worker"),
    ),
    Signature(
        "worker_oom_death",
        "fatal",
        "Ray worker killed by node OOM (exit type NODE_OUT_OF_MEMORY)",
        _p(r"NODE_OUT_OF_MEMORY"),
    ),
    Signature(
        "node_marked_dead",
        "fatal",
        "Ray node died (marked dead / heartbeats / raylet OOM crash)",
        _p(
            r"has been marked dead because the detector has missed too many heartbeats"
            r"|node has died"
            r"|node was terminated unexpectedly"
            r"|health check failed due to missing too many heartbeats"
        ),
    ),
    Signature(
        "raylet_death", "fatal", "Raylet terminated / died", _p(r"Raylet is terminated|[Rr]aylet has died|Raylet died")
    ),
    # -- Node/host resource exhaustion (the 'running hotter' CAUSE) -----------
    Signature(
        "node_oom_kill",
        "resource",
        "Kernel/Ray OOM-killed a process (node low on memory)",
        _p(
            r"killed due to the node running low on memory|killed due to memory pressure \(OOM\)|Out of memory: Killed process"
        ),
    ),
    Signature(
        "process_exhaustion",
        "resource",
        "Process/thread/FD exhaustion (fork/thread/mem) from too many procs",
        _p(
            r"Resource temporarily unavailable|Cannot allocate memory|cannot fork|pthread_create failed|failed to create new OS thread"
        ),
    ),
    # -- Daemon / host-level Podman concurrency & crashes (INFRA) -------------
    Signature(
        "daemon_cannot_clone",
        "daemon",
        "Podman 'cannot clone: Operation not permitted' (nesting/limits)",
        _p(r"cannot clone: Operation not permitted"),
    ),  # scripts/docker/docker_login.sh:225 ; also raw podman stderr
    Signature(
        "daemon_too_many_locks", "daemon", "Podman 'too many locks' (raise PODMAN_NUM_LOCKS)", _p(r"too many locks")
    ),  # not matched in Python; only appears as raw podman stderr
    Signature(
        "daemon_shard_exited",
        "daemon",
        "Podman service shard exited before socket appeared",
        _p(r"Podman service shard \S+ exited before socket appeared"),
    ),  # scripts/docker/docker_login.sh:117
    Signature(
        "daemon_socket_missing",
        "daemon",
        "Podman socket missing / not created",
        _p(r"Podman socket|Podman sockets were not created"),
    ),  # scripts/docker/docker_login.sh:202,209
    Signature(
        "disk_no_space",
        "daemon",
        "No space left on device (container-churn disk exhaustion)",
        _p(r"no space left on device"),
    ),
    # -- Pool host rotation / cooldown (INFRA, pool.py) -----------------------
    Signature(
        "host_rotation",
        "pool",
        "Reset failed on a Podman host -> tried another host",
        _p(r"Reset failed on Podman Docker host"),
        recovered=True,
    ),  # open_instruct/environments/pool.py:171
    Signature(
        "host_cooldown",
        "pool",
        "Podman host temporarily disabled (cooldown)",
        _p(r"Temporarily disabling Podman Docker host"),
    ),  # open_instruct/environments/pool.py:212
    Signature(
        "pool_acquire_timeout",
        "pool",
        "Pool acquire timed out (actor likely crashed unreleased)",
        _p(r"Pool acquire timed out"),
    ),  # open_instruct/environments/pool.py:108
    Signature(
        "actor_crash_not_returned",
        "pool",
        "Crashed env actor not returned to pool after reset failure",
        _p(r"Not returning crashed environment actor to pool"),
    ),  # open_instruct/environments/pool.py:129
    # -- Exec-level transient contention (backends.py, retried) --------------
    Signature(
        "exec_transient_apierror",
        "exec",
        "Transient exec APIError (database locked / exec session), retried",
        _p(r"Transient Docker exec APIError"),
        recovered=True,
    ),  # open_instruct/environments/backends.py:308
    Signature(
        "exec_409_conflict",
        "exec",
        "Exec 409 Conflict (container not running), restarted+retried",
        _p(r"Docker exec 409 Conflict"),
        recovered=True,
    ),  # open_instruct/environments/backends.py:300
    Signature(
        "container_disappeared",
        "exec",
        "Container disappeared before exec, restarted+retried",
        _p(r"Docker container disappeared before exec"),
        recovered=True,
    ),  # open_instruct/environments/backends.py:283
    Signature(
        "exec_apierror_fatal",
        "exec",
        "Non-transient exec APIError (re-raised; may be lock/clone at exec)",
        _p(r"(?<!Transient )Docker exec APIError"),
    ),  # open_instruct/environments/backends.py:318
    # -- OOM: reward-0 episodes that are INVISIBLE to tools/*/failure_rate ----
    Signature(
        "oom_killed",
        "oom",
        "Sandbox OOM-killed (scored success=True in wandb -- undercounted!)",
        _p(r"was OOM-killed|sandbox OOM"),
    ),  # backends.py:297 (SandboxOOMError) ; swerl_sandbox.py:366
    # -- Failures that reach the rollout loop (vllm_utils.py) -----------------
    Signature(
        "step_timeout",
        "rollout",
        "Tool step timed out (tool_call_timeout) -> reward 0",
        _p(r"Step '.*?' timed out after"),
    ),  # open_instruct/vllm_utils.py:1303
    Signature(
        "step_failed", "rollout", "Tool step raised -> reward 0", _p(r"Step '.*?' failed:")
    ),  # open_instruct/vllm_utils.py:1314
    Signature(
        "reset_zero_reward",
        "rollout",
        "Env reset failed -> rollout marked zero reward",
        _p(r"Environment reset failed; marking rollout as zero reward"),
    ),  # open_instruct/vllm_utils.py:1171
    Signature(
        "reset_failed_after",
        "rollout",
        "Reset failed after N attempts (swerl/pool RuntimeError)",
        _p(r"[Rr]eset failed after"),
    ),  # swerl_sandbox.py:189 ; pool.py:179
    # -- Misc lifecycle noise ------------------------------------------------
    Signature(
        "container_stop_error", "misc", "Error stopping container on close()", _p(r"Error stopping container")
    ),  # open_instruct/environments/backends.py:530
    # -- Catch-all: daemon connectivity errors not already attributed above ---
    # Deliberately LAST so specific outcome buckets (reset/rotation) win their
    # own lines; this then captures standalone exec/create connectivity errors
    # -- the surge you'd see when podman is "falling apart" under a hot node.
    # SDK-specific markers only, so it never matches task output that merely
    # contains "connection refused".
    Signature(
        "podman_connectivity",
        "daemon",
        "Podman daemon unreachable (docker-SDK connectivity error)",
        _p(r"[Ee]rror while fetching server API version|UnixHTTPConnectionPool"),
    ),  # open_instruct/environments/backends.py:25-33 (connectivity markers)
]

# Buckets that represent genuine infra pressure worth totalling separately.
# "fatal"/"resource" = the "running hot" regime (job-crashing); tallied
# separately from recoverable INFRA pressure.
FATAL_GROUPS = {"fatal", "resource"}
INFRA_GROUPS = {"daemon", "pool", "exec", "oom"}
GROUP_ORDER = ["fatal", "resource", "daemon", "pool", "exec", "oom", "rollout", "misc", "denominator"]
GROUP_TITLES = {
    "fatal": "FATAL: Ray actor / node death (job-crashing)",
    "resource": "Node resource exhaustion ('running hot' cause)",
    "daemon": "Daemon / host-level Podman crashes",
    "pool": "Pool host rotation / cooldown",
    "exec": "Exec-level contention (retried)",
    "oom": "OOM kills",
    "rollout": "Reached rollout loop (reward 0)",
    "misc": "Misc lifecycle",
    "denominator": "Denominators",
}


@dataclass
class ScanResult:
    counts: Counter = field(default_factory=Counter)
    samples: dict[str, list[str]] = field(default_factory=dict)
    lines_scanned: int = 0


def scan_lines(lines, sample_n: int = 0) -> ScanResult:
    result = ScanResult()
    for raw in lines:
        result.lines_scanned += 1
        line = raw.rstrip("\n")
        for sig in SIGNATURES:
            if sig.pattern.search(line):
                result.counts[sig.key] += 1
                if sample_n:
                    bucket = result.samples.setdefault(sig.key, [])
                    if len(bucket) < sample_n:
                        bucket.append(line.strip()[:300])
                # A line matches at most one signature; stop at the first.
                break
    return result


def fetch_beaker_logs(experiment: str, all_jobs: bool):
    cmd = ["beaker", "experiment", "logs", experiment]
    if all_jobs:
        cmd.append("--all-jobs")
    print(f"# Fetching logs: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    yield from proc.stdout
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read() if proc.stderr else ""
        raise SystemExit(f"beaker logs failed (exit {proc.returncode}): {err.strip()}")


def iter_files(paths):
    for path in paths:
        if path == "-":
            yield from sys.stdin
        else:
            with open(path, errors="replace") as fh:
                yield from fh


def format_report(result: ScanResult, sample_n: int) -> str:
    denom = result.counts.get("container_started", 0)

    def rate(n: int) -> str:
        if not denom:
            return "   n/a"
        return f"{100.0 * n / denom:5.2f}%"

    by_key = {s.key: s for s in SIGNATURES}
    out: list[str] = []
    out.append("=" * 74)
    out.append("Sandbox / Podman failure tally")
    out.append(f"lines scanned: {result.lines_scanned:,}    container starts (~rollouts): {denom:,}")
    out.append("=" * 74)
    header = f"{'signature':<26}{'count':>8}{'rate/roll':>11}   description"
    for group in GROUP_ORDER:
        keys = [s.key for s in SIGNATURES if s.group == group and result.counts.get(s.key)]
        if not keys:
            continue
        out.append("")
        out.append(f"-- {GROUP_TITLES[group]} " + "-" * max(0, 60 - len(GROUP_TITLES[group])))
        out.append(header)
        for key in keys:
            sig = by_key[key]
            n = result.counts[key]
            rate_str = "" if sig.is_denominator else rate(n)
            tag = " [recovered]" if sig.recovered else ""
            out.append(f"{key:<26}{n:>8}{rate_str:>11}   {sig.description}{tag}")

    def group_total(groups) -> int:
        return sum(c for k, c in result.counts.items() if k in by_key and by_key[k].group in groups)

    fatal_total = group_total(FATAL_GROUPS)
    infra_total = group_total(INFRA_GROUPS)
    rollout_total = group_total({"rollout"})
    out.append("")
    out.append("=" * 74)
    out.append(
        f"FATAL 'running hot' incidents (actor/node death + exhaustion): "
        f"{fatal_total:,}   rate/roll: {rate(fatal_total).strip()}"
    )
    out.append(
        f"INFRA incidents total (daemon+pool+exec+oom): {infra_total:,}   rate/roll: {rate(infra_total).strip()}"
    )
    out.append(
        f"Rollout-visible failures (reward 0):          {rollout_total:,}   rate/roll: {rate(rollout_total).strip()}"
    )
    out.append("=" * 74)
    if fatal_total:
        out.append(
            "WARNING: FATAL signatures present -- the node likely ran hot "
            "(too many\n         processes for its RAM/CPU). Prefer more "
            "NODES / lower pool_size\n         over more daemons-per-node."
        )
    else:
        out.append(
            "No FATAL (actor/node-death) signatures -- failures stayed at the "
            "recoverable\n      daemon layer; actors survived (the throttled, "
            "not 'running hot', regime)."
        )
    if not denom:
        out.append("NOTE: no 'Docker container started' lines found -- rates unavailable.")
        out.append("      (logs truncated, or a non-Docker/Podman backend.)")
    out.append(
        "NOTE: oom_killed episodes are scored success=True in wandb "
        "tools/*/failure_rate,\n      so they are UNDERCOUNTED there but "
        "surfaced here."
    )

    if sample_n:
        out.append("")
        out.append("-- sample lines " + "-" * 58)
        for group in GROUP_ORDER:
            for sig in SIGNATURES:
                if sig.group == group and sig.key in result.samples:
                    out.append(f"[{sig.key}]")
                    for s in result.samples[sig.key]:
                        out.append(f"    {s}")
    return "\n".join(out)


def resolve_source(spec: str, all_jobs: bool):
    """Return a line iterator for a compare source spec.

    ``exp:<id>`` fetches a Beaker experiment; anything else is a file path
    (``-`` for stdin).
    """
    if spec.startswith("exp:"):
        return fetch_beaker_logs(spec[len("exp:") :], all_jobs=all_jobs)
    return iter_files([spec])


def format_compare(results: list) -> str:
    """Render a side-by-side table. ``results`` is a list of (label, ScanResult)."""
    by_key = {s.key: s for s in SIGNATURES}
    denoms = [r.counts.get("container_started", 0) for _, r in results]
    ncol = len(results)
    sig_w, col_w = 26, 18
    tags = [chr(ord("A") + i) for i in range(ncol)]

    def cell(count: int, denom: int, is_denom: bool) -> str:
        if is_denom:
            return f"{count:,}"
        if not denom:
            return f"{count:,} (n/a)"
        return f"{count:,} ({100.0 * count / denom:.1f}%)"

    hdr = f"{'signature':<{sig_w}}" + "".join(f"{t:>{col_w}}" for t in tags)
    out = ["=" * (sig_w + ncol * col_w)]
    out.append("Side-by-side comparison   (cell = count (rate/rollout))")
    for tag, (label, _) in zip(tags, results):
        out.append(f"  {tag} = {label}")
    out.append("=" * (sig_w + ncol * col_w))
    out.append(hdr)
    out.append(f"{'container starts (~rollouts)':<{sig_w}}" + "".join(f"{cell(d, d, True):>{col_w}}" for d in denoms))

    for group in GROUP_ORDER:
        if group == "denominator":
            continue
        keys = [s.key for s in SIGNATURES if s.group == group and any(r.counts.get(s.key) for _, r in results)]
        if not keys:
            continue
        out.append("")
        out.append(f"-- {GROUP_TITLES[group]} " + "-" * max(0, 60 - len(GROUP_TITLES[group])))
        for key in keys:
            row = f"{key:<{sig_w}}"
            for i, (_, r) in enumerate(results):
                row += f"{cell(r.counts.get(key, 0), denoms[i], False):>{col_w}}"
            out.append(row)

    def gt(counts, groups) -> int:
        return sum(c for k, c in counts.items() if k in by_key and by_key[k].group in groups)

    out.append("")
    out.append("=" * (sig_w + ncol * col_w))
    for name, groups in [
        ("FATAL (actor/node death)", FATAL_GROUPS),
        ("INFRA recoverable total", INFRA_GROUPS),
        ("Rollout reward-0 total", {"rollout"}),
    ]:
        row = f"{name:<{sig_w}}"
        for i, (_, r) in enumerate(results):
            row += f"{cell(gt(r.counts, groups), denoms[i], False):>{col_w}}"
        out.append(row)
    out.append("=" * (sig_w + ncol * col_w))
    out.append(
        "FATAL>0 = node ran hot (job-crashing). High INFRA with FATAL=0 = "
        "throttled but\nsurviving. See per-signature rows above for the profile shift."
    )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--experiment", "-e", help="Beaker experiment ID to fetch logs from")
    src.add_argument("--log-file", "-f", nargs="+", metavar="PATH", help="Local log file(s); use '-' for stdin")
    src.add_argument(
        "--compare",
        "-c",
        nargs="+",
        metavar="SRC",
        help="Compare 2+ sources side-by-side. Each SRC is a log file path or exp:<beaker_id>.",
    )
    ap.add_argument("positional", nargs="?", help="Same as --log-file (e.g. '-' for stdin)")
    ap.add_argument(
        "--single-job",
        action="store_true",
        help="For --experiment/exp: sources: only most-recent job per task (default: --all-jobs)",
    )
    ap.add_argument("--samples", type=int, default=0, metavar="N", help="Print up to N example lines per signature")
    ap.add_argument("--json", action="store_true", help="Emit raw counts as JSON")
    args = ap.parse_args()

    if args.compare:
        if len(args.compare) < 2:
            ap.error("--compare needs at least 2 sources")
        results = [(spec, scan_lines(resolve_source(spec, all_jobs=not args.single_job))) for spec in args.compare]
        if args.json:
            by_key = {s.key: s for s in SIGNATURES}
            print(
                json.dumps(
                    {
                        spec: {"container_starts": r.counts.get("container_started", 0), "counts": dict(r.counts)}
                        for spec, r in results
                    },
                    indent=2,
                )
            )
        else:
            print(format_compare(results))
        return

    if args.experiment:
        lines = fetch_beaker_logs(args.experiment, all_jobs=not args.single_job)
    else:
        paths = args.log_file or ([args.positional] if args.positional else None)
        if not paths:
            ap.error("provide --experiment, --log-file, --compare, or a positional path ('-' for stdin)")
        lines = iter_files(paths)

    result = scan_lines(lines, sample_n=args.samples)

    if args.json:
        by_key = {s.key: s for s in SIGNATURES}
        denom = result.counts.get("container_started", 0)
        payload = {
            "lines_scanned": result.lines_scanned,
            "container_starts": denom,
            "counts": dict(result.counts),
            "rates_per_rollout": {
                k: (result.counts[k] / denom if denom else None) for k in result.counts if not by_key[k].is_denominator
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(result, sample_n=args.samples))


if __name__ == "__main__":
    main()
