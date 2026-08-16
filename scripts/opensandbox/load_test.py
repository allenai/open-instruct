#!/usr/bin/env python3
"""Synthetic load test for an OpenSandbox deployment.

Simulates the sandbox churn of one or more concurrent RL training runs
WITHOUT GPUs, by driving ``OpenSandboxBackend`` (the exact production code
path: create throttle, adopt-on-504, SandboxDiedError handling, registry
auth) with N concurrent workers. Each worker loops episodes:
start -> a few execs spread over the episode duration -> close.

Sizing guide: one ps512 prod run sustains ~512 concurrent sandboxes with
~3-5 min episodes. To test "two concurrent full runs", use
--workers 1024; for headroom, 1536.

The client-side create throttle (SWERL_OPENSANDBOX_START_CONCURRENCY,
default 64) is PER PROCESS/NODE — one load-test process equals one training
node. To simulate the create concurrency of a 2-node job, export
SWERL_OPENSANDBOX_START_CONCURRENCY=128 (or run two processes).

Usage (endpoint config comes from the usual env vars):
    export SWERL_OPENSANDBOX_DOMAIN=... SWERL_OPENSANDBOX_PROTOCOL=https \
           OPEN_SANDBOX_API_KEY=...
    python scripts/opensandbox/load_test.py --workers 1024 --duration-s 3600

Watch alongside on the cluster:
    kubectl top pods -n opensandbox-system
    kubectl get batchsandbox -n opensandbox --no-headers | wc -l

Exit code is nonzero if the success criteria (--max-create-p95-s,
--max-failure-rate) are breached. Cleanup: episodes close their own
sandboxes; stragglers are reclaimed by the app-tag janitor
(cleanup_opensandbox_sandboxes.sh <app-name>) and the GC CronJob.
"""

import argparse
import os
import random
import threading
import time
from dataclasses import dataclass, field

from open_instruct import logger_utils
from open_instruct.environments.backends import OpenSandboxBackend, SandboxDiedError

logger = logger_utils.setup_logger(__name__)


@dataclass
class Stats:
    """Thread-safe counters and latency samples."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    creates_ok: int = 0
    creates_failed: int = 0
    create_failures_by_kind: dict = field(default_factory=dict)
    execs_ok: int = 0
    execs_failed: int = 0
    deaths: int = 0
    episodes_done: int = 0
    active: int = 0
    active_peak: int = 0
    started_at: float = field(default_factory=time.monotonic)
    create_latencies: list = field(default_factory=list)  # (offset_from_start_s, latency_s)
    exec_latencies: list = field(default_factory=list)

    def record_create(self, ok: bool, latency_s: float | None = None, kind: str = "") -> None:
        with self.lock:
            if ok:
                self.creates_ok += 1
                self.create_latencies.append((time.monotonic() - self.started_at, latency_s))
                self.active += 1
                self.active_peak = max(self.active_peak, self.active)
            else:
                self.creates_failed += 1
                self.create_failures_by_kind[kind] = self.create_failures_by_kind.get(kind, 0) + 1

    def record_close(self, died: bool) -> None:
        with self.lock:
            self.active = max(0, self.active - 1)
            self.episodes_done += 1
            if died:
                self.deaths += 1

    def record_exec(self, ok: bool, latency_s: float | None = None) -> None:
        with self.lock:
            if ok:
                self.execs_ok += 1
                self.exec_latencies.append(latency_s)
            else:
                self.execs_failed += 1


def percentile(values: list, fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def classify_create_failure(error: Exception) -> str:
    message = str(error)
    for marker in ("503", "504", "health check", "connection", "adopt"):
        if marker.lower() in message.lower():
            return marker
    return type(error).__name__


def worker_loop(args, stats: Stats, stop_event: threading.Event, worker_id: int) -> None:
    # Jitter the initial start so the ramp resembles a pool filling, not a bomb.
    time.sleep(random.uniform(0, args.ramp_s))
    while not stop_event.is_set():
        backend = OpenSandboxBackend(
            image=args.image,
            timeout=args.exec_timeout_s,
            mem_limit=args.mem_limit,
            cpu=args.cpu,
            app_name=args.app_name,
            sandbox_lifetime=args.sandbox_lifetime_s,
        )
        create_start = time.perf_counter()
        try:
            backend.start()
        except Exception as e:
            stats.record_create(ok=False, kind=classify_create_failure(e))
            time.sleep(random.uniform(1, 5))  # Mirror env reset retry pacing loosely.
            continue
        stats.record_create(ok=True, latency_s=time.perf_counter() - create_start)

        died = False
        exec_gap_s = args.episode_s / max(1, args.execs_per_episode)
        try:
            for _ in range(args.execs_per_episode):
                if stop_event.is_set():
                    break
                exec_start = time.perf_counter()
                try:
                    result = backend.run_command("echo load-test && ls / > /dev/null", timeout=60)
                    stats.record_exec(ok=result.exit_code == 0, latency_s=time.perf_counter() - exec_start)
                except SandboxDiedError:
                    stats.record_exec(ok=False)
                    died = True
                    break
                except Exception:
                    stats.record_exec(ok=False)
                # Idle between steps, like a rollout waiting on generation.
                stop_event.wait(random.uniform(0.5 * exec_gap_s, 1.5 * exec_gap_s))
        finally:
            try:
                backend.close()
            except Exception:
                logger.warning("worker %s: close failed", worker_id, exc_info=True)
            stats.record_close(died=died)


def reporter_loop(stats: Stats, stop_event: threading.Event, interval_s: float = 30.0) -> None:
    while not stop_event.wait(interval_s):
        with stats.lock:
            latencies = [latency for _, latency in stats.create_latencies]
            create_p50 = percentile(latencies, 0.5)
            create_p95 = percentile(latencies, 0.95)
            logger.info(
                "active=%d (peak %d) episodes=%d creates ok/fail=%d/%d create p50/p95=%.1fs/%.1fs "
                "execs ok/fail=%d/%d deaths=%d",
                stats.active,
                stats.active_peak,
                stats.episodes_done,
                stats.creates_ok,
                stats.creates_failed,
                create_p50,
                create_p95,
                stats.execs_ok,
                stats.execs_failed,
                stats.deaths,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=512, help="Concurrent sandbox episodes to sustain")
    parser.add_argument("--duration-s", type=int, default=1800, help="Total test duration")
    parser.add_argument("--episode-s", type=int, default=240, help="Mean episode duration (holds the sandbox)")
    parser.add_argument("--execs-per-episode", type=int, default=8)
    parser.add_argument("--ramp-s", type=int, default=300, help="Worker start jitter window")
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument("--cpu", type=float, default=1.0)
    parser.add_argument("--mem-limit", default="4g")
    parser.add_argument("--exec-timeout-s", type=int, default=120)
    parser.add_argument("--sandbox-lifetime-s", type=int, default=1800)
    parser.add_argument("--app-name", default="opensandbox-load-test", help="Metadata tag for janitor cleanup")
    parser.add_argument("--max-create-p95-s", type=float, default=30.0, help="Success criterion")
    parser.add_argument("--max-failure-rate", type=float, default=0.02, help="Success criterion (create failures)")
    args = parser.parse_args()

    if not os.getenv("SWERL_OPENSANDBOX_DOMAIN"):
        parser.error("SWERL_OPENSANDBOX_DOMAIN must be set")

    logger.info(
        "Load test: %d workers, %ds, episode~%ds x %d execs, image=%s, app=%s, create throttle/process=%s",
        args.workers,
        args.duration_s,
        args.episode_s,
        args.execs_per_episode,
        args.image,
        args.app_name,
        os.getenv("SWERL_OPENSANDBOX_START_CONCURRENCY", "64 (default)"),
    )

    stats = Stats()
    stop_event = threading.Event()
    threads = [
        threading.Thread(target=worker_loop, args=(args, stats, stop_event, i), daemon=True)
        for i in range(args.workers)
    ]
    reporter = threading.Thread(target=reporter_loop, args=(stats, stop_event), daemon=True)
    for thread in threads:
        thread.start()
    reporter.start()

    try:
        time.sleep(args.duration_s)
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down workers.")
    stop_event.set()
    logger.info("Waiting for in-flight episodes to close (up to 120s)...")
    deadline = time.time() + 120
    for thread in threads:
        thread.join(timeout=max(0, deadline - time.time()))

    with stats.lock:
        total_creates = stats.creates_ok + stats.creates_failed
        failure_rate = stats.creates_failed / total_creates if total_creates else 0.0
        all_latencies = [latency for _, latency in stats.create_latencies]
        # Judge the p95 criterion on steady state: creates after the ramp
        # window, so a cold cluster's node-provisioning starts don't fail an
        # otherwise-healthy test. Fall back to all creates if too few remain.
        steady_cutoff_s = args.ramp_s + 120
        steady_latencies = [latency for offset, latency in stats.create_latencies if offset > steady_cutoff_s]
        criteria_basis = "steady-state" if len(steady_latencies) >= 5 else "all-creates"
        create_p95 = percentile(steady_latencies if criteria_basis == "steady-state" else all_latencies, 0.95)
        print("\n================ LOAD TEST SUMMARY ================")
        print(f"episodes completed:    {stats.episodes_done}")
        print(f"peak concurrent:       {stats.active_peak} (target {args.workers})")
        print(f"creates ok/failed:     {stats.creates_ok}/{stats.creates_failed} (failure rate {failure_rate:.2%})")
        print(f"create failures:       {stats.create_failures_by_kind}")
        print(
            f"create latency (all):  p50={percentile(all_latencies, 0.5):.1f}s "
            f"p95={percentile(all_latencies, 0.95):.1f}s max={max(all_latencies, default=0):.1f}s"
        )
        print(
            f"create latency (steady, >{steady_cutoff_s}s in): "
            f"p95={percentile(steady_latencies, 0.95):.1f}s over {len(steady_latencies)} creates"
        )
        print(
            f"execs ok/failed:       {stats.execs_ok}/{stats.execs_failed} | "
            f"exec p95={percentile(stats.exec_latencies, 0.95):.2f}s"
        )
        print(f"mid-episode deaths:    {stats.deaths} (Spot preemptions; expected nonzero on Spot)")
        print(f"sandboxes left behind: {stats.active} (janitor: cleanup_opensandbox_sandboxes.sh {args.app_name})")

        passed = failure_rate <= args.max_failure_rate and create_p95 <= args.max_create_p95_s
        print(
            f"\nRESULT: {'PASS' if passed else 'FAIL'} "
            f"(criteria on {criteria_basis} creates: failure rate {failure_rate:.2%} <= {args.max_failure_rate:.0%}, "
            f"create p95 {create_p95:.1f}s <= {args.max_create_p95_s}s)"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
