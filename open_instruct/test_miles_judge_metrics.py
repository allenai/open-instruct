"""Guard completed logical judge requests across optimizer boundaries."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from open_instruct.miles.rewards import judge_metrics


def test_remote_finish_forwards_instruction_ids_with_the_single_terminal_event():
    finish = Mock()
    remote = judge_metrics._Remote(SimpleNamespace(finish=SimpleNamespace(remote=finish)))
    ids = ["detectable_content:postscript", "keywords:existence"]
    remote.finish(False, 3, 2.0, 1.0, True, ids)
    finish.assert_called_once_with(False, 3, 2.0, 1.0, True, ids)


def complete_requests(metrics, requests, failures):
    for index in range(requests):
        metrics.arrive()
        metrics.admit()
        failed = index < failures
        metrics.finish(not failed, 3 if failed else 1, 1.0, 1.0, terminal_failure=failed)


@pytest.mark.parametrize("failures", [4, 5, 6])
def test_guard_threshold_over_fifty_updates(failures):
    async def run():
        metrics = judge_metrics.JudgeMetrics()
        for update in range(50):
            complete_requests(metrics, 20, failures if update == 0 else 0)
            counters = await metrics.optimizer_boundary()
            if update < 49:
                judge_metrics.check_failure_guard(counters)
        assert counters["judge_window_updates"] == 50
        assert counters["judge_requests_window"] == 1000
        assert counters["judge/terminal_failures"] == failures
        assert counters["judge/terminal_failure_rate_window"] == failures / 1000
        if failures > 5:
            with pytest.raises(RuntimeError, match="zero_guarded.*6/1000.*50 consecutive optimizer updates.*0.5%"):
                judge_metrics.check_failure_guard(counters)
        else:
            judge_metrics.check_failure_guard(counters)

    asyncio.run(run())


def test_guard_uses_cumulative_warmup_and_waits_for_one_thousand_requests():
    async def run():
        metrics = judge_metrics.JudgeMetrics()
        complete_requests(metrics, 1, 1)
        judge_metrics.check_failure_guard(await metrics.optimizer_boundary())
        complete_requests(metrics, 998, 5)
        counters = await metrics.optimizer_boundary()
        assert counters["judge_requests_window"] == 999
        judge_metrics.check_failure_guard(counters)
        complete_requests(metrics, 1, 0)
        counters = await metrics.optimizer_boundary()
        with pytest.raises(RuntimeError, match="6/1000.*3 consecutive optimizer updates"):
            judge_metrics.check_failure_guard(counters)

    asyncio.run(run())


def test_guard_rolls_off_old_counts_and_detects_new_window_despite_lifetime_rate():
    async def run():
        metrics = judge_metrics.JudgeMetrics()
        complete_requests(metrics, 10000, 0)
        judge_metrics.check_failure_guard(await metrics.optimizer_boundary())
        for _ in range(49):
            complete_requests(metrics, 20, 0)
            judge_metrics.check_failure_guard(await metrics.optimizer_boundary())
        complete_requests(metrics, 20, 6)
        counters = await metrics.optimizer_boundary()
        assert counters["judge_requests"] == 11000
        assert counters["judge_requests_window"] == 1000
        assert counters["judge/terminal_failure_rate_window"] == 0.006
        with pytest.raises(RuntimeError, match="6/1000"):
            judge_metrics.check_failure_guard(counters)
        for _ in range(50):
            complete_requests(metrics, 20, 0)
            counters = await metrics.optimizer_boundary()
        assert counters["judge/terminal_failures"] == 6
        assert counters["judge/terminal_failure_rate_window"] == 0
        judge_metrics.check_failure_guard(counters)

    asyncio.run(run())


def test_fifty_update_window_keeps_minimum_denominator_and_empty_updates_age_counts():
    async def run():
        metrics = judge_metrics.JudgeMetrics()
        complete_requests(metrics, 1, 1)
        for _ in range(49):
            judge_metrics.check_failure_guard(await metrics.optimizer_boundary())
        counters = await metrics.optimizer_boundary()
        assert counters["judge_window_updates"] == 50
        assert counters["judge_requests_window"] == 1
        judge_metrics.check_failure_guard(counters)
        counters = await metrics.optimizer_boundary()
        assert counters["judge_requests_window"] == 0
        assert counters["judge/terminal_failure_rate_window"] == 0
        judge_metrics.check_failure_guard(counters)

    asyncio.run(run())


def test_family_and_instruction_counts_use_logical_requests_and_never_gate():
    async def run():
        metrics = judge_metrics.JudgeMetrics()
        ids = ["detectable_content:postscript", "keywords:existence", "keywords:frequency", "keywords:existence"]
        for failed, instructions in ((True, ids), (False, ids[1:]), (False, ["startend:quotation"])):
            metrics.arrive()
            metrics.admit()
            metrics.finish(not failed, 3, 1.0, 1.0, failed, instructions)
        metrics.skip()
        metrics.arrive()
        metrics.finish(False, 0, 1.0, instruction_ids=ids)  # cancelled
        metrics.arrive()
        metrics.admit()
        metrics.finish(False, 3, 1.0, 1.0, instruction_ids=ids)  # fatal
        # Plenty of successful grades make the overall rate small, even though
        # the postscript family's one request failed. Family rates never gate.
        complete_requests(metrics, 1000, 0)
        state = await metrics.optimizer_boundary()
        assert state["judge_requests"] == 1003
        assert state["judge/terminal_failures"] == 1
        for scope, expected in (
            ("family", {"detectable_content": (1, 1), "keywords": (1, 2), "startend": (0, 1)}),
            (
                "instruction",
                {
                    "detectable_content:postscript": (1, 1),
                    "keywords:existence": (1, 2),
                    "keywords:frequency": (1, 2),
                    "startend:quotation": (0, 1),
                },
            ),
        ):
            for key, (failures, requests) in expected.items():
                assert state[f"judge/terminal_failures_by_{scope}/{key}"] == failures
                assert state[f"judge/requests_by_{scope}/{key}"] == requests
        judge_metrics.check_failure_guard(state)

    asyncio.run(run())


def test_only_completed_grades_enter_guard_and_snapshot_does_not_advance_updates():
    async def run():
        metrics = judge_metrics.JudgeMetrics()
        complete_requests(metrics, 999, 6)
        metrics.skip()
        metrics.arrive()
        metrics.finish(False, 0, 1.0)  # cancelled before admission
        metrics.arrive()
        metrics.admit()
        metrics.finish(False, 3, 1.0, 1.0)  # fatal context/programming error
        counters = await metrics.optimizer_boundary()
        assert counters["completed"] == 1001
        assert counters["failed"] == 8
        assert counters["judge_requests_window"] == 999
        assert counters["judge/terminal_failures"] == 6
        for _ in range(60):
            counters = await metrics.snapshot()
            assert counters["judge_window_updates"] == 1
            judge_metrics.check_failure_guard(counters)

    asyncio.run(run())


@pytest.mark.parametrize("failures", [5, 6])
def test_pending_window_obeys_threshold_without_completing_an_update(failures):
    async def run():
        metrics = judge_metrics.JudgeMetrics()
        complete_requests(metrics, 999, failures)
        for _ in range(60):
            judge_metrics.check_failure_guard(await metrics.pending_window())
        complete_requests(metrics, 1, 0)
        pending = await metrics.pending_window()
        assert pending["judge_window_updates"] == 1
        assert pending["judge_requests_window"] == 1000
        if failures == 6:
            with pytest.raises(RuntimeError, match="6/1000.*pending update"):
                judge_metrics.check_failure_guard(pending)
        else:
            judge_metrics.check_failure_guard(pending)
        assert (await metrics.snapshot())["judge_window_updates"] == 0
        closed = await metrics.optimizer_boundary()
        assert closed["judge_requests_window"] == 1000
        assert closed["terminal_failures_window"] == failures
        assert closed["judge_window_updates"] == 1

    asyncio.run(run())


def test_pending_window_keeps_only_49_completed_updates_and_never_ages_them():
    async def run():
        metrics = judge_metrics.JudgeMetrics()
        complete_requests(metrics, 10000, 0)
        await metrics.optimizer_boundary()
        for _ in range(49):
            complete_requests(metrics, 20, 0)
            await metrics.optimizer_boundary()
        complete_requests(metrics, 20, 6)
        for _ in range(60):
            pending = await metrics.pending_window()
            assert pending["judge_window_updates"] == 50
            assert pending["judge_requests_window"] == 1000
            with pytest.raises(RuntimeError, match="6/1000"):
                judge_metrics.check_failure_guard(pending)
        closed = await metrics.snapshot()
        assert closed["judge_requests_window"] == 10980
        assert closed["terminal_failures_window"] == 0
        boundary = await metrics.optimizer_boundary()
        assert boundary["judge_requests_window"] == pending["judge_requests_window"]
        assert boundary["terminal_failures_window"] == pending["terminal_failures_window"]

    asyncio.run(run())


def boundary_line(step, requests, failures, unix):
    counters = {"step": step, "judge_requests": requests, "judge/terminal_failures": failures, "unix": unix}
    return f"2026-10-18 INFO driver.py:199 - h087_judge_boundary {json.dumps(counters)}\n"


def write_driver_log(root, attempt, boundaries):
    """boundaries: (step, cumulative requests, cumulative failures, unix) within one attempt."""
    log = root / "cluster" / attempt / "replica" / "driver-1.log"
    log.parent.mkdir(parents=True)
    log.write_text("".join(boundary_line(*b) for b in boundaries))


def test_resume_at_u76_rebuilds_the_runs_window_from_driver_logs(tmp_path):
    # Attempt 1 reached u99 (1,000 requests each, a failure every 10th update); its u76-u99 are lost.
    write_driver_log(tmp_path, "a1", [(s, 1000 * s, s // 10, 1000.0 + s) for s in range(1, 100)])
    history = judge_metrics.resume_history(tmp_path, 75)
    assert len(history) == judge_metrics.FAILURE_WINDOW_UPDATES and history[-1] == (1000, 0)
    assert history[-6] == (1000, 1)  # u70
    metrics = judge_metrics.JudgeMetrics()
    metrics.seed_window(history)
    # u76 pending: u27..u75 closed plus the pending update, never a fresh window.
    complete_requests(metrics, 991, 9)
    window = asyncio.run(metrics.pending_window())
    assert window["judge_window_updates"] == judge_metrics.FAILURE_WINDOW_UPDATES
    assert window["judge_requests_window"] == 1000 * 49 + 991
    assert window["terminal_failures_window"] == 5 + 9  # u30, u40, u50, u60, u70 and the pending update
    # Nine failures in the first resumed update no longer halt (H087 s18, 10-18).
    judge_metrics.check_failure_guard(window)
    # Without the rebuilt window the same update trips, as it did.
    fresh = judge_metrics.JudgeMetrics()
    complete_requests(fresh, 1005, 9)
    with pytest.raises(RuntimeError, match="exceeds 0.5%"):
        judge_metrics.check_failure_guard(asyncio.run(fresh.pending_window()))


def test_replayed_updates_and_persisted_records_supersede_lost_attempts(tmp_path):
    write_driver_log(tmp_path, "a1", [(s, 10 * s, 0, 100.0 + s) for s in range(1, 80)])
    # Attempt 2 resumed from u75 and replayed u76-u77 with failures.
    write_driver_log(tmp_path, "a2", [(76, 10, 2, 900.0), (77, 20, 3, 901.0)])
    judge_metrics.record_update(tmp_path, 77, 30, 7, 950.0)
    history = judge_metrics.resume_history(tmp_path, 77)
    assert history[-2:] == [(10, 2), (30, 7)]


def test_resume_before_u50_stays_cumulative_from_u1(tmp_path):
    for step in range(1, 31):
        judge_metrics.record_update(tmp_path, step, 40, int(step == 3), float(step))
    history = judge_metrics.resume_history(tmp_path, 30)
    assert len(history) == 30 and sum(r for r, _ in history) == 1200 and sum(f for _, f in history) == 1
    assert judge_metrics.resume_history(tmp_path, 0) == []


def test_missing_record_refuses_to_start(tmp_path):
    write_driver_log(tmp_path, "a1", [(s, 10 * s, 0, float(s)) for s in range(1, 60) if s != 40])
    with pytest.raises(RuntimeError, match="refusing to start"):
        judge_metrics.resume_history(tmp_path, 59)
    with pytest.raises(RuntimeError, match="refusing to start"):
        judge_metrics.resume_history(tmp_path / "empty", 10)
