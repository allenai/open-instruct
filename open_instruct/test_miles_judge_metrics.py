"""Guard completed logical judge requests across optimizer boundaries."""

import asyncio
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
