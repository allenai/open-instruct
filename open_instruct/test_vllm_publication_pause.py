"""Requests must drain before pausing vLLM and stay blocked until publication ends."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from open_instruct import vllm_utils


@pytest.fixture
def actor(monkeypatch):
    value = object.__new__(vllm_utils.LLMRayActor)
    value.inflight_updates = False
    value.active_tasks = {"old-request": Mock()}
    value._request_admission_lock = threading.Lock()
    value._publication_paused = False
    value.check_background_threads = Mock()
    value.llm_engine = SimpleNamespace(sleep=AsyncMock(), wake_up=AsyncMock())
    value._run_async = asyncio.run
    monkeypatch.setattr(vllm_utils, "DRAIN_ACTIVE_TASKS_SLEEP_S", 0.01)
    return value


def test_requests_finish_before_engine_sleep_and_resume_after_wake(actor):
    events = []

    def finish_requests():
        assert actor._publication_paused
        actor.active_tasks.clear()
        events.append("requests-finished")

    async def engine_sleep(**kwargs):
        assert not actor.active_tasks
        events.append("engine-sleep")

    actor.check_background_threads.side_effect = finish_requests
    actor.llm_engine.sleep.side_effect = engine_sleep
    actor.sleep()
    assert events == ["requests-finished", "engine-sleep"]
    assert actor._publication_paused
    actor.wake_up()
    actor.llm_engine.wake_up.assert_awaited_once_with(tags=["scheduling"])
    assert not actor._publication_paused


def test_failed_drain_never_sleeps_engine_or_reopens_admission(actor, monkeypatch):
    monkeypatch.setattr(vllm_utils, "DRAIN_ACTIVE_TASKS_TIMEOUT_S", 0)
    with pytest.raises(TimeoutError, match="could not drain 1 active requests"):
        actor.sleep()
    actor.llm_engine.sleep.assert_not_called()
    assert actor._publication_paused


def test_inflight_mode_retains_existing_non_draining_behavior(actor):
    actor.inflight_updates = True
    actor.sleep()
    actor.check_background_threads.assert_not_called()
    assert actor.active_tasks
    actor.llm_engine.sleep.assert_awaited_once()


def test_request_dequeued_during_publication_waits_for_wake(actor, monkeypatch):
    queue_waiting = threading.Event()
    release_queue = threading.Event()
    admitted = threading.Event()
    stop = threading.Event()
    errors = []
    actor.active_tasks.clear()
    actor.inference_batch_size = 16

    class StopWorker(Exception):
        pass

    def should_stop():
        if stop.is_set():
            raise StopWorker
        return False

    def dequeue():
        queue_waiting.set()
        assert release_queue.wait(3)
        return "late-request"

    def admit(current_actor, request):
        assert request == "late-request"
        assert not current_actor._publication_paused
        admitted.set()
        stop.set()

    actor._should_stop = should_stop
    actor.prompt_queue = SimpleNamespace(get=dequeue)
    monkeypatch.setattr(vllm_utils, "add_request", admit)

    def worker():
        try:
            vllm_utils._prefetch_worker(actor)
        except StopWorker:
            pass
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert queue_waiting.wait(3)
        actor.sleep()
        release_queue.set()
        assert not admitted.wait(0.1)
        actor.wake_up()
        assert admitted.wait(3)
    finally:
        stop.set()
        release_queue.set()
        actor._publication_paused = False
        thread.join(3)
    assert not thread.is_alive()
    assert not errors
