import logging
import time

from open_instruct.utils import MainLoopStallWatchdog


def test_watchdog_warns_and_dumps_on_stall(caplog):
    watchdog = MainLoopStallWatchdog(dump_after_s=0.2, poll_interval_s=0.05)
    watchdog.beat("step 1: loop top")
    with caplog.at_level(logging.WARNING, logger="open_instruct.utils"):
        watchdog.start()
        try:
            time.sleep(0.6)  # No beats: the watchdog must fire.
        finally:
            watchdog.stop()
    assert "has not advanced past 'step 1: loop top'" in caplog.text


def test_watchdog_quiet_while_beating(caplog):
    watchdog = MainLoopStallWatchdog(dump_after_s=0.3, poll_interval_s=0.05)
    with caplog.at_level(logging.WARNING, logger="open_instruct.utils"):
        watchdog.start()
        try:
            for _ in range(8):
                watchdog.beat("progressing")
                time.sleep(0.05)
        finally:
            watchdog.stop()
    assert "has not advanced" not in caplog.text


def test_watchdog_disabled_with_nonpositive_threshold():
    watchdog = MainLoopStallWatchdog(dump_after_s=0)
    watchdog.start()
    assert watchdog._thread is None  # start() is a no-op when disabled
    watchdog.stop()
