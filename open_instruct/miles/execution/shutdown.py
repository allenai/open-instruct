"""Keep the end of a training process bounded.

Closing tracking runs in a daemon thread with a deadline, so a stuck flush cannot
hold the driver. When it finishes, the CLI exits through normal interpreter
shutdown (atexit handlers, Ray's own shutdown), with a watchdog that forces the
exit if that shutdown itself hangs. When a bounded operation was abandoned, its
thread may still hold the process, so the CLI shuts down driver-owned Ray and
then ends the process with os._exit, keeping the success or failure exit code.
"""

import contextlib
import faulthandler
import logging
import os
import sys
import threading

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

TRACKING_CLOSE_TIMEOUT = 120
RAY_SHUTDOWN_TIMEOUT = 60
# Upper bound on normal interpreter shutdown after the result is recorded.
EXIT_WATCHDOG_SECONDS = 300
# Logging and flushing take handler and stream locks that an abandoned or stuck
# thread may hold, so on the forced-exit paths they run in a daemon thread for at
# most this long.
LOCKED_IO_TIMEOUT = 5

# Names of bounded operations whose threads were abandoned in this process.
_abandoned = []
# Set by the CLI entry point only, so importing callers (and tests) never get a timer
# that ends their process.
_exit_watchdog_enabled = False


def run_bounded(operation, name, timeout=TRACKING_CLOSE_TIMEOUT):
    """Run ``operation`` in a daemon thread for at most ``timeout`` seconds.

    Returns True when it returned or raised, False on timeout. Its exceptions are
    logged, not raised: a tracking backend that fails to close must not turn a
    training run whose result is recorded into a failure. On timeout the thread is
    abandoned with a warning and a stack dump of every thread, and ``abandoned()``
    reports it so the CLI can force the process exit.
    """

    def target():
        try:
            operation()
        except Exception:
            logger.exception("%s failed", name)

    thread = threading.Thread(target=target, name=f"bounded-{name}", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        _abandoned.append(name)
        faulthandler.dump_traceback(all_threads=True)
        _bounded(lambda: logger.warning("%s did not finish within %ss; continuing shutdown without it", name, timeout))
        return False
    return True


def _bounded(call, timeout=None):
    """Run ``call`` in a daemon thread and wait at most ``timeout`` seconds for it."""
    thread = threading.Thread(target=call, name="bounded-io", daemon=True)
    thread.start()
    thread.join(LOCKED_IO_TIMEOUT if timeout is None else timeout)


def abandoned():
    return list(_abandoned)


def finish_process(code):
    """End the CLI with ``code`` once the run's result is recorded.

    If a bounded operation was abandoned, shut down driver-owned Ray and exit now:
    the abandoned thread could otherwise keep the process alive. Otherwise return,
    so the caller exits through normal interpreter shutdown; when the entry point
    enabled it, arm a watchdog that exits with the same code if that shutdown hangs.
    """
    if _abandoned:
        names = ", ".join(_abandoned)
        _bounded(lambda: logger.warning("Forcing exit %s; still running: %s", code, names))
        shutdown_ray()
        exit_process(code)
    if _exit_watchdog_enabled:
        arm_exit_watchdog(code)


def enable_exit_watchdog():
    global _exit_watchdog_enabled
    _exit_watchdog_enabled = True


def shutdown_ray():
    """Shut down this process's Ray runtime (or driver connection), bounded."""
    ray = sys.modules.get("ray")
    with contextlib.suppress(Exception):
        if ray is None or not ray.is_initialized():
            return
        run_bounded(ray.shutdown, "Ray shutdown", timeout=RAY_SHUTDOWN_TIMEOUT)


def arm_exit_watchdog(code, timeout=None):
    """Force ``exit_process(code)`` if the process is still alive after ``timeout`` seconds.

    The timer is a daemon thread, so it keeps running while interpreter shutdown
    joins non-daemon threads and runs atexit handlers, and never delays a clean exit.
    """

    timeout = EXIT_WATCHDOG_SECONDS if timeout is None else timeout

    def expire():
        # Only lock-free writes here: a thread stuck in shutdown may hold logging or
        # stream locks.
        with contextlib.suppress(Exception):
            message = f"Process still alive {timeout}s after its result was recorded; forcing exit {code}\n"
            os.write(2, message.encode())
            faulthandler.dump_traceback(all_threads=True)
        os._exit(code)

    timer = threading.Timer(timeout, expire)
    timer.daemon = True
    timer.start()
    return timer


def exit_process(code=0):
    """Flush output and end the process now, skipping interpreter shutdown.

    Only for a run whose result is already recorded: atexit handlers and
    non-daemon threads do not run or get joined. Flushing is bounded, because an
    abandoned thread may hold a handler or stream lock.
    """
    _bounded(_flush_output)
    os._exit(code)


def _flush_output():
    loggers = [logging.root, *logging.root.manager.loggerDict.values()]
    for handler in [h for each in loggers for h in getattr(each, "handlers", ())]:
        with contextlib.suppress(Exception):
            handler.flush()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
