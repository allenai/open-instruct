"""Keep the end of a training process bounded. Closing W&B runs in a daemon thread
with a deadline and a delayed stack dump, so a stuck flush cannot hold the driver,
and a completed workflow ends the process with os._exit so stray non-daemon
threads cannot keep it alive after its results are recorded.
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
# Armed before the close; a hang past this point dumps every thread's stack to stderr.
HANG_DUMP_SECONDS = 600


def run_bounded(operation, name, timeout=TRACKING_CLOSE_TIMEOUT, dump_after=HANG_DUMP_SECONDS):
    """Run ``operation`` in a daemon thread for at most ``timeout`` seconds.

    Returns True when it finished. Its exceptions are logged, not raised; on timeout
    the thread is abandoned with a warning and a stack dump of every thread.
    """

    def target():
        try:
            operation()
        except Exception:
            logger.exception("%s failed", name)

    thread = threading.Thread(target=target, name=f"bounded-{name}", daemon=True)
    faulthandler.dump_traceback_later(dump_after, exit=False)
    try:
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            logger.warning("%s did not finish within %ss; continuing shutdown without it", name, timeout)
            faulthandler.dump_traceback(all_threads=True)
            return False
        return True
    finally:
        faulthandler.cancel_dump_traceback_later()


def exit_process(code=0):
    """Flush output and end the process now, skipping interpreter shutdown.

    Only for a run whose result is already recorded: atexit handlers and
    non-daemon threads do not run or get joined.
    """
    loggers = [logging.root, *logging.root.manager.loggerDict.values()]
    for handler in [h for each in loggers for h in getattr(each, "handlers", ())]:
        with contextlib.suppress(Exception):
            handler.flush()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    os._exit(code)
