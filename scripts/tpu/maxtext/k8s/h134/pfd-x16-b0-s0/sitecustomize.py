# H103 restore-only gcsfs patch (2026-10-23). gcsfs retries each GCS request 6 times with backoff
# min(rand + 2**(k-1), 32) s, about 18 s in all, fixed when GCSFileSystem is defined. A burst over the project's
# GoogleEgressBandwidth quota outlasts that, so a few hosts give up and the slice aborts before step 1.
# This re-wraps GCSFileSystem._request and _get_file_request (every gcsfs read and write, checkpoint saves included)
# with GCSFS_RETRIES attempts (20: ~8.7 min of patience) and up to GCSFS_JITTER_S extra seconds per retry. It changes
# only how long a failed GCS request is retried. Pinned: if gcsfs is not GCSFS_PIN or a wrapped method is missing,
# it logs "GCSFS-RETRY not patched" and exits 3, so the launcher (set -e) stops before any restore.
# Logs: "GCSFS-RETRY retried ..." for each request that needed a retry (attempts, wait), plus a running max.
import os
import sys

if os.environ.get("GCSFS_RETRIES"):
    try:
        import asyncio
        import contextvars
        import random
        import time

        import gcsfs
        from gcsfs import core as _core
        from gcsfs import retry as _retry

        _pin = os.environ.get("GCSFS_PIN", "2026.2.0")
        if gcsfs.__version__ != _pin:
            raise RuntimeError(f"gcsfs {gcsfs.__version__} is not the pinned {_pin}")
        _names = ("_request", "_get_file_request")
        _orig = {}
        for _name in _names:
            _f = getattr(_core.GCSFileSystem, _name, None)
            if _f is None or not hasattr(_f, "__wrapped__"):
                raise RuntimeError(f"GCSFileSystem.{_name} is missing or not retry-wrapped")
            _orig[_name] = _f.__wrapped__
        _n = int(os.environ["GCSFS_RETRIES"])
        _jit = float(os.environ.get("GCSFS_JITTER_S", "4"))
        _host = os.environ.get("HOSTNAME", "?")
        _stat = {"requests": 0, "max_attempts": 1, "max_wait": 0.0}
        _cur = contextvars.ContextVar("gcsfs_retry_cur", default=None)

        _real_sleep = asyncio.sleep

        async def _retry_sleep(t, *a, **kw):  # the only sleep in retry_request: add jitter, count the wait
            t = t + random.random() * _jit
            c = _cur.get()
            if c is not None:
                c[1] += t
            return await _real_sleep(t, *a, **kw)

        class _AsyncioShim:
            sleep = staticmethod(_retry_sleep)

            def __getattr__(self, k):
                return getattr(asyncio, k)

        _retry.asyncio = _AsyncioShim()

        def _install(name, orig):
            async def counted(*a, **kw):
                c = _cur.get()
                if c is not None:
                    c[0] += 1
                return await orig(*a, **kw)

            counted.__name__ = name
            retried = _retry.retry_request(retries=_n)(counted)

            async def outer(*a, **kw):
                c = [0, 0.0]  # attempts, seconds waited
                tok = _cur.set(c)
                t0 = time.monotonic()
                ok = False
                try:
                    r = await retried(*a, **kw)
                    ok = True
                    return r
                finally:
                    _cur.reset(tok)
                    if c[0] > 1:
                        _stat["requests"] += 1
                        _stat["max_attempts"] = max(_stat["max_attempts"], c[0])
                        _stat["max_wait"] = max(_stat["max_wait"], c[1])
                        print(
                            f"GCSFS-RETRY retried host={_host} method={name} attempts={c[0]}/{_n} "
                            f"wait={c[1]:.1f}s elapsed={time.monotonic() - t0:.1f}s ok={ok} "
                            f"| host so far: requests={_stat['requests']} max_attempts={_stat['max_attempts']} "
                            f"max_wait={_stat['max_wait']:.1f}s",
                            file=sys.stderr,
                            flush=True,
                        )

            outer.__wrapped__ = orig
            setattr(_core.GCSFileSystem, name, outer)

        for _name in _names:
            _install(_name, _orig[_name])
        _core.GCSFileSystem.retries = _n
        if os.environ.get("GCSFS_RETRY_QUIET") != "1":
            print(f"GCSFS-RETRY patched host={_host} gcsfs={gcsfs.__version__} methods={list(_names)} retries={_n} "
                  f"jitter<={_jit}s", file=sys.stderr, flush=True)
    except Exception as e:  # pinned: never fall back to stock gcsfs
        print(f"GCSFS-RETRY not patched: {e!r}; exiting before any restore", file=sys.stderr, flush=True)
        sys.stderr.flush()
        os._exit(3)
