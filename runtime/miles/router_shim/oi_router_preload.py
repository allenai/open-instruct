"""Re-exec the sgl-router process with the TCP_USER_TIMEOUT shim preloaded.

Imported at interpreter startup by oi_router_preload.pth in the MILES image. It does
nothing unless this process is `python -m sglang_router.launch_router`, which is how the
pinned MILES launches the Rust router. Set OI_MILES_ROUTER_SHIM=0 to turn it off.
Every Python process in the image imports this module at startup, so keep it import-safe
and free of dependencies beyond the standard library. See docs/miles/operations.md.
"""

import os
import sys
from collections.abc import MutableMapping

SHIM = "/opt/core-rl/lib/no_tcp_user_timeout.so"
ROUTER_MODULE = "sglang_router.launch_router"


def _is_router(argv: list[str]) -> bool:
    """True for `python [interpreter options] -m sglang_router.launch_router ...`."""
    tokens = iter(argv[1:])
    for token in tokens:
        if token == "-m":
            return next(tokens, None) == ROUTER_MODULE
        if token.startswith("-m"):
            return token[2:] == ROUTER_MODULE
        if token in ("-X", "-W"):
            next(tokens, None)
        elif not token.startswith("-") or token.startswith("-c") or token == "--":
            return False
    return False


def preload_argv(argv: list[str], environ: MutableMapping[str, str], shim: str = SHIM) -> list[str] | None:
    """Return the argv to re-exec with, after adding the shim to environ["LD_PRELOAD"], or None."""
    if environ.get("OI_MILES_ROUTER_SHIM") == "0" or not _is_router(argv) or not os.path.exists(shim):
        return None
    preload = environ.get("LD_PRELOAD", "")
    if shim in preload.replace(":", " ").split():
        return None
    environ["LD_PRELOAD"] = f"{shim}:{preload}" if preload else shim
    return [sys.executable, *argv[1:]]


def _main() -> None:
    argv = preload_argv(list(getattr(sys, "orig_argv", ())), os.environ)
    if argv is not None and sys.executable:
        os.execv(sys.executable, argv)


_main()
