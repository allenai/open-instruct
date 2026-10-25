"""Keep diagnostic load alive until probes finish, propagating either side's failures."""

import asyncio


async def run_wave(foreground, background, streams, shutdown=None):
    stop = asyncio.Event()
    probes = asyncio.gather(*foreground)
    workers = [asyncio.create_task(background(index, stop)) for index in range(streams)]
    try:
        finished, _ = await asyncio.wait([probes, *workers], return_when=asyncio.FIRST_COMPLETED)
        for worker in workers:
            if worker in finished:
                await worker
                raise RuntimeError("Diagnostic background stream ended before probes completed")
        return await probes
    finally:
        probes.cancel()
        await asyncio.gather(probes, return_exceptions=True)
        stop.set()
        try:
            if workers and shutdown is not None:
                await shutdown(workers)
        finally:
            for task in workers:
                task.cancel()
            results = await asyncio.gather(*workers, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                raise result
