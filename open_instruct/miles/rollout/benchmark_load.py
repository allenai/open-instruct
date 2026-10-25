"""Keep diagnostic load alive until probes finish, propagating either side's failures."""

import asyncio


async def run_wave(foreground, background, streams, shutdown=None):
    stop = asyncio.Event()
    tasks = [asyncio.ensure_future(probe) for probe in foreground]
    probes = asyncio.gather(*tasks)
    workers = [asyncio.create_task(background(index, stop)) for index in range(streams)]
    failure = None
    try:
        finished, _ = await asyncio.wait([probes, *workers], return_when=asyncio.FIRST_COMPLETED)
        for worker in workers:
            if worker in finished:
                await worker
                raise RuntimeError("Diagnostic background stream ended before probes completed")
        return await probes
    except BaseException as error:
        failure = error
        raise
    finally:
        # A gather that has already raised does not cancel its remaining children.
        # Join every foreground request before aborting the private server fleet.
        for task in tasks:
            task.cancel()
        probes.cancel()
        await asyncio.gather(probes, *tasks, return_exceptions=True)
        stop.set()
        cleanup_error = None
        try:
            if workers and shutdown is not None:
                await shutdown(workers)
        except BaseException as error:
            cleanup_error = error
        finally:
            for task in workers:
                task.cancel()
            results = await asyncio.gather(*workers, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                cleanup_error = cleanup_error or result
        if cleanup_error is not None:
            if failure is not None:
                failure.add_note(f"Diagnostic cleanup also failed: {cleanup_error!r}")
            else:
                raise cleanup_error
