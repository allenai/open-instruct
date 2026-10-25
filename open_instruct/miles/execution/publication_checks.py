"""Prepare opt-in weight round trips without exposing reset weights to decoding."""


async def prepare_round_trip(inference, *, refresh, selector, skip_list):
    if refresh:
        # Publication has already resumed decoding. Closing producer admission
        # does not stop existing requests, so pause the engines before poisoning
        # any weights. This preserves requests rather than draining them.
        info = await inference.start_update_weights()
        try:
            for engine in info.rollout_engines:
                await engine.pause_generation()
        except BaseException as error:
            try:
                await inference.abort_update_weights()
            except BaseException as cleanup_error:
                error.add_note(f"Releasing the diagnostic update window also failed: {cleanup_error!r}")
            raise
        # Release the controller lock before check_weights / the next publisher
        # acquire it. This does not resume engine generation. Weights are still
        # valid here; only the normal successful publisher may reopen decoding.
        await inference.end_update_weights(info.snapshot_cell_id_to_hashes)
    await inference.check_weights(action="snapshot", selector=selector)
    await inference.check_weights(action="reset_tensors", selector=selector, skip_list=skip_list)
