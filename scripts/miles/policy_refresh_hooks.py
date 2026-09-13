"""Probe-only scheduler instrumentation; never imported by normal training."""

import json
import os
import time
from pathlib import Path

import torch
from sglang.srt.managers import schedule_batch

ROOT = Path(os.environ.get("POLICY_REFRESH_TRACE", "/output/trace"))
CURRENT = None


def install(scheduler_class):
    """Record exact pre-retraction behavior before the stock code clears routes."""
    old_pause = scheduler_class.pause_generation
    old_release = schedule_batch.release_req

    def release(**kwargs):
        req = kwargs["req"]
        if CURRENT is not None and req.output_ids:
            started = time.perf_counter()
            CURRENT.batch_result_processor._maybe_collect_routed_experts(req)
            ROOT.mkdir(parents=True, exist_ok=True)
            stem = f"{req.rid}-{req.retraction_count}"
            routes = req.routed_experts
            if routes is not None:
                torch.save(routes.cpu(), ROOT / f"{stem}.routes.pt")
            record = {
                "rid": req.rid,
                "prompt": list(req.origin_input_ids),
                "output_ids": list(req.output_ids),
                "behavior_logprobs": list(req.output_token_logprobs_val),
                "behavior_token_ids": list(req.output_token_logprobs_idx),
                "routes_shape": list(routes.shape) if routes is not None else None,
                "instrumentation_seconds": time.perf_counter() - started,
            }
            (ROOT / f"{stem}.json").write_text(json.dumps(record))
        return old_release(**kwargs)

    def pause(self, request):
        global CURRENT
        CURRENT = self
        try:
            return old_pause(self, request)
        finally:
            CURRENT = None

    schedule_batch.release_req = release
    scheduler_class.pause_generation = pause
