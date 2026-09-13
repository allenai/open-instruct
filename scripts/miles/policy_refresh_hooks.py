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
    old_result = scheduler_class.process_batch_result
    old_continue = scheduler_class.continue_generation

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
            req._policy_refresh_cut = len(req.output_ids)
            record = {
                "rid": req.rid,
                "prompt": list(req.origin_input_ids),
                "output_ids": list(req.output_ids),
                "behavior_logprobs": list(req.logprob.output_token_logprobs_val),
                "behavior_token_ids": list(req.logprob.output_token_logprobs_idx),
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

    def resume(self, request):
        now = time.perf_counter()
        for req in self.waiting_queue:
            if hasattr(req, "_policy_refresh_cut"):
                req._policy_refresh_resume = now
        return old_continue(self, request)

    def result(self, batch, output):
        answer = old_result(self, batch, output)
        now = time.perf_counter()
        for req in batch.reqs:
            started = getattr(req, "_policy_refresh_resume", None)
            if started is not None and len(req.output_ids) > req._policy_refresh_cut:
                (ROOT / f"{req.rid}.first-token.json").write_text(
                    json.dumps(
                        {
                            "scheduler_resume_to_first_new_token_seconds": now - started,
                            "cut": req._policy_refresh_cut,
                            "output_tokens": len(req.output_ids),
                        }
                    )
                )
                req._policy_refresh_resume = None
        return answer

    scheduler_class.process_batch_result = result
    scheduler_class.continue_generation = resume
    schedule_batch.release_req = release
    scheduler_class.pause_generation = pause
