from __future__ import annotations

import ray

from open_instruct.tool_utils.tool_vllm import Tool, ToolOutput


class ToolProxy(Tool):
    """Generic Tool proxy that forwards calls to a ToolActor.

    Safe to send to other actors; it only holds an actor handle and strings.
    """

    def __init__(self, actor_handle: ray.actor.ActorHandle, start_str: str, end_str: str):
        super().__init__(start_str=start_str, end_str=end_str)
        self._actor = actor_handle

    def __call__(self, prompt: str) -> ToolOutput:
        return ray.get(self._actor.call.remote(prompt))


