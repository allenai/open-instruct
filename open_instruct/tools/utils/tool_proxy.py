from __future__ import annotations

import ray
from typing import List

from open_instruct.tools.utils.tool_classes import Tool, ToolOutput


class ToolProxy(Tool):
    """Generic Tool proxy that forwards calls to a ToolActor.

    Safe to send to other actors; it only holds an actor handle and strings.
    """

    def __init__(self, actor_handle: ray.actor.ActorHandle, start_str: str, end_str: str, name: str | None = None):
        # Get the name from the actor if not provided
        if name is None:
            name = ray.get(actor_handle.get_name.remote())
        super().__init__(name=name, start_str=start_str, end_str=end_str)
        self._actor = actor_handle

    def __call__(self, prompt: str) -> ToolOutput:
        return ray.get(self._actor.call.remote(prompt))

    def is_triggered(self, prompt: str) -> bool:
        return ray.get(self._actor.is_triggered.remote(prompt))

    def get_stop_strings(self) -> List[str]:
        return ray.get(self._actor.get_stop_strings.remote())

    def get_name(self) -> str:
        return ray.get(self._actor.get_name.remote())
