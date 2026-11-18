"""Two-turn TITO mechanics probe using a fixed, side-effect-free tool.

Use as custom_agent_function_path in a disposable Core refresh run. Slow rows
wait between model calls so accepted trajectories can span weight publications.
This checks agent transport and training, not terminal-task learning quality.
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

from miles.rollout.agentic.session import openai_session_url
from openai import AsyncOpenAI

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_value",
            "description": "Read the integer stored in the task environment.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }
]


async def run(base_url, prompt, request_kwargs, metadata, **kwargs):
    started = time.monotonic()
    messages = list(prompt)
    value = str(metadata.get("value", 42))
    delay = float(metadata.get("tool_delay_seconds", 0))
    metrics = {"turns": 0, "tool_calls": 0, "total_tool_time": 0.0}
    reward = 0.0
    async with AsyncOpenAI(base_url=openai_session_url(base_url), api_key="EMPTY", timeout=900) as client:
        first = (
            await client.chat.completions.create(model="model", messages=messages, tools=TOOLS, **request_kwargs)
        ).choices[0]
        metrics["turns"] += 1
        messages.append(first.message.model_dump(exclude_none=True))
        if first.finish_reason != "length":
            calls = first.message.tool_calls or []
            metrics["tool_calls"] = len(calls)
            tool_started = time.monotonic()
            await asyncio.sleep(delay)
            metrics["total_tool_time"] = time.monotonic() - tool_started
            if calls:
                for call in calls:
                    if call.function.name != "read_value":
                        raise ValueError(f"Unexpected smoke tool: {call.function.name}")
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": value})
            else:
                messages.append(
                    {"role": "user", "content": f"Environment observation: the value is {value}. Reply with it."}
                )
            final = (
                await client.chat.completions.create(
                    model="model", messages=messages, tools=TOOLS, tool_choice="none", **request_kwargs
                )
            ).choices[0]
            metrics["turns"] += 1
            messages.append(final.message.model_dump(exclude_none=True))
            reward = float(value in (final.message.content or "") and final.finish_reason != "length")
    record = dict(
        task_id=metadata.get("task_id"),
        reward=reward,
        agent_metrics=metrics,
        elapsed_seconds=time.monotonic() - started,
        messages=messages,
    )
    trace = Path(os.environ.get("OI_MILES_AGENT_SMOKE_TRACE_DIR", "/output/agent-smoke"))
    trace.mkdir(parents=True, exist_ok=True)
    (trace / f"{uuid.uuid4().hex}.json").write_text(json.dumps(record) + "\n")
    logger.info("Agent smoke episode: task=%s metrics=%s reward=%s", record["task_id"], metrics, reward)
    return {"reward": reward, "agent_metrics": metrics}
