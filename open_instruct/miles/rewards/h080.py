"""Pinned H080 rendering and H066/H086 service identity; safe for CPU planning."""

import hashlib
import re
from pathlib import Path

PROFILE = "open-instruct/if-gate-yes"
PROMPT_SHA256 = "133a5e5448fff28da6b848efefb303fd7bd51dc27d37981e48054db90555cb36"
PROMPT_PATH = Path(__file__).with_name("h080_judge_prompt.txt")
VERDICT = re.compile(r"Verdict: (yes|partial|no)\.?", re.I)
SERVICE = {
    "mode": "managed",
    "backend": "sglang",
    "model": "Qwen/Qwen3-32B",
    "revision": "9216db5781bf21249d130ec9da846c4624c16137",
    "gpus": 1,
    "tensor_parallel_size": 1,
    "max_context_length": 131072,
    "context_extension": "qwen3-yarn-128k",
    "max_concurrent_calls": 16,
    "timeout": 600,
    "chat_template": "qwen3-no-thinking",
    "prepared_dir": "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/judges/qwen3-32b-9216db57/prepared",
}


def parse_verdict(reply):
    """H080 rule_check.parse, including its full-match and whitespace semantics."""
    match = VERDICT.fullmatch(reply.strip())
    return match.group(1).lower() if match else None


def render_conversation(messages):
    """Port of H080 build_requests.render_conversation; strings are raw user text."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list) or not messages:
        raise ValueError("conversation must be nonempty role-labelled turns")
    if messages[-1].get("role") != "user":
        raise ValueError("graded conversation must end with the user request")
    for message in messages:
        if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise ValueError("unsupported message content; do not flatten or omit turns")
    return "\n\n".join(f"{message['role']}:\n{message['content']}" for message in messages)


def sample_messages(metadata):
    # H070's prepared single-turn rows preserve raw user text in metadata.query.
    # Multi-turn callers must preserve the complete turns, never decode sample.prompt.
    messages = metadata.get("judge_conversation", metadata.get("judge_query", metadata.get("query")))
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    render_conversation(messages)  # validate without flattening structured turns
    return messages


def sample_conversation(metadata):
    return render_conversation(sample_messages(metadata))


def render_prompt(conversation, response):
    """Format already role-labelled conversation and checker-scored final text verbatim.

    H086 requests.jsonl stores this rendered conversation, not raw user messages.
    Do not strip whitespace or re-extract thinking here.
    """
    template = PROMPT_PATH.read_bytes()
    actual = hashlib.sha256(template).hexdigest()
    if actual != PROMPT_SHA256:
        raise RuntimeError(f"H080 rubric hash mismatch: {actual}")
    if not isinstance(conversation, str) or not isinstance(response, str):
        raise ValueError("conversation and response must be text")
    return template.decode("utf-8").format(conversation=conversation, response=response)
