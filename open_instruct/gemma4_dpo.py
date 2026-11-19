"""Gemma-4 DPO support for the HF/DeepSpeed trainer (``dpo_tune_cache.py``): SafeComp post-ARR Phase 4, Gate 3.

Target model: the text-only ``Gemma4ForCausalLM`` extracted from google/gemma-4-12B-it
(``models/gemma-4-12B-it-text``), served by vLLM 0.19.1 as ``gemma4-12b-nothink`` (Google's stock chat template,
thinking OFF). Needs transformers >= 5.5 (5.5.3 = the vLLM 0.19.1 serving pin); see
``agent-safety/scripts/cross_family/gemma4_train_env.sh``.

This module is imported ONLY by the new entry point ``open_instruct/dpo_tune_cache_gemma4.py``. Nothing here runs at
import time of any shared module, so the Qwen3.5 OLMo-core path (``dpo.py``) and stock ``dpo_tune_cache.py`` runs are
unchanged. It has three parts:

1. ``preference_gemma4_tail_tokenize_v1``: preference tokenization in which EVERY labeled assistant turn sits in
   exactly the token context vLLM 0.19.1 renders for that turn at inference.
   * Messages are first normalized the way vLLM 0.19.1's chat preprocessing builds its "conversation" for this
     template (``openai`` content format: text parts; JSON-string tool arguments -> dicts; empty ``tool_calls``
     dropped; only a ``reasoning`` key survives on assistant messages, ``reasoning_content`` never does; tool
     messages keep ``tool_call_id``). Our exports carry ``tool_call_ids`` (a list) on tool messages; it is mapped to
     the ``tool_call_id`` the agent sends, otherwise the template would render ``response:unknown{...}``.
   * Thinking-off student policy (decided 2026-10-26, see the Gate 3 record):
     - ``reasoning_content`` / ``reasoning`` is DROPPED everywhere (history and labels). At inference the history never
       carries reasoning (the agent does not send the ``reasoning`` key vLLM 0.19.1 reads), and the thinking-off
       student should not be trained to emit thought channels.
     - ``content`` on an assistant turn that HAS tool calls is DROPPED. Gemma-4 thinking-off emits a bare tool call on
       >99% of turns; the stock template renders such content AFTER the tool response and then closes the turn, i.e.
       in a different order from how the model generates it, so the turn could not be labeled exactly. Text-only
       assistant turns (no tool call) keep their content.
   * Labeled span of a turn = its generation-order completion, ending with the stop token the model emits:
     ``<|tool_call>call:...<tool_call|><|tool_response>`` for a tool-call turn (``<|tool_response>`` = eos id 50 in
     generation_config) or ``<content><turn|>`` for a text turn.
   * The template is not prefix-stable everywhere: after a user message the generation prompt appends
     ``<|turn>model\\n<|channel>thought\\n<channel|>``, which never appears in history. Turns are therefore grouped into
     SEGMENTS: consecutive turns share a segment while the next turn's exact inference prompt starts with the current
     segment text; otherwise a new segment starts from that turn's full inference prompt. Each side becomes one row
     of packed segments; ``attention_mask`` holds the 1-based SEGMENT ID (0 = padding), and the forward builds
     block-diagonal causal / sliding-window masks and per-segment position ids from it. For SafeComp pairs
     (``n_prefix_messages`` = 2, every turn a tool call) that is exactly two segments: [prompt + turn 1] and
     [full history + turns 2..n].
   * Every labeled span is checked at tokenization time: the token ids before it must equal the tokenization of
     that turn's inference prompt on its own (vLLM tokenizes the rendered prompt with add_special_tokens=False), so
     spans fall on token boundaries and the context is token-identical.
   * A pair whose packed length exceeds ``max_seq_length`` gets all-masked labels on both sides, so
     ``preference_tulu_filter_v1`` drops it (truncation would cut the closing turn).

2. ``chunked_softcap_logps``: per-token log-probs of the labels from final hidden states, computed in sequence chunks
   (training forward: a FIXED number of chunks, ``GEMMA4_LOGPROB_NUM_CHUNKS`` = 32, so every ZeRO-3 rank makes the same
   number of lm_head calls) with Gemma's final-logit softcap (30.0) applied inside each chunk and each chunk recomputed
   in backward
   (``torch.utils.checkpoint``), so the 32k x 262,144 logit tensor is never materialized. ``install_forward_patch``
   adds a ``dpo_labels=`` path to ``Gemma4ForCausalLM.forward`` that returns these per-token log-probs; called with
   ``dpo_labels=None`` it is the original forward. The patch is installed by the Gemma entry point only, inside the
   process; nothing on disk changes.

3. ``gemma4_separate_forward``: the DPO ``forward_fn``. Per example it strips padding, derives segment positions and
   masks from the segment-id ``attention_mask``, and calls the wrapped model once per branch (chosen, rejected).

Added 2026-11-08 (Gemma DPO stage 0; every addition is opt-in and the defaults above are unchanged):

4. ``preference_gemma4_tail_tokenize_v2`` (labeler policy v2, for Gemma-4-31B teacher pools): ``content`` on a
   tool-call turn is KEPT in the history, exactly as the agent sends it back to vLLM; the content-bearing tool-call
   turn itself and the assistant turn right after it are left UNLABELED (the template renders such content after the
   tool response and closes the turn, so neither completion can be written in generation order). Reasoning stays
   dropped. Every labeled context is then the exact vLLM prompt; see ``v2_unlabeled_turns``.
5. ``LengthGroupedMicrostepSampler``: an opt-in training sampler that makes the ``num_processes`` pairs processed
   concurrently in one micro-step similar in cost (sorted within seeded mega-batches of ``group_steps`` micro-steps,
   micro-step groups shuffled), so the slowest rank wastes less time. Same pairs per epoch; different step composition.
6. ``gemma4_hybrid_attention``: an opt-in attention function. Sliding-window layers (head_dim 256) run FlashAttention-2
   varlen (window ``sliding_window - 1`` to the left, causal, segments from the restarted position ids); the global
   layers (head_dim 512, no FlashAttention) keep the SDPA path with the explicit block-diagonal mask.
"""

from __future__ import annotations

import ast
import json
import os
import time
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint
from transformers import Gemma4ForCausalLM, PreTrainedModel  # >= 5.5; only the Gemma entry point imports this module

GEMMA4_TAIL_TRANSFORM = "preference_gemma4_tail_tokenize_v1"
GEMMA4_TAIL_TRANSFORM_V2 = "preference_gemma4_tail_tokenize_v2"
GEMMA4_TAIL_TRANSFORMS = (GEMMA4_TAIL_TRANSFORM, GEMMA4_TAIL_TRANSFORM_V2)
HYBRID_ATTN = "gemma4_hybrid"
MASKED = -100
TURN_HEADER = "<|turn>model\n"
TOOL_RESPONSE_OPEN = "<|tool_response>"
TURN_CLOSE = "<turn|>"
LOGPROB_CHUNK_SIZE = int(os.environ.get("GEMMA4_LOGPROB_CHUNK_SIZE", "2048"))
# The TRAINING forward splits every sequence into a FIXED number of chunks (not a fixed chunk size): under DeepSpeed
# ZeRO-3 each lm_head call all-gathers its (tied, 2 GB) weight, so every rank must make the same number of calls or
# the collectives deadlock (observed 2026-10-26: 4 ranks with different sequence lengths hung at 100% GPU util).
# 32 chunks = chunk length <= 1,024 at 32k tokens.
LOGPROB_NUM_CHUNKS = int(os.environ.get("GEMMA4_LOGPROB_NUM_CHUNKS", "32"))
MEM_LOG = os.environ.get("GEMMA4_MEM_LOG") == "1"  # per-call peak-memory / timing lines (GEMMA4_MEM ...) on every rank


# ---------------------------------------------------------------------------------------------------------------------
# 1. Message normalization (vLLM 0.19.1 conversation for the Gemma-4 template) and segment construction
# ---------------------------------------------------------------------------------------------------------------------
def _text_parts(content: Any) -> list[dict[str, str]]:
    """vLLM ``openai`` content format: None -> [], str -> [{"type": "text", "text": s}] (kept even when empty)."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append({"type": "text", "text": p})
            elif isinstance(p, dict) and p.get("type") == "text":
                parts.append({"type": "text", "text": p.get("text") or ""})
            else:
                raise ValueError(f"unsupported content part for a text-only Gemma student: {p!r}")
        return parts
    raise TypeError(f"unsupported message content type {type(content).__name__}")


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            value = json.loads(s)
        except json.JSONDecodeError:
            value = ast.literal_eval(s)
    if not isinstance(value, list):
        raise TypeError(f"expected a list, got {type(value).__name__}")
    return value


def _tool_calls_vllm(raw: Any) -> list[dict[str, Any]]:
    """OpenAI tool calls -> vLLM ``_postprocess_messages`` form (arguments as a dict; empty/None -> {})."""
    calls = []
    for tc in _as_list(raw):
        fn = dict(tc.get("function") or {})
        args = fn.get("arguments")
        if args:
            if not isinstance(args, (dict, list)):
                args = json.loads(args)
        else:
            args = {}
        call = {"id": tc.get("id"), "type": tc.get("type") or "function", "function": {"name": fn["name"], "arguments": args}}
        calls.append(call)
    return calls


def normalize_conversation_gemma4(messages: Any, keep_call_content: bool = False) -> list[dict[str, Any]]:
    """Row messages -> the conversation vLLM 0.19.1 passes to the Gemma-4 chat template (thinking-off student policy).

    See the module docstring for the policy (reasoning dropped; content dropped on tool-call turns). With
    ``keep_call_content`` (labeler policy v2) the content of a tool-call turn is kept as the agent sends it.
    """
    if isinstance(messages, str):
        messages = json.loads(messages)
    conv: list[dict[str, Any]] = []
    last_call_ids: list[str] = []
    for m in messages:
        role = m["role"]
        if role == "assistant":
            calls = _tool_calls_vllm(m.get("tool_calls"))
            keep = keep_call_content or not calls
            msg: dict[str, Any] = {"role": "assistant", "content": _text_parts(m.get("content")) if keep else []}
            if calls:
                msg["tool_calls"] = calls
                last_call_ids = [c["id"] for c in calls]
            else:
                last_call_ids = []
        elif role == "tool":
            tcid = m.get("tool_call_id")
            if not tcid:
                ids = _as_list(m.get("tool_call_ids"))
                tcid = ids[0] if ids else None
            if not tcid and len(last_call_ids) == 1:
                tcid = last_call_ids[0]
            if not tcid:
                raise ValueError("tool message without a resolvable tool_call_id")
            msg = {"role": "tool", "content": _text_parts(m.get("content")), "tool_call_id": tcid}
        elif role in ("system", "user", "developer"):
            msg = {"role": role, "content": _text_parts(m.get("content"))}
        else:
            raise ValueError(f"unsupported role {role!r}")
        if isinstance(m.get("name"), str) and m.get("name"):
            msg["name"] = m["name"]
        conv.append(msg)
    return conv


def normalize_tools(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None or tools == "":
        return None
    if isinstance(tools, str):
        tools = json.loads(tools)
    if isinstance(tools, dict):
        tools = [tools]
    return tools or None


def _render(tokenizer, conv, tools, add_generation_prompt: bool) -> str:
    out = tokenizer.apply_chat_template(
        conv, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt, continue_final_message=False
    )
    assert isinstance(out, str)
    return out


def turn_prompt_and_completion(tokenizer, conv, tools, i: int) -> tuple[str, str]:
    """(exact inference prompt, generation-order completion) for the assistant message ``conv[i]``."""
    prompt = _render(tokenizer, conv[:i], tools, True)
    before = _render(tokenizer, conv[:i], tools, False)
    with_turn = _render(tokenizer, conv[: i + 1], tools, False)
    if not with_turn.startswith(before):
        raise ValueError(f"history render is not prefix-stable at message {i}")
    completion = with_turn[len(before) :]
    if prompt != before:  # the generation prompt opened a new model turn (after a user / system message)
        if not (prompt.startswith(before) and prompt[len(before) :].startswith(TURN_HEADER)):
            raise ValueError(f"unexpected generation prompt at message {i}: {prompt[len(before):]!r}")
        if not completion.startswith(TURN_HEADER):
            raise ValueError(f"model turn without a header at message {i}: {completion[:40]!r}")
        completion = completion[len(TURN_HEADER) :]
    if conv[i].get("tool_calls"):
        if not (completion.startswith("<|tool_call>") and completion.endswith("<tool_call|>" + TOOL_RESPONSE_OPEN)):
            raise ValueError(f"unexpected tool-call completion at message {i}: {completion[:60]!r}...{completion[-60:]!r}")
    else:
        if not completion.endswith(TURN_CLOSE + "\n"):
            raise ValueError(f"unexpected text completion at message {i}: ...{completion[-60:]!r}")
        completion = completion[:-1]  # the model emits <turn|> (eos 106); the "\n" is history glue
    return prompt, completion


def is_content_bearing_call(msg: dict[str, Any]) -> bool:
    """A normalized assistant message with tool calls AND non-empty text content (any character, whitespace too)."""
    return bool(msg.get("tool_calls")) and any(p.get("text") for p in msg.get("content") or [])


def v2_unlabeled_turns(conv: list[dict[str, Any]], label_from_idx: int) -> set[int]:
    """Labeler policy v2: tail assistant turns left unlabeled = every content-bearing tool-call turn and the assistant
    turn right after one (the next assistant message, even if a user/tool message sits in between)."""
    skip: set[int] = set()
    prev_cb = False
    for i, m in enumerate(conv):
        if m["role"] != "assistant":
            continue
        cb = is_content_bearing_call(m)
        if i >= label_from_idx and (cb or prev_cb):
            skip.add(i)
        prev_cb = cb
    return skip


def build_segments(tokenizer, conv, tools, label_from_idx: int, skip: set[int] | None = None) -> list[dict[str, Any]]:
    """Group the tail assistant turns into exact segments (see the module docstring).

    Turns in ``skip`` (labeler policy v2) are not labeled: they stay in the history of later turns' prompts only.
    Returns a list of {"text": str, "spans": [(char_start, char_end), ...], "prompts": [prompt_str per span]}.
    """
    tail = [i for i in range(label_from_idx, len(conv)) if conv[i]["role"] == "assistant"]
    if not tail:
        raise ValueError("no assistant turn in the tail")
    if skip:
        tail = [i for i in tail if i not in skip]
    segments: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for i in tail:
        prompt, completion = turn_prompt_and_completion(tokenizer, conv, tools, i)
        if cur is not None and prompt.startswith(cur["text"]):
            cur["text"] = prompt + completion
        else:
            cur = {"text": prompt + completion, "spans": [], "prompts": []}
            segments.append(cur)
        cur["spans"].append((len(prompt), len(prompt) + len(completion)))
        cur["prompts"].append(prompt)
    return segments


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def tokenize_segments(tokenizer, segments: list[dict[str, Any]]) -> tuple[list[int], list[int], list[int]]:
    """-> (input_ids, labels, segment_ids) for one side; verifies every span's context is token-identical."""
    input_ids: list[int] = []
    labels: list[int] = []
    seg_ids: list[int] = []
    for s_idx, seg in enumerate(segments, start=1):
        ids = _encode(tokenizer, seg["text"])
        lab = [MASKED] * len(ids)
        for (c0, c1), prompt in zip(seg["spans"], seg["prompts"]):
            p_ids = _encode(tokenizer, prompt)
            if ids[: len(p_ids)] != p_ids:
                raise ValueError("labeled span does not start on a token boundary / prompt tokens differ from vLLM's")
            through = _encode(tokenizer, seg["text"][:c1])
            if ids[: len(through)] != through:
                raise ValueError("labeled span does not end on a token boundary")
            for t in range(len(p_ids), len(through)):
                lab[t] = ids[t]
        input_ids += ids
        labels += lab
        seg_ids += [s_idx] * len(ids)
    return input_ids, labels, seg_ids


def tokenize_side_gemma4(
    messages: Any, tools: Any, tokenizer, label_from_idx: int, policy: str = "v1"
) -> tuple[list[int], list[int], list[int]]:
    """One side -> (input_ids, labels, segment_ids). ``policy`` "v1" (default, Gate 3 stage 1) or "v2" (module
    docstring, part 4). A v2 side whose tail turns are all unlabeled returns empty lists (the pair is then dropped by
    ``preference_tulu_filter_v1``)."""
    if policy not in ("v1", "v2"):
        raise ValueError(f"unknown Gemma-4 labeler policy {policy!r}")
    conv = normalize_conversation_gemma4(messages, keep_call_content=policy == "v2")
    skip = v2_unlabeled_turns(conv, label_from_idx) if policy == "v2" else None
    tail = [i for i in range(label_from_idx, len(conv)) if conv[i]["role"] == "assistant"]
    if skip and tail and all(i in skip for i in tail):
        return [], [], []
    segs = build_segments(tokenizer, conv, normalize_tools(tools), label_from_idx, skip=skip)
    return tokenize_segments(tokenizer, segs)


def preference_gemma4_tail_tokenize_v1(
    row: dict[str, Any],
    tokenizer,
    max_seq_length: int,
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    policy: str = "v1",
):
    """SafeComp tail-labeled preference tokenization for the thinking-off Gemma-4 student (module docstring, part 1).

    Output keys match ``TOKENIZED_PREFERENCE_DATASET_KEYS``; ``*_attention_mask`` holds 1-based segment ids.
    ``policy`` = labeler policy ("v1" default; "v2" is what ``preference_gemma4_tail_tokenize_v2`` passes).
    """
    metadata = row.get("metadata") or {}
    n_prefix = metadata.get("n_prefix_messages")
    chosen, rejected = row[chosen_key], row[rejected_key]
    if not isinstance(n_prefix, int) or not 0 < n_prefix <= min(len(chosen), len(rejected)):
        raise ValueError(f"metadata.n_prefix_messages missing or invalid for tail labeling: {n_prefix!r}")
    tools = row.get("tools")
    out: dict[str, Any] = {}
    over = False
    for branch, msgs in (("chosen", chosen), ("rejected", rejected)):
        ids, lab, seg = tokenize_side_gemma4(msgs, tools, tokenizer, n_prefix, policy=policy)
        over = over or len(ids) > max_seq_length
        out[f"{branch}_input_ids"] = ids[:max_seq_length]
        out[f"{branch}_labels"] = lab[:max_seq_length]
        out[f"{branch}_attention_mask"] = seg[:max_seq_length]
    if over:
        for branch in ("chosen", "rejected"):
            out[f"{branch}_labels"] = [MASKED] * len(out[f"{branch}_labels"])
    return out


def preference_gemma4_tail_tokenize_v2(
    row: dict[str, Any],
    tokenizer,
    max_seq_length: int,
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
):
    """Labeler policy v2 (module docstring, part 4): tool-call content kept in history; the content-bearing tool-call
    turn and the turn right after it unlabeled; reasoning dropped. Same output keys as v1."""
    return preference_gemma4_tail_tokenize_v1(
        row, tokenizer, max_seq_length, chosen_key=chosen_key, rejected_key=rejected_key, policy="v2"
    )


def register_transform() -> None:
    """Register the Gemma transforms (v1 and v2) in open-instruct's registry (in-process only; called by the Gemma
    entry point).

    Imported here, not at module top, so the tokenization half of this module also runs in the vLLM 0.19.1 env used
    by the render check (agent-safety/scripts/cross_family/gemma4_render_check.py), which has no open-instruct deps.
    """
    from open_instruct import dataset_transformation as dt  # noqa: PLC0415

    dt.TRANSFORM_FNS[GEMMA4_TAIL_TRANSFORM] = (preference_gemma4_tail_tokenize_v1, "map")
    dt._SFT_TOKENIZE_FNS.add(GEMMA4_TAIL_TRANSFORM)  # consumes the tools column (strip it after tokenization)
    dt.TRANSFORM_FNS[GEMMA4_TAIL_TRANSFORM_V2] = (preference_gemma4_tail_tokenize_v2, "map")
    dt._SFT_TOKENIZE_FNS.add(GEMMA4_TAIL_TRANSFORM_V2)


# ---------------------------------------------------------------------------------------------------------------------
# 2. Chunked, softcapped per-token log-probs and the Gemma4ForCausalLM forward patch
# ---------------------------------------------------------------------------------------------------------------------
def _chunk_logps(hidden: torch.Tensor, weight_module: torch.nn.Module, targets: torch.Tensor, softcap: float | None):
    logits = weight_module(hidden).float()
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    return torch.gather(logits.log_softmax(-1), -1, targets.unsqueeze(-1)).squeeze(-1)


def chunked_softcap_logps(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Module,
    softcap: float | None,
    chunk_size: int | None = None,
    num_chunks: int | None = None,
) -> torch.Tensor:
    """Per-token log-probs, open-instruct convention: out[:, i] = log p(labels[:, i+1] | <= i); out[:, -1] is filler.

    ``hidden`` is (B, L, H) final hidden states (after the final norm). Logits are computed by ``lm_head`` in its
    dtype, upcast to fp32, softcapped (tanh(x / c) * c) and log-softmaxed one chunk of positions at a time. Under
    grad every chunk is checkpointed, so peak extra memory is ~2-3 x chunk_size x vocab x 4 bytes.
    """
    B, L, _ = hidden.shape
    targets = torch.zeros_like(labels)
    targets[:, :-1] = labels[:, 1:]
    targets = targets.clamp(min=0)
    if num_chunks is not None:
        # exactly num_chunks pieces (identical lm_head call count on every rank; pieces may be empty if L < num_chunks)
        pieces = list(zip(torch.tensor_split(hidden, num_chunks, dim=1), torch.tensor_split(targets, num_chunks, dim=1)))
    else:
        chunk_size = chunk_size or LOGPROB_CHUNK_SIZE  # module global, read at call time
        pieces = [(hidden[:, s : s + chunk_size], targets[:, s : s + chunk_size]) for s in range(0, L, chunk_size)]
    outs = []
    for h, t in pieces:
        if torch.is_grad_enabled() and h.requires_grad:
            outs.append(checkpoint(_chunk_logps, h, lm_head, t, softcap, use_reentrant=False))
        else:
            outs.append(_chunk_logps(h, lm_head, t, softcap))
    return torch.cat(outs, dim=1)


_ORIG_FORWARD = None


def _gemma4_forward_with_dpo_labels(self, *args, dpo_labels: torch.Tensor | None = None, **kwargs):
    if dpo_labels is None:
        return _ORIG_FORWARD(self, *args, **kwargs)
    if args:
        raise TypeError("pass inputs by keyword on the dpo_labels path")
    kwargs.pop("labels", None)
    kwargs["use_cache"] = False
    outputs = self.model(
        input_ids=kwargs.pop("input_ids"),
        attention_mask=kwargs.pop("attention_mask", None),
        position_ids=kwargs.pop("position_ids", None),
        **kwargs,
    )
    return chunked_softcap_logps(
        outputs.last_hidden_state,
        dpo_labels,
        self.lm_head,
        self.config.final_logit_softcapping,
        num_chunks=LOGPROB_NUM_CHUNKS,
    )


def install_forward_patch() -> None:
    """Add the ``dpo_labels=`` path to ``Gemma4ForCausalLM.forward`` (idempotent; this process only)."""
    global _ORIG_FORWARD
    if getattr(Gemma4ForCausalLM.forward, "_gemma4_dpo_patched", False):
        return
    _ORIG_FORWARD = Gemma4ForCausalLM.forward
    _gemma4_forward_with_dpo_labels._gemma4_dpo_patched = True
    Gemma4ForCausalLM.forward = _gemma4_forward_with_dpo_labels


def load_persistent_buffers(model, checkpoint_dir: str) -> dict[str, Any]:
    """Copy every persistent buffer stored in the checkpoint into ``model`` (in place, on every rank).

    Needed under DeepSpeed ZeRO-3: transformers 5.5.3's ``_load_state_dict_into_zero3_model`` calls
    ``module._load_from_state_dict`` only for modules that own PARAMETERS in the state dict. Gemma-4's per-layer
    ``layer_scalar`` (0.0045-0.887) is a buffer on ``Gemma4TextDecoderLayer``, which owns no direct parameters, so it
    is never loaded and stays at its init value 1.0 (observed 2026-10-26: mean labeled log-prob -47 instead of ~-0.3).
    Buffers are not partitioned by ZeRO-3, so a direct copy is correct. Returns a small report.
    """
    from safetensors import safe_open  # noqa: PLC0415

    idx_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        weight_map = json.load(open(idx_path))["weight_map"]
    else:
        weight_map = {k: "model.safetensors" for k in safe_open(os.path.join(checkpoint_dir, "model.safetensors"), "pt").keys()}
    persistent = {}
    for mod_name, mod in model.named_modules():
        for b_name, buf in mod._buffers.items():
            if buf is None or b_name in mod._non_persistent_buffers_set:
                continue
            persistent[f"{mod_name}.{b_name}" if mod_name else b_name] = buf
    handles, fixed, max_change = {}, 0, 0.0
    for key, buf in persistent.items():
        if key not in weight_map:
            continue
        f = weight_map[key]
        if f not in handles:
            handles[f] = safe_open(os.path.join(checkpoint_dir, f), "pt")
        src = handles[f].get_tensor(key).to(device=buf.device, dtype=buf.dtype)
        change = (buf.float() - src.float()).abs().max().item()
        if change > 0:
            fixed += 1
            max_change = max(max_change, change)
        with torch.no_grad():
            buf.copy_(src)
    return {"persistent_buffers": len(persistent), "in_checkpoint": sum(k in weight_map for k in persistent),
            "changed": fixed, "max_abs_change": max_change}


def recompute_rope_buffers(model) -> dict[str, Any]:
    """Recompute Gemma-4's RoPE inverse-frequency buffers in fp32 (in place, on every rank) and report what was there.

    They are non-persistent buffers built in ``__init__``; under DeepSpeed ZeRO-3 ``zero.Init`` the model is
    constructed inside DeepSpeed's tensor-creation hooks, so these buffers may not hold the fp32 values a plain load
    gives. This rebuilds them with the module's own rope init functions, outside any init context, exactly as
    ``Gemma4PreTrainedModel._init_weights`` does, and stores them as fp32.
    """
    report = {}
    for m in model.modules():
        if type(m).__name__ != "Gemma4TextRotaryEmbedding":
            continue
        for layer_type, fn in m.rope_init_fns.items():
            kwargs = {"device": None, "layer_type": layer_type}
            if layer_type == "full_attention" and m.rope_type[layer_type] == "proportional":
                kwargs["head_dim_key"] = "global_head_dim"
            inv, _ = fn(m.config, **kwargs)
            for name in (f"{layer_type}_inv_freq", f"{layer_type}_original_inv_freq"):
                old = m._buffers[name]
                report[name] = {"dtype_before": str(old.dtype),
                                "max_abs_change": (old.detach().float().cpu() - inv.float().cpu()).abs().max().item()}
                m._buffers[name] = inv.to(device=old.device, dtype=torch.float32)
    return report


# ---------------------------------------------------------------------------------------------------------------------
# 3. Masks / positions from segment ids, and the DPO forward_fn
# ---------------------------------------------------------------------------------------------------------------------
def segment_positions(seg: torch.Tensor) -> torch.Tensor:
    """(L,) 1-based segment ids (contiguous runs) -> (L,) positions restarting at 0 in every segment."""
    idx = torch.arange(seg.numel(), device=seg.device)
    starts = torch.ones_like(seg, dtype=torch.bool)
    starts[1:] = seg[1:] != seg[:-1]
    start_idx = torch.where(starts, idx, torch.zeros_like(idx))
    start_idx = torch.cummax(start_idx, dim=0).values
    return idx - start_idx


def gemma4_masks(seg: torch.Tensor, sliding_window: int, attn_implementation: str = "sdpa", dtype=None):
    """Block-diagonal causal (full) and sliding-window masks, (1, 1, L, L), keyed by Gemma-4 layer type.

    Same predicates as transformers' masking_utils: causal kv <= q; sliding additionally kv > q - window; packed
    segments additionally same segment. ``sdpa`` gets bool masks (True = attend); ``eager`` gets additive float masks
    (0 / dtype-min), the forms transformers itself builds for those backends. ``gemma4_hybrid`` gets the bool full mask
    and None for the sliding layers (they run FlashAttention varlen on the position ids; no L x L mask is built).
    """
    L = seg.numel()
    q = torch.arange(L, device=seg.device)
    same = seg[:, None] == seg[None, :]
    full = same & (q[None, :] <= q[:, None])
    if attn_implementation == HYBRID_ATTN:  # sliding layers: FlashAttention varlen from position ids, no dense mask
        return {"full_attention": full[None, None], "sliding_attention": None}
    sliding = full & (q[None, :] > (q[:, None] - sliding_window))
    masks = {"full_attention": full[None, None], "sliding_attention": sliding[None, None]}
    if attn_implementation == "sdpa":
        return masks
    if attn_implementation == "eager":
        dtype = dtype or torch.float32
        neg = torch.finfo(dtype).min
        return {k: torch.zeros(v.shape, dtype=dtype, device=v.device).masked_fill(~v, neg) for k, v in masks.items()}
    raise ValueError(f"gemma4 DPO forward supports sdpa / eager attention, not {attn_implementation!r}")


def _unwrap_config(model):
    """HF config of the (possibly DeepSpeed / accelerate / compile -wrapped) model."""
    m = model
    while not isinstance(m, PreTrainedModel):
        if hasattr(m, "module"):
            m = m.module
        elif hasattr(m, "_orig_mod"):
            m = m._orig_mod
        else:
            raise TypeError(f"cannot find the HF model inside {type(model).__name__}")
    return m.config


def batch_logps(per_token_logps: torch.Tensor, labels: torch.Tensor, average_log_prob: bool) -> torch.Tensor:
    """Same aggregation as ``dpo_utils._get_batch_logps`` (kept local so this module imports without OLMo-core;
    equality is unit-tested)."""
    per_token_logps = per_token_logps[:, :-1]
    loss_mask = labels[:, 1:] != MASKED
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
    return (per_token_logps * loss_mask).sum(-1)


def gemma4_branch_logps(model, input_ids, seg_ids, labels, average_log_prob: bool) -> torch.Tensor:
    cfg = _unwrap_config(model)
    outs = []
    for b in range(input_ids.shape[0]):
        keep = seg_ids[b] > 0
        ids_b = input_ids[b][keep][None]
        seg_b = seg_ids[b][keep]
        lab_b = labels[b][keep][None]
        pos_b = segment_positions(seg_b)[None]
        masks = gemma4_masks(seg_b, cfg.sliding_window, cfg._attn_implementation, cfg.dtype)
        tok_logps = model(input_ids=ids_b, attention_mask=masks, position_ids=pos_b, dpo_labels=lab_b)
        outs.append(batch_logps(tok_logps, lab_b, average_log_prob))
        del masks
    return torch.cat(outs)


_MEM_STATE = {"calls": 0, "t": None, "lens": None}


def _mem_log(batch) -> None:
    """One line per forward_fn call: peak memory and wall time SINCE THE PREVIOUS CALL (= the previous micro-step's
    forward + backward + optimizer step during training; the previous batch's forward during reference caching)."""
    if not (MEM_LOG and torch.cuda.is_available()):
        return
    now = time.time()
    st = _MEM_STATE
    lens = [int((batch[f"{b}_attention_mask"] > 0).sum()) for b in ("chosen", "rejected")]
    if st["t"] is not None:
        print(
            f"GEMMA4_MEM rank={os.environ.get('RANK', '0')} call={st['calls']} prev_lens={st['lens']} "
            f"prev_peak_alloc_gib={torch.cuda.max_memory_allocated() / 2**30:.2f} "
            f"prev_peak_reserved_gib={torch.cuda.max_memory_reserved() / 2**30:.2f} "
            f"dt_s={now - st['t']:.2f} grad={torch.is_grad_enabled()}",
            flush=True,
        )
    torch.cuda.reset_peak_memory_stats()
    st.update(calls=st["calls"] + 1, t=now, lens=lens)


def gemma4_separate_forward(model, batch, average_log_prob: bool = False, output_router_logits: bool = False):
    """DPO forward_fn: chosen then rejected, each one call per example (padding stripped, segments packed)."""
    del output_router_logits
    _mem_log(batch)
    res = []
    for branch in ("chosen", "rejected"):
        res.append(
            gemma4_branch_logps(
                model,
                batch[f"{branch}_input_ids"],
                batch[f"{branch}_attention_mask"],
                batch[f"{branch}_labels"],
                average_log_prob,
            )
        )
    return res[0], res[1], None


# ---------------------------------------------------------------------------------------------------------------------
# 5. Length-grouped micro-steps across ranks (opt-in; added 2026-11-08)
# ---------------------------------------------------------------------------------------------------------------------
# Quadratic term of the per-side cost model, in tokens: g(L) ~ L + L^2 / COST_L0. From the 10/26 smoke fit on
# 8 x A100 (g(L) = -7.1 + 2.90e-3 L + 4.97e-8 L^2 s): 2.90e-3 / 4.97e-8 = 58,350. Only the ranking of pair costs
# matters to the sampler, so the constant term is dropped.
COST_L0 = 58_350.0


def pair_cost(len_chosen: int, len_rejected: int, l0: float = COST_L0) -> float:
    """Relative cost of one pair (both sides' packed lengths; chosen and rejected are separate forwards)."""
    return sum(L + L * L / l0 for L in (len_chosen, len_rejected))


def length_grouped_order(
    costs: list[float], num_replicas: int, per_device_batch: int, group_steps: int, seed: int, epoch: int
) -> list[int]:
    """One epoch's index order for accelerate's ``BatchSamplerShard`` (split_batches=False), under which micro-step g on
    rank r processes the ``per_device_batch`` consecutive indices at positions (g * num_replicas + r) * B ... + B - 1.

    1. Seeded permutation of all indices (numpy PCG64 seeded by (seed, epoch): identical on every rank).
    2. Cut into mega-batches of ``group_steps`` micro-steps (``group_steps * num_replicas * B`` pairs; the last one may
       be short). Within each, sort by cost (descending) and cut into micro-step groups of ``num_replicas * B`` pairs,
       so the pairs that run concurrently on different ranks have similar cost. Within a group the pairs are dealt to
       ranks largest-first onto the least-loaded rank (B = 1: one pair per rank).
    3. Shuffle the micro-step groups within the mega-batch (so long groups are not always first), then concatenate.
    ``group_steps`` = 1 keeps the random order (no balancing within a micro-step is possible beyond B > 1 dealing).
    """
    import numpy as np  # noqa: PLC0415  (only the sampler needs it)

    n = len(costs)
    rng = np.random.default_rng([int(seed), int(epoch)])
    perm = rng.permutation(n).tolist()
    step_size = num_replicas * per_device_batch
    mega = max(1, group_steps) * step_size
    order: list[int] = []
    for m0 in range(0, n, mega):
        chunk = perm[m0 : m0 + mega]
        if group_steps > 1:
            chunk = sorted(chunk, key=lambda i: (-costs[i], i))
        groups = [chunk[g : g + step_size] for g in range(0, len(chunk), step_size)]
        if per_device_batch > 1:
            dealt = []
            for grp in groups:
                bins: list[list[int]] = [[] for _ in range(num_replicas)]
                loads = [0.0] * num_replicas
                for i in sorted(grp, key=lambda i: (-costs[i], i)):
                    r = min((r for r in range(num_replicas) if len(bins[r]) < per_device_batch), key=lambda r: (loads[r], r))
                    bins[r].append(i)
                    loads[r] += costs[i]
                dealt.append([i for b in bins for i in b])
            groups = dealt
        if group_steps > 1:
            gperm = rng.permutation(len(groups)).tolist()
            groups = [groups[g] for g in gperm]
        for grp in groups:
            order.extend(grp)
    return order


class LengthGroupedMicrostepSampler(torch.utils.data.Sampler):
    """Training sampler for ``dpo_tune_cache`` (opt-in via the Gemma entry point): ``length_grouped_order`` per epoch.

    Every rank builds the same order from (seed, epoch); accelerate's ``BatchSamplerShard`` then gives micro-step g of
    rank r the batch at position g * num_replicas + r. ``set_epoch`` is called by accelerate's ``DataLoaderShard``.
    """

    def __init__(self, costs: list[float], num_replicas: int, per_device_batch: int = 1, group_steps: int = 8,
                 seed: int = 0):
        self.costs = [float(c) for c in costs]
        self.num_replicas = int(num_replicas)
        self.per_device_batch = int(per_device_batch)
        self.group_steps = int(group_steps)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        return iter(length_grouped_order(self.costs, self.num_replicas, self.per_device_batch, self.group_steps,
                                         self.seed, self.epoch))

    def __len__(self) -> int:
        return len(self.costs)


def simulate_wall(order: list[int], costs: list[float], num_replicas: int, per_device_batch: int = 1) -> float:
    """Sum over micro-steps of the slowest rank's cost (the wall-time model of a synchronous ZeRO-3 micro-step)."""
    step = num_replicas * per_device_batch
    total = 0.0
    for g in range(0, len(order), step):
        grp = order[g : g + step]
        loads = [sum(costs[i] for i in grp[r * per_device_batch : (r + 1) * per_device_batch]) for r in range(num_replicas)]
        total += max(loads)
    return total


# ---------------------------------------------------------------------------------------------------------------------
# 6. Hybrid attention: FlashAttention-2 varlen on sliding layers, SDPA (explicit mask) on global layers (opt-in)
# ---------------------------------------------------------------------------------------------------------------------
_FLASH_VARLEN = None  # resolved on first use (flash_attn.flash_attn_varlen_func); tests inject a reference


def _flash_varlen():
    global _FLASH_VARLEN
    if _FLASH_VARLEN is None:
        from flash_attn import flash_attn_varlen_func  # noqa: PLC0415  (GPU-only optional dependency)

        _FLASH_VARLEN = flash_attn_varlen_func
    return _FLASH_VARLEN


def cu_seqlens_from_positions(position_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(1, L) positions restarting at 0 per segment -> (int32 cumulative segment lengths [0, ..., L], max length)."""
    pos = position_ids.reshape(-1)
    L = pos.numel()
    starts = torch.nonzero(pos == 0, as_tuple=False).reshape(-1)
    if starts.numel() == 0 or int(starts[0]) != 0:
        raise ValueError("hybrid attention needs per-segment position ids starting at 0")
    cu = torch.cat([starts, starts.new_tensor([L])]).to(torch.int32)
    return cu, int((cu[1:] - cu[:-1]).max())


def gemma4_hybrid_attention(module, query, key, value, attention_mask, dropout: float = 0.0, scaling=None,
                            sliding_window=None, **kwargs):
    """transformers attention-interface function (registered as ``gemma4_hybrid``).

    Sliding layers (``module.is_sliding``): FlashAttention-2 varlen over the packed segments (cu_seqlens from the
    restarted ``position_ids``), causal, window_size = (sliding_window - 1, 0), i.e. key j visible to query i iff
    i - sliding_window < j <= i within the same segment: the same predicate as ``gemma4_masks``' sliding mask.
    ``attention_mask`` must be None there. Global layers: transformers' SDPA function with the explicit block-diagonal
    mask, exactly as the default training path. Input (B=1, H, L, D); output (1, L, H, D) like transformers' functions.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: PLC0415

    if not getattr(module, "is_sliding", False):
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      sliding_window=sliding_window, **kwargs)
    if attention_mask is not None:
        raise ValueError("gemma4_hybrid: sliding layers take no explicit mask (pass masks from gemma4_masks)")
    if query.shape[0] != 1:
        raise ValueError("gemma4_hybrid: one packed sequence per forward (batch size 1)")
    position_ids = kwargs.get("position_ids")
    L = query.shape[2]
    if position_ids is None:
        cu, max_len = torch.tensor([0, L], dtype=torch.int32, device=query.device), L
    else:
        cu, max_len = cu_seqlens_from_positions(position_ids)
        cu = cu.to(query.device)
    q = query[0].transpose(0, 1).contiguous()  # (L, Hq, D)
    k = key[0].transpose(0, 1).contiguous()  # (L, Hkv, D)
    v = value[0].transpose(0, 1).contiguous()
    window = (int(sliding_window) - 1, 0) if sliding_window else (-1, -1)
    out = _flash_varlen()(q, k, v, cu, cu, max_len, max_len, dropout_p=dropout, softmax_scale=scaling, causal=True,
                          window_size=window)
    return out.unsqueeze(0), None


def register_hybrid_attention() -> None:
    """Register ``gemma4_hybrid`` with transformers' attention interface (idempotent; this process only)."""
    from transformers import AttentionInterface  # noqa: PLC0415

    AttentionInterface.register(HYBRID_ATTN, gemma4_hybrid_attention)


def reference_varlen_attention(q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=0.0, softmax_scale=None, causal=False,
                               window_size=(-1, -1)):
    """Pure-torch reference of ``flash_attn_varlen_func``'s documented semantics (CPU tests only; equal seqlens).

    flash_attn 2.8.3 docstring: query i attends keys in [i + seqlen_k - seqlen_q - window_size[0],
    i + seqlen_k - seqlen_q + window_size[1]] inclusive (-1 = unbounded), causal aligned bottom-right; GQA by
    head groups (query head h uses KV head h // (Hq / Hkv)). q: (total, Hq, D), k/v: (total, Hkv, D).
    """
    assert dropout_p == 0.0
    Hq, Hkv, D = q.shape[1], k.shape[1], q.shape[2]
    scale = softmax_scale if softmax_scale is not None else D ** -0.5
    outs = []
    for s in range(cu_q.numel() - 1):
        a, b = int(cu_q[s]), int(cu_q[s + 1])
        qs, ks, vs = q[a:b], k[a:b], v[a:b]
        n = b - a
        ks = ks.repeat_interleave(Hq // Hkv, dim=1)
        vs = vs.repeat_interleave(Hq // Hkv, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qs.float(), ks.float()) * scale
        i = torch.arange(n)[:, None]
        j = torch.arange(n)[None, :]
        allowed = torch.ones(n, n, dtype=torch.bool)
        if causal:
            allowed &= j <= i
        if window_size[0] >= 0:
            allowed &= j >= i - window_size[0]
        if window_size[1] >= 0:
            allowed &= j <= i + window_size[1]
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), vs.float()).to(q.dtype))
    return torch.cat(outs, dim=0)
