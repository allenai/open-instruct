"""Chosen-side tool-call template tokens for the SafeComp DPO ablations (2026-11-03; NOT upstream).

Collapse in the SafeComp DPO runs is the loss of probability of one or two near-deterministic tool-call template
tokens (agent-safety/docs/experiments/2026-10-28-collapse-diagnosis.md). This module finds those template positions in
a chosen-side label tensor so the trainer can (a) log the policy probability of template tokens every step and
(b) apply a confidence-gated NLL on them (``--template_nll_coef`` / ``--template_nll_tau``).

Template = the token classes of the collapse diagnosis (§1, §6.1), as exact strings of the Qwen3.5 chat/tool-call
template. Their token ids are NOT hard-coded: ``build_template_patterns`` tokenizes the strings with the run's own
tokenizer (and refuses if a string does not tokenize to the expected number of pieces). For Qwen3.5-9B
(hamishivi/Qwen3.5-9B) they are:

  think_close  '\\n</think>\\n\\n'                                     [198, 248069, 271]
  tc_open      '<tool_call>\\n<function=bash>\\n<parameter=command>\\n'   14 tokens
  tc_close     '\\n</parameter>\\n</function>\\n</tool_call>'            10 tokens
  end          '<|im_end|>\\n'                                         [248046, 198]  (the '\\n' is labeled too)
  content_sep  '\\n\\n' directly before '<tool_call>' (only the '\\n\\n'; skipped where it is think_close's '\\n\\n')

Matching is on token-id sequences inside the LABELED tokens only (positions whose label is -100 never match), so only
positions in the trained span count. Each full string is matched first; then, where nothing matched, the same string
with a leading and/or trailing WHITESPACE token dropped (such a token can merge with the adjacent free text, e.g. the
command-ending '\n' tokenized together with the command's last characters, or fall outside the label span, e.g. the
'\n' after a final '<|im_end|>'); a trimmed match keeps the position types (offsets) of the full string. Non-whitespace
end tokens are never dropped, so e.g. 'end' never degrades to a bare '\n'. This reproduces the collapse diagnosis's token classes
(``collapse_diag_20261028/stage1/data_comp.py:classify_turn``) on the v2-1,738 training pairs. Positions are reported in the trainer's shifted frame: index i of the returned
tensors refers to the target token ``labels[:, i + 1]``, i.e. to ``per_token_logps[:, i]`` (DPOLMHead convention:
output[i] = log p(labels[i+1] | x_<=i)), the same frame ``_get_batch_logps`` uses.

Only torch is imported here, so the matcher can be checked against rendered rows outside the training venv.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# (class, string), in priority order: a position already claimed by an earlier class is not re-assigned.
TEMPLATE_SPANS: tuple[tuple[str, str], ...] = (
    ("think_close", "\n</think>\n\n"),
    ("tc_open", "<tool_call>\n<function=bash>\n<parameter=command>\n"),
    ("tc_close", "\n</parameter>\n</function>\n</tool_call>"),
    ("end", "<|im_end|>\n"),
)
CONTENT_SEP = ("content_sep", "\n\n", "<tool_call>")
# Expected piece counts (Qwen3.5 tokenizer); a different tokenizer that splits these strings differently is refused.
EXPECTED_LENGTHS = {"think_close": 3, "tc_open": 14, "tc_close": 10, "end": 2}
# Position types whose next-token alternatives are legitimate decisions, not template (collapse diagnosis §6.2 excludes
# the first two from its call-markup failure): '\n' that ends the reasoning, '<tool_call>' (call vs more prose), the
# '\n' that ends the command, and the prose->call '\n\n'. They stay in the template mask (gated NLL, p_mean, p_min,
# per-type metrics) but are left out of the "deterministic" minimum ``det_p_min``.
DECISION_TYPES = ("think_close_0", "tc_open_0", "tc_close_0", "content_sep_0")


@dataclass(frozen=True)
class TemplatePatterns:
    """Token-id patterns of the chosen-side template (see module docstring)."""

    spans: tuple[tuple[str, tuple[int, ...]], ...]
    content_sep: tuple[int, int]
    type_names: tuple[str, ...]
    type_tokens: tuple[str, ...]
    whitespace_ids: frozenset[int] = frozenset()

    def match_order(self) -> list[tuple[tuple[int, ...], int]]:
        """(token ids, type id of the first token) in matching priority: every full string; then every string with
        a leading whitespace token dropped; then with a trailing whitespace token dropped; then with both dropped."""
        bases, b = [], 0
        for _cls, ids in self.spans:
            bases.append(b)
            b += len(ids)
        ws = self.whitespace_ids
        full = list(zip((ids for _c, ids in self.spans), bases))
        out = list(full)
        out += [(ids[1:], base + 1) for ids, base in full if len(ids) >= 2 and ids[0] in ws]
        out += [(ids[:-1], base) for ids, base in full if len(ids) >= 2 and ids[-1] in ws]
        out += [(ids[1:-1], base + 1) for ids, base in full if len(ids) >= 3 and ids[0] in ws and ids[-1] in ws]
        return out

    @property
    def num_types(self) -> int:
        return len(self.type_names)

    def deterministic_type_mask(self, device: torch.device | None = None) -> torch.Tensor:
        """(num_types,) bool: True for the deterministic (non-decision) position types."""
        return torch.tensor([n not in DECISION_TYPES for n in self.type_names], dtype=torch.bool, device=device)


def build_template_patterns(tokenizer) -> TemplatePatterns:
    """Tokenize the template strings with ``tokenizer`` (no special-token insertion)."""
    spans, names, toks = [], [], []
    for cls, s in TEMPLATE_SPANS:
        ids = tuple(tokenizer.encode(s, add_special_tokens=False))
        want = EXPECTED_LENGTHS[cls]
        if len(ids) != want:
            raise ValueError(f"template string {cls}={s!r} tokenizes to {len(ids)} pieces {ids}, expected {want}")
        spans.append((cls, ids))
        for j, t in enumerate(ids):
            names.append(f"{cls}_{j}")
            toks.append(tokenizer.decode([t]))
    sep = tuple(tokenizer.encode(CONTENT_SEP[1], add_special_tokens=False))
    tc = tuple(tokenizer.encode(CONTENT_SEP[2], add_special_tokens=False))
    if len(sep) != 1 or len(tc) != 1:
        raise ValueError(f"content_sep pieces {sep} / {tc}: expected one token each")
    names.append("content_sep_0")
    toks.append(tokenizer.decode([sep[0]]))
    ws = frozenset(t for _c, ids in spans for t in ids if tokenizer.decode([t]).strip() == "")
    return TemplatePatterns(
        spans=tuple(spans),
        content_sep=(sep[0], tc[0]),
        type_names=tuple(names),
        type_tokens=tuple(toks),
        whitespace_ids=ws,
    )


def _match_starts(seq: torch.Tensor, pat: tuple[int, ...]) -> torch.Tensor:
    """(B, T) bool: True at i iff seq[:, i:i+len(pat)] == pat."""
    b, t = seq.shape
    n = len(pat)
    out = torch.zeros((b, t), dtype=torch.bool, device=seq.device)
    if t < n:
        return out
    p = torch.tensor(pat, dtype=seq.dtype, device=seq.device)
    out[:, : t - n + 1] = (seq.unfold(1, n, 1) == p).all(-1)
    return out


def template_types(labels: torch.Tensor, patterns: TemplatePatterns) -> torch.Tensor:
    """Per-position template type ids in the trainer's shifted frame.

    Args:
        labels: (B, S) labels (-100 = not trained).
        patterns: from ``build_template_patterns``.

    Returns:
        (B, S - 1) int64: type index into ``patterns.type_names`` at template positions, -1 elsewhere. Index i refers
        to the target token labels[:, i + 1] (= per_token_logps[:, i]).
    """
    tgt = labels[:, 1:]
    types = torch.full(tgt.shape, -1, dtype=torch.long, device=labels.device)
    for ids, type0 in patterns.match_order():
        starts = _match_starts(tgt, ids)
        free = types < 0
        # a (trimmed) match counts only where none of its positions is already claimed
        ok = starts.clone()
        for j in range(len(ids)):
            ok[:, : ok.shape[1] - j] &= free[:, j:]
        for j in range(len(ids)):
            hit = torch.zeros_like(ok)
            hit[:, j:] = ok[:, : ok.shape[1] - j]
            types = torch.where(hit, torch.full_like(types, type0 + j), types)
    sep_starts = _match_starts(tgt, patterns.content_sep)
    content_sep_type = len(patterns.type_names) - 1
    types = torch.where(sep_starts & (types < 0), torch.full_like(types, content_sep_type), types)
    return types


def template_mask(labels: torch.Tensor, patterns: TemplatePatterns) -> torch.Tensor:
    """(B, S - 1) bool: chosen-side template positions (shifted frame), see ``template_types``."""
    return template_types(labels, patterns) >= 0
