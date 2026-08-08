"""Rung-1 data-validation tests: SafeComp preference tokenization with the real Qwen3.5 tokenizer.

Validates, on rows of the exported preference set (``data/dpo/dpo_comp1k_v2.jsonl``):
  - the tools column is rendered into both branches' chat templates (P1 regression guard);
  - loss labels cover exactly the TAIL assistant turns (``preference_safecomp_tail_tokenize_v1``);
  - chosen/rejected token streams are identical over the shared prefix;
  - no branch exceeds the 32,768-token budget (truncation would cut the legal ending —
    a prereg validity condition);
  - labeled-token lengths per branch are reported (prereg seam policy).

Environment:
  SAFECOMP_DPO_JSONL   path to the exported pairs (default: the comp1k_v2 export on weka;
                       tests skip if absent)
  QWEN3_5_TOKENIZER    tokenizer path (default: the SFT anchor checkpoint on weka)
  SAFECOMP_FULL_SWEEP  set to 1 to check every row instead of a 20-row sample
"""

import json
import os

import pytest
import transformers

from open_instruct.dataset_transformation import (
    CHOSEN_INPUT_IDS_KEY,
    CHOSEN_LABELS_KEY,
    MASKED_TOKEN_VALUE,
    REJECTED_INPUT_IDS_KEY,
    REJECTED_LABELS_KEY,
    TOOLS_COLUMN_KEY,
    _normalize_tools_for_chat_template,
    _tokenize_tulu_sft_with_assistant_labels,
    preference_safecomp_tail_tokenize_v1,
    preference_tulu_tokenize_and_truncate_v1_2,
)

MAX_SEQ_LENGTH = 32768
_DEFAULT_JSONL = "/weka/oe-adapt-default/mingqianz/agent-safety/data/dpo/dpo_comp1k_v2.jsonl"
_DEFAULT_TOKENIZER = "/weka/oe-adapt-default/mingqianz/models/qwen35-9b-safecomp-v2/sft_qwen35_9b_safecomp_v2"

_jsonl_path = os.environ.get("SAFECOMP_DPO_JSONL", _DEFAULT_JSONL)
pytestmark = pytest.mark.skipif(
    not os.path.exists(_jsonl_path), reason=f"SafeComp DPO export not found at {_jsonl_path}"
)


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(
        os.environ.get("QWEN3_5_TOKENIZER", _DEFAULT_TOKENIZER), use_fast=True
    )


@pytest.fixture(scope="module")
def rows():
    with open(_jsonl_path) as f:
        all_rows = [json.loads(line) for line in f]
    if os.environ.get("SAFECOMP_FULL_SWEEP"):
        return all_rows
    stride = max(1, len(all_rows) // 20)
    return all_rows[::stride][:20]


def _tool_names(row) -> list[str]:
    tools = _normalize_tools_for_chat_template(row.get(TOOLS_COLUMN_KEY)) or []
    return [t["function"]["name"] for t in tools if isinstance(t.get("function"), dict) and t["function"].get("name")]


def test_tools_rendered_into_both_branches(tokenizer, rows):
    checked = 0
    for row in rows:
        names = _tool_names(row)
        if not names:
            continue
        encoded = preference_safecomp_tail_tokenize_v1(dict(row), tokenizer, MAX_SEQ_LENGTH)
        for ids_key in (CHOSEN_INPUT_IDS_KEY, REJECTED_INPUT_IDS_KEY):
            decoded = tokenizer.decode(encoded[ids_key])
            for name in names:
                assert name in decoded, f"tool {name!r} not rendered in {ids_key}"
        # Regression guard: stripping tools must change the tokenization.
        stripped = dict(row)
        stripped[TOOLS_COLUMN_KEY] = None
        encoded_no_tools = preference_safecomp_tail_tokenize_v1(stripped, tokenizer, MAX_SEQ_LENGTH)
        assert encoded_no_tools[CHOSEN_INPUT_IDS_KEY].tolist() != encoded[CHOSEN_INPUT_IDS_KEY].tolist()
        checked += 1
    assert checked > 0, "no sampled row carried tools; widen the sample"


def test_stock_preference_transform_forwards_tools(tokenizer, rows):
    row = next(r for r in rows if _tool_names(r))
    encoded = preference_tulu_tokenize_and_truncate_v1_2(dict(row), tokenizer, MAX_SEQ_LENGTH)
    stripped = dict(row)
    stripped[TOOLS_COLUMN_KEY] = None
    encoded_no_tools = preference_tulu_tokenize_and_truncate_v1_2(stripped, tokenizer, MAX_SEQ_LENGTH)
    assert encoded[CHOSEN_INPUT_IDS_KEY].tolist() != encoded_no_tools[CHOSEN_INPUT_IDS_KEY].tolist()


def _prefix_token_len(tokenizer, row) -> int:
    n_prefix = row["metadata"]["n_prefix_messages"]
    tools = _normalize_tools_for_chat_template(row.get(TOOLS_COLUMN_KEY))
    rendered = tokenizer.apply_chat_template(
        conversation=row["chosen"][:n_prefix], tools=tools, tokenize=False, add_generation_prompt=False
    )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def _is_over_budget(tokenizer, row) -> bool:
    tools = _normalize_tools_for_chat_template(row.get(TOOLS_COLUMN_KEY))
    n_prefix = row["metadata"]["n_prefix_messages"]
    for branch in ("chosen", "rejected"):
        input_ids, _, _ = _tokenize_tulu_sft_with_assistant_labels(
            row[branch], tokenizer, tools, max_seq_length=None, label_from_message_idx=n_prefix
        )
        if input_ids.shape[1] > MAX_SEQ_LENGTH:
            return True
    return False


def test_labels_cover_exactly_tail_assistant_turns(tokenizer, rows):
    for row in rows:
        encoded = preference_safecomp_tail_tokenize_v1(dict(row), tokenizer, MAX_SEQ_LENGTH)
        prefix_len = _prefix_token_len(tokenizer, row)
        over_budget = _is_over_budget(tokenizer, row)
        for ids_key, labels_key in (
            (CHOSEN_INPUT_IDS_KEY, CHOSEN_LABELS_KEY),
            (REJECTED_INPUT_IDS_KEY, REJECTED_LABELS_KEY),
        ):
            ids, labels = encoded[ids_key], encoded[labels_key]
            unmasked = (labels != MASKED_TOKEN_VALUE).nonzero().flatten()
            if over_budget:
                # The drop signal: an over-budget pair must carry no labels on either branch,
                # so preference_tulu_filter_v1 removes it instead of training on a cut ending.
                assert len(unmasked) == 0, f"{labels_key}: over-budget pair still carries labels"
                continue
            assert len(unmasked) > 0, f"{labels_key}: no labeled tokens"
            assert (labels[unmasked] == ids[unmasked]).all(), f"{labels_key}: labels disagree with input_ids"
            assert int(unmasked[0]) >= prefix_len, (
                f"{labels_key}: first labeled token {int(unmasked[0])} inside the shared prefix ({prefix_len})"
            )


def test_shared_prefix_tokens_identical(tokenizer, rows):
    for row in rows:
        encoded = preference_safecomp_tail_tokenize_v1(dict(row), tokenizer, MAX_SEQ_LENGTH)
        chosen, rejected = encoded[CHOSEN_INPUT_IDS_KEY], encoded[REJECTED_INPUT_IDS_KEY]
        prefix_len = _prefix_token_len(tokenizer, row)
        n = min(len(chosen), len(rejected), prefix_len)
        assert chosen[:n].tolist() == rejected[:n].tolist(), (
            f"chosen/rejected diverge inside the shared prefix (first {n} tokens)"
        )


def test_budget_report_and_lengths(tokenizer, rows):
    """Report token/labeled-length distributions (prereg seam policy) and the over-budget pairs
    that the transform drops (their closing turn would not survive truncation)."""
    total_lengths, labeled_lengths = {"chosen": [], "rejected": []}, {"chosen": [], "rejected": []}
    over_budget = []
    for row in rows:
        tools = _normalize_tools_for_chat_template(row.get(TOOLS_COLUMN_KEY))
        n_prefix = row["metadata"]["n_prefix_messages"]
        row_over = False
        for branch in ("chosen", "rejected"):
            input_ids, _, labels = _tokenize_tulu_sft_with_assistant_labels(
                row[branch], tokenizer, tools, max_seq_length=None, label_from_message_idx=n_prefix
            )
            n_tokens = input_ids.shape[1]
            total_lengths[branch].append(n_tokens)
            labeled_lengths[branch].append(int((labels != MASKED_TOKEN_VALUE).sum()))
            row_over = row_over or n_tokens > MAX_SEQ_LENGTH
        if row_over:
            over_budget.append(row["metadata"].get("task_id"))
    for branch in ("chosen", "rejected"):
        totals, labeled = total_lengths[branch], labeled_lengths[branch]
        print(
            f"\n{branch}: n={len(totals)} total-tokens mean={sum(totals) / len(totals):.0f} "
            f"max={max(totals)} | labeled-tokens mean={sum(labeled) / len(labeled):.0f} "
            f"min={min(labeled)} max={max(labeled)}"
        )
    print(f"\nover-budget pairs dropped by the transform: {len(over_budget)}/{len(rows)}: {over_budget}")
