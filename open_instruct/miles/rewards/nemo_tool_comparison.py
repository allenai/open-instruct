# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Vendored from NVIDIA-NeMo/Gym at 966145fc02ae58b887fd49379675604cc38fdd43:
# resources_servers/single_step_tool_use_with_argument_comparison/common/verification_utils.py
# Local changes: project logger/import/type style only. Keep comparison semantics upstream.
import json
from collections import Counter
from enum import StrEnum
from json import JSONDecodeError
from typing import Annotated, Any, Literal, TypeAlias, cast

from jsonschema import validators
from jsonschema.exceptions import SchemaError as JSONSchemaError
from jsonschema.exceptions import ValidationError as JSONValidationError
from pydantic import BaseModel, Field

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


class MessageAction(BaseModel):
    type: Literal["message"]
    content: str


class FunctionCallAction(BaseModel):
    type: Literal["function_call"]
    name: str
    arguments: str


class FunctionCallBatchAction(BaseModel):
    type: Literal["function_call_batch"]
    calls: list[FunctionCallAction] = Field(min_length=1)


# Actions are canonical on both sides of a comparison: dataset rows deserialize into them, and model
# responses are normalized into them by `response_utils.extract_action`. The alias keeps the name
# that dataset validation tooling imports.
ExpectedAction: TypeAlias = Annotated[
    MessageAction | FunctionCallAction | FunctionCallBatchAction, Field(discriminator="type")
]


def get_tool_calls(action: ExpectedAction) -> list[FunctionCallAction]:
    """Flatten an action into the tool calls it represents, so single and parallel calls share a path."""
    if isinstance(action, FunctionCallBatchAction):
        return list(action.calls)

    if isinstance(action, FunctionCallAction):
        return [action]

    return []


class StepRewardCategory(StrEnum):
    NO_ACTION_FOUND = "No tool call or chat message was found in the response"
    NO_EXPECTED_TOOL_CALL = "No tool call was found when one was expected"
    EXPECTED_CHAT_MESSAGE_FOUND = "A chat message was found as expected"
    NO_EXPECTED_CHAT_MESSAGE = "A tool call was executed when a chat message was expected"
    UNEXPECTED_TOOL = "The tool in a tool call is not the expected tool"
    ARGUMENTS_DECODE_ERROR = "An error occurred when decoding the arguments string in a tool call as a JSON object"
    ARGUMENT_VALUE_TYPE_DIFFERENT = "The type of an argument value in a tool call is different than the expected type"
    ARGUMENT_OBJECT_KEYS_DIFFERENT = (
        "The keys in an object in an argument value in a tool call are different than the keys in the expected object"
    )
    ARGUMENT_LIST_LENGTH_DIFFERENT = (
        "A list in an argument value in a tool call has a different length than the expected list"
    )
    ARGUMENT_LIST_F1_BELOW_THRESHOLD = "The F1 score for a list argument is below the threshold"
    ARGUMENT_LIST_F1_PARTIAL = "The F1 score for a list argument is a partial match"
    ARGUMENT_LIST_F1_WEAK_MATCH = "The list argument F1 match failed strong-match guardrails"
    ARGUMENT_LIST_SCHEMA_MALFORMED = "A list argument in a tool call does not conform to the expected schema"
    TOOL_SCHEMA_NOT_FOUND = "No declared tool schema was found for the tool call"
    TOOL_SCHEMA_VALIDATION_FAILED = "The tool call arguments do not conform to the declared tool schema"
    ARGUMENT_VALUE_DIFFERENT = "An argument value in a tool call is different than the expected value"
    EXPECTED_TOOL_CALL = "A tool call that matches the expected tool call was found"
    FUNCTION_CALL_BATCH_LENGTH_DIFFERENT = "The number of tool calls in a batch is different than expected"
    EXPECTED_TOOL_CALL_BATCH = "A tool-call batch that matches the expected tool calls was found"


class ParallelToolCallRewardMode(StrEnum):
    """How an admissible parallel tool-call response converts its matched-call count into a reward."""

    BINARY_STRICT = "binary_strict"
    FRACTIONAL = "fractional"
    F1 = "f1"


class ActionComparisonResult(BaseModel):
    reward: float
    category: StepRewardCategory


class ListF1MatchDetail(BaseModel):
    expected_values: list[Any]
    actual_values: list[Any]
    score_matrix: list[list[float]]
    matched_pairs: list[tuple[int, int]]
    tp: int
    precision: float
    recall: float
    f1: float
    actual_duplicate_count: int = 0
    actual_to_expected_ratio: float | None = None
    strong_match_failure_reasons: list[str] = Field(default_factory=list)


class ToolCallArgumentFilter(BaseModel):
    included_argument_names: list[str] | None = None


class ToolCallArgumentComparisonOverride(BaseModel):
    """Comparison settings for one argument of one tool, set under `argument_comparison_overrides`."""

    word_count_similarity_threshold: float | None = None
    word_count_min_precision: float | None = None
    word_count_min_recall: float | None = None
    word_count_max_actual_to_expected_ratio: float | None = None
    word_count_max_unmatched_actual_words: int | None = None
    list_f1_relaxed_min_expected_len: int | None = None
    list_f1_max_unmatched_expected: int | None = None
    list_f1_max_unmatched_actual: int | None = None
    keep_quotes: bool | None = None
    coerce_actual_string_to_list: bool = False
    coerce_actual_string_to_singleton_list: bool = False
    validate_list_item_schema: bool = False
    reject_empty_string_list_items: bool = False


class ToolCallComparatorConfig(BaseModel):
    word_count_similarity_threshold: float
    floating_point_comparison_threshold: float = 1e-6

    # Master switch: does the NUMBER of tool calls the model made affect its reward?
    #
    # False (default) turns parallel tool-call rewarding off entirely. The question asked of a
    # response is only "did it make the expected call(s)?", so surplus calls are never penalized.
    # This reproduces the behaviour that predates parallel tool-call support, which is why it is the
    # default: existing datasets score exactly as they did before. It is also the honest default,
    # because chat templates do not render differently for `parallel_tool_calls` (the Nemotron
    # template never references the flag), so a model is never told how many calls it may make.
    #
    # True turns it on: the call count becomes part of the verdict, and the three settings below
    # decide which counts are admissible and how partial matches are scored. Turn this on for
    # datasets that use `expected_action.type: function_call_batch`.
    parallel_tool_call_rewarding: bool = False

    # The three settings below are only consulted when `parallel_tool_call_rewarding` is True.

    # Cardinality gate: whether a response that makes fewer / more tool calls than expected is
    # admissible at all. Both default to False, so the call count must match exactly.
    allow_subset: bool = False
    allow_superset: bool = False

    # Scoring for responses that clear the gate. See `ActionComparator.compare_tool_calls`.
    parallel_tool_call_reward_mode: ParallelToolCallRewardMode = ParallelToolCallRewardMode.BINARY_STRICT

    # The settings below apply only when one expected call is compared with one actual call. A
    # parallel batch is always scored with the settings above. See `without_single_call_options`.

    # Score list arguments as an unordered set with F1 instead of element by element in order.
    use_f1_for_list: bool = False
    list_f1_threshold: float = 0.5
    # True: full credit when F1 >= `list_f1_threshold`. False: the reward is the F1 score itself.
    use_list_f1_threshold: bool = True
    # Full credit only when every guardrail below holds. Overrides `use_list_f1_threshold`.
    use_strong_list_reward: bool = False
    list_f1_min_precision: float = 1.0
    list_f1_min_recall: float = 1.0
    list_f1_max_actual_to_expected_ratio: float = 1.0
    list_f1_reject_duplicate_actual_values: bool = True
    # Tool name -> the top-level arguments that are compared. Other arguments are ignored.
    argument_filters: dict[str, ToolCallArgumentFilter] | None = None
    # Tool name -> argument name -> comparison settings for that argument.
    argument_comparison_overrides: dict[str, dict[str, ToolCallArgumentComparisonOverride]] | None = None
    # Keep a run inside matching quotes as one token, so a quoted phrase differs from its bare words.
    keep_quotes: bool = False
    # Reject a call whose tool is not declared in the request, or whose arguments fail its JSON Schema.
    validate_against_declared_tool_schema: bool = False

    def without_single_call_options(self) -> "ToolCallComparatorConfig":
        """The config that scores a parallel batch: the base and parallel settings only."""
        return ToolCallComparatorConfig(
            word_count_similarity_threshold=self.word_count_similarity_threshold,
            floating_point_comparison_threshold=self.floating_point_comparison_threshold,
            parallel_tool_call_rewarding=self.parallel_tool_call_rewarding,
            allow_subset=self.allow_subset,
            allow_superset=self.allow_superset,
            parallel_tool_call_reward_mode=self.parallel_tool_call_reward_mode,
        )


_QUOTE_CHARS = ('"', "'")


def tokenize_with_quoted_phrases(value: str) -> list[str]:
    """Tokenize on whitespace, but keep a run inside matching quotes (`"..."` or `'...'`)
    as one token with the quote characters preserved.

    If any quote is opened and never closed, all quote characters are stripped from the
    original string and the result is whitespace-split. A warning is logged so
    mis-quoted inputs are observable.
    """
    lowered = value.strip().lower()
    tokens: list[str] = []
    buf: list[str] = []
    active_quote: str | None = None

    def flush_buf() -> None:
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    for ch in lowered:
        if active_quote is not None:
            buf.append(ch)
            if ch == active_quote:
                flush_buf()
                active_quote = None
        elif ch in _QUOTE_CHARS:
            flush_buf()
            active_quote = ch
            buf.append(ch)
        elif ch.isspace():
            flush_buf()
        else:
            buf.append(ch)

    if active_quote is not None:
        logger.warning(
            "Unbalanced quote character %r in string %r; stripping all quote chars and falling back to whitespace split.",
            active_quote,
            value,
        )
        stripped = lowered
        for q in _QUOTE_CHARS:
            stripped = stripped.replace(q, "")
        return stripped.split()

    flush_buf()
    return tokens


def build_declared_tool_schema_map(declared_tools: list[Any] | None) -> dict[str, dict[str, Any]]:
    """Map each declared tool name to its parameters JSON Schema. Accepts Responses and Chat Completions shapes."""
    declared_tool_schemas: dict[str, dict[str, Any]] = {}
    for tool_definition in declared_tools or []:
        if hasattr(tool_definition, "model_dump"):
            tool_definition = tool_definition.model_dump(mode="python")

        if not isinstance(tool_definition, dict):
            continue

        function_definition = tool_definition.get("function")
        if isinstance(function_definition, dict):
            tool_name = function_definition.get("name")
            tool_schema = function_definition.get("parameters")
        else:
            tool_name = tool_definition.get("name")
            tool_schema = tool_definition.get("parameters")

        if isinstance(tool_name, str) and isinstance(tool_schema, dict):
            declared_tool_schemas[tool_name] = tool_schema

    return declared_tool_schemas


def validate_tool_call_against_declared_schema(
    tool_call: FunctionCallAction, declared_tools: list[Any] | None
) -> StepRewardCategory | None:
    """Return the failure category, or None when the call's arguments conform to its declared tool schema."""
    tool_schema = build_declared_tool_schema_map(declared_tools).get(tool_call.name)
    if tool_schema is None:
        return StepRewardCategory.TOOL_SCHEMA_NOT_FOUND

    try:
        actual_arguments = json.loads(tool_call.arguments)
    except (JSONDecodeError, UnicodeDecodeError):
        return StepRewardCategory.ARGUMENTS_DECODE_ERROR

    try:
        validator_class = validators.validator_for(tool_schema)
        validator_class.check_schema(tool_schema)
        validator_class(tool_schema).validate(actual_arguments)
    except (JSONSchemaError, JSONValidationError):
        return StepRewardCategory.TOOL_SCHEMA_VALIDATION_FAILED

    return None


def find_maximum_matching(candidates: list[list[int]]) -> dict[int, int]:
    """Maximum bipartite matching between expected and actual tool calls (Kuhn's algorithm).

    `candidates[expected_index]` lists the actual-call indices that expected call could match, and the
    returned mapping goes from actual index to the expected index it was matched with. Greedy pairing
    is not enough here: argument matching is a fuzzy relation, so one actual call can satisfy several
    expected calls and an early arbitrary pairing can strand a later expected call that had no
    alternative. Augmenting paths undo those pairings and recover the true maximum.
    """
    matching: dict[int, int] = {}

    # Matching the most constrained expected calls first keeps the number of augmenting paths small.
    for expected_index in sorted(range(len(candidates)), key=lambda index: len(candidates[index])):
        _augment_matching(expected_index, candidates, matching, set())

    return matching


def _augment_matching(
    expected_index: int, candidates: list[list[int]], matching: dict[int, int], visited_actual_indices: set[int]
) -> bool:
    for actual_index in candidates[expected_index]:
        if actual_index in visited_actual_indices:
            continue

        visited_actual_indices.add(actual_index)
        is_unmatched = actual_index not in matching
        if is_unmatched or _augment_matching(matching[actual_index], candidates, matching, visited_actual_indices):
            matching[actual_index] = expected_index
            return True

    return False


class ActionComparator(BaseModel):
    config: ToolCallComparatorConfig
    list_f1_match_details: list[ListF1MatchDetail] = Field(default_factory=list)

    def compare_action(self, expected_action: ExpectedAction, actual_action: ExpectedAction) -> ActionComparisonResult:
        # The details describe this comparison only, so a comparator reused across rows does not accumulate them.
        self.list_f1_match_details = []
        match expected_action:
            case MessageAction():
                # Currently, any chat message is assigned a reward of one.
                if isinstance(actual_action, MessageAction):
                    return ActionComparisonResult(reward=1.0, category=StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND)

                return ActionComparisonResult(reward=0.0, category=StepRewardCategory.NO_EXPECTED_CHAT_MESSAGE)

            case FunctionCallAction() | FunctionCallBatchAction():
                if isinstance(actual_action, MessageAction):
                    return ActionComparisonResult(reward=0.0, category=StepRewardCategory.NO_EXPECTED_TOOL_CALL)

                return self.compare_tool_calls(get_tool_calls(expected_action), get_tool_calls(actual_action))

            case _:
                raise NotImplementedError(f"Unsupported expected action: {expected_action!r}")

    def compare_tool_calls(
        self, expected_calls: list[FunctionCallAction], actual_calls: list[FunctionCallAction]
    ) -> ActionComparisonResult:
        """Score a set of tool calls against the expected set, ignoring the order they were emitted in.

        `parallel_tool_call_rewarding` is the master switch. While it is False (the default) the number
        of calls is simply not part of the verdict: the response is asked only whether it made the
        expected call(s), and surplus calls cost nothing. That reproduces the behaviour that predates
        parallel tool-call support.

        Turning it on makes the call count matter, in two independent stages. The cardinality gate
        (`allow_subset` / `allow_superset`) decides whether a response that under- or over-calls is
        admissible at all; anything it rejects scores zero. `parallel_tool_call_reward_mode` then decides
        how much credit an admissible response earns:

        - `binary_strict` — 1.0 only if every required call matched, else 0.0.
        - `fractional` — the matched fraction of the required calls. Surplus calls permitted by
          `allow_superset` are free, so this rewards recall but not precision.
        - `f1` — the harmonic mean of precision and recall, `2 * matched / (expected + actual)`. Missing
          and surplus calls are penalized symmetrically, so a response only reaches 1.0 by matching the
          expected calls exactly.
        """
        expected_count = len(expected_calls)
        actual_count = len(actual_calls)

        if expected_count == 1 and actual_count == 1:
            # Preserve the single-call categories that predate parallel tool-call support.
            return self.compare_tool_call(expected_calls[0], actual_calls[0])

        if not self.is_call_count_admissible(expected_count, actual_count):
            return ActionComparisonResult(reward=0.0, category=StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT)

        batch_comparator = ActionComparator(config=self.config.without_single_call_options())
        candidates, failure_categories = batch_comparator.build_match_candidates(expected_calls, actual_calls)
        matching = find_maximum_matching(candidates)
        reward = self.score_matched_calls(len(matching), expected_count, actual_count)
        if reward == 1.0:
            # A single expected call keeps the category it had before parallel support existed.
            category = (
                StepRewardCategory.EXPECTED_TOOL_CALL
                if expected_count == 1
                else StepRewardCategory.EXPECTED_TOOL_CALL_BATCH
            )
            return ActionComparisonResult(reward=reward, category=category)

        category = self.resolve_failure_category(
            matched_expected_indices=set(matching.values()),
            failure_categories=failure_categories,
            expected_count=expected_count,
            actual_count=actual_count,
        )
        return ActionComparisonResult(reward=reward, category=category)

    def is_call_count_admissible(self, expected_count: int, actual_count: int) -> bool:
        if actual_count == expected_count:
            return True

        # With parallel tool-call rewarding off, the call count is not part of the verdict at all.
        if not self.config.parallel_tool_call_rewarding:
            return True

        if actual_count < expected_count:
            return self.config.allow_subset

        return self.config.allow_superset

    def required_match_count(self, expected_count: int, actual_count: int) -> int:
        """How many expected calls must match for full credit under `binary_strict` and `fractional`."""
        if self.config.allow_subset:
            return min(expected_count, actual_count)

        return expected_count

    def score_matched_calls(self, matched_count: int, expected_count: int, actual_count: int) -> float:
        if not self.config.parallel_tool_call_rewarding:
            # The call count is not part of the verdict, so neither the gate nor the reward mode
            # applies: full credit exactly when every expected call was matched.
            return 1.0 if matched_count == expected_count else 0.0

        if self.config.parallel_tool_call_reward_mode == ParallelToolCallRewardMode.F1:
            total_count = expected_count + actual_count
            return 2 * matched_count / total_count if total_count else 0.0

        required_count = self.required_match_count(expected_count, actual_count)
        if self.config.parallel_tool_call_reward_mode == ParallelToolCallRewardMode.FRACTIONAL:
            return matched_count / required_count if required_count else 0.0

        return 1.0 if matched_count == required_count else 0.0

    def build_match_candidates(
        self, expected_calls: list[FunctionCallAction], actual_calls: list[FunctionCallAction]
    ) -> tuple[list[list[int]], list[StepRewardCategory]]:
        """Pair every expected call with the actual calls it matches, keeping why the others missed."""
        candidates: list[list[int]] = []
        failure_categories: list[StepRewardCategory] = []

        for expected_call in expected_calls:
            matching_actual_indices: list[int] = []
            failure_category = StepRewardCategory.UNEXPECTED_TOOL

            for actual_index, actual_call in enumerate(actual_calls):
                result = self.compare_tool_call(expected_call, actual_call)
                if result.reward == 1.0:
                    matching_actual_indices.append(actual_index)

                elif failure_category == StepRewardCategory.UNEXPECTED_TOOL:
                    # Keep the first reason that got past the tool name; it explains the closest near miss.
                    failure_category = result.category

            candidates.append(matching_actual_indices)
            failure_categories.append(failure_category)

        return candidates, failure_categories

    def resolve_failure_category(
        self,
        matched_expected_indices: set[int],
        failure_categories: list[StepRewardCategory],
        expected_count: int,
        actual_count: int,
    ) -> StepRewardCategory:
        unmatched_expected_indices = [
            expected_index
            for expected_index in range(expected_count)
            if expected_index not in matched_expected_indices
        ]

        # Either every expected call was matched and the only defect left is surplus calls, or every
        # actual call was consumed and the only defect left is missing calls. Reporting an argument
        # mismatch in those cases would point at a comparison that is not why the reward was docked.
        if not unmatched_expected_indices or len(matched_expected_indices) == actual_count:
            return StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT

        return failure_categories[unmatched_expected_indices[0]]

    def compare_tool_call(
        self, expected_tool_call: FunctionCallAction, actual_tool_call: FunctionCallAction
    ) -> ActionComparisonResult:
        tool_name = expected_tool_call.name
        if tool_name != actual_tool_call.name:
            return ActionComparisonResult(reward=0.0, category=StepRewardCategory.UNEXPECTED_TOOL)

        # It is assumed that the expected arguments string is a string representation of a JSON object.
        expected_arguments = json.loads(expected_tool_call.arguments)

        try:
            actual_arguments = json.loads(actual_tool_call.arguments)
        except (JSONDecodeError, UnicodeDecodeError):
            return ActionComparisonResult(reward=0.0, category=StepRewardCategory.ARGUMENTS_DECODE_ERROR)

        argument_filter = (self.config.argument_filters or {}).get(tool_name)
        if argument_filter is not None:
            expected_arguments = self._apply_argument_filter(argument_filter, expected_arguments)
            actual_arguments = self._apply_argument_filter(argument_filter, actual_arguments)

        score, category = self.compare_tool_call_arguments(expected_arguments, actual_arguments, tool_name=tool_name)
        if score == 1.0:
            return ActionComparisonResult(reward=1.0, category=StepRewardCategory.EXPECTED_TOOL_CALL)

        return ActionComparisonResult(reward=score, category=cast(StepRewardCategory, category))

    def compare_tool_call_arguments(
        self,
        expected_value: Any,
        actual_value: Any,
        *,
        tool_name: str | None = None,
        argument_override: ToolCallArgumentComparisonOverride | None = None,
    ) -> tuple[float, StepRewardCategory | None]:
        """Score an argument value in [0, 1]. Only list F1 without a threshold yields a partial score."""
        if not isinstance(actual_value, type(expected_value)):
            return 0.0, StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT

        if isinstance(expected_value, dict):
            if set(expected_value.keys()) != set(actual_value.keys()):
                return 0.0, StepRewardCategory.ARGUMENT_OBJECT_KEYS_DIFFERENT

            for expected_dict_key, expected_dict_value in expected_value.items():
                # An override set on a top-level argument also applies to everything nested under it.
                child_argument_override = argument_override or self._argument_comparison_override(
                    tool_name, expected_dict_key
                )
                actual_dict_value = self._maybe_coerce_actual_string_to_list(
                    expected_dict_value, actual_value[expected_dict_key], child_argument_override
                )
                dict_value_score, dict_value_category = self.compare_tool_call_arguments(
                    expected_dict_value,
                    actual_dict_value,
                    tool_name=tool_name,
                    argument_override=child_argument_override,
                )
                if dict_value_score < 1.0:
                    return dict_value_score, dict_value_category

            return 1.0, None

        elif isinstance(expected_value, list):
            list_schema_category = self._validate_actual_list_schema(expected_value, actual_value, argument_override)
            if list_schema_category is not None:
                return 0.0, list_schema_category

            if self.config.use_f1_for_list:
                return self._compare_lists_with_f1(
                    expected_value, actual_value, tool_name=tool_name, argument_override=argument_override
                )

            if len(expected_value) != len(actual_value):
                return 0.0, StepRewardCategory.ARGUMENT_LIST_LENGTH_DIFFERENT

            for expected_list_element, actual_list_element in zip(expected_value, actual_value):
                list_element_score, list_element_category = self.compare_tool_call_arguments(
                    expected_list_element, actual_list_element
                )
                if list_element_score < 1.0:
                    return list_element_score, list_element_category

            return 1.0, None

        elif isinstance(expected_value, float):
            if abs(actual_value - expected_value) < self.config.floating_point_comparison_threshold:
                return 1.0, None
            else:
                return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

        elif isinstance(expected_value, str):
            return self._compare_strings(expected_value, actual_value, argument_override)

        elif expected_value == actual_value:
            return 1.0, None

        else:
            return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

    def _compare_strings(
        self, expected_value: str, actual_value: str, argument_override: ToolCallArgumentComparisonOverride | None
    ) -> tuple[float, StepRewardCategory | None]:
        # For now, strings are compared by using whitespace to split them into lower-case
        # words, counting the words, and comparing the word counts using Jaccard similarity.
        # With keep_quotes, a run inside matching quotes is kept as one token, so an
        # exact-phrase query like '"hello world"' is distinct from the unquoted words.
        keep_quotes = self.config.keep_quotes
        word_count_similarity_threshold = self.config.word_count_similarity_threshold
        if argument_override is not None:
            if argument_override.keep_quotes is not None:
                keep_quotes = argument_override.keep_quotes
            if argument_override.word_count_similarity_threshold is not None:
                word_count_similarity_threshold = argument_override.word_count_similarity_threshold

        if keep_quotes:
            expected_tokens = tokenize_with_quoted_phrases(expected_value)
            actual_tokens = tokenize_with_quoted_phrases(actual_value)
        else:
            expected_tokens = expected_value.strip().lower().split()
            actual_tokens = actual_value.strip().lower().split()
        expected_word_counts = Counter(expected_tokens)
        actual_word_counts = Counter(actual_tokens)
        expected_word_total = expected_word_counts.total()
        actual_word_total = actual_word_counts.total()

        if expected_word_total < 2 or actual_word_total < 2:
            if expected_value != actual_value:
                return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
            return 1.0, None

        intersection_word_total = (expected_word_counts & actual_word_counts).total()
        word_count_similarity = intersection_word_total / (expected_word_total + actual_word_total)
        if word_count_similarity < word_count_similarity_threshold:
            return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

        if argument_override is not None:
            word_count_precision = intersection_word_total / actual_word_total
            word_count_recall = intersection_word_total / expected_word_total
            actual_to_expected_ratio = actual_word_total / expected_word_total
            unmatched_actual_words = actual_word_total - intersection_word_total

            if (
                argument_override.word_count_min_precision is not None
                and word_count_precision < argument_override.word_count_min_precision
            ):
                return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
            if (
                argument_override.word_count_min_recall is not None
                and word_count_recall < argument_override.word_count_min_recall
            ):
                return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
            if (
                argument_override.word_count_max_actual_to_expected_ratio is not None
                and actual_to_expected_ratio > argument_override.word_count_max_actual_to_expected_ratio
            ):
                return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
            if (
                argument_override.word_count_max_unmatched_actual_words is not None
                and unmatched_actual_words > argument_override.word_count_max_unmatched_actual_words
            ):
                return 0.0, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT

        return 1.0, None

    def _compare_lists_with_f1(
        self,
        expected_value: list[Any],
        actual_value: list[Any],
        *,
        tool_name: str | None,
        argument_override: ToolCallArgumentComparisonOverride | None,
    ) -> tuple[float, StepRewardCategory | None]:
        """Match list elements as an unordered set and score the match with F1."""
        if not expected_value and not actual_value:
            return 1.0, None
        if not expected_value or not actual_value:
            return 0.0, StepRewardCategory.ARGUMENT_LIST_F1_BELOW_THRESHOLD

        score_matrix = [
            [
                self.compare_tool_call_arguments(
                    expected_element, actual_element, tool_name=tool_name, argument_override=argument_override
                )[0]
                for actual_element in actual_value
            ]
            for expected_element in expected_value
        ]
        # An element pair counts as a match only on a full score.
        candidates = [[actual_index for actual_index, score in enumerate(row) if score == 1.0] for row in score_matrix]
        matched_pairs = sorted(
            (expected_index, actual_index)
            for actual_index, expected_index in find_maximum_matching(candidates).items()
        )
        tp = len(matched_pairs)

        precision = tp / len(actual_value)
        recall = tp / len(expected_value)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        actual_duplicate_count = self._count_duplicate_list_values(actual_value)
        actual_to_expected_ratio = len(actual_value) / len(expected_value)
        strong_match_failure_reasons = self._strong_list_match_failure_reasons(
            expected_count=len(expected_value),
            actual_count=len(actual_value),
            true_positive_count=tp,
            precision=precision,
            recall=recall,
            f1=f1,
            actual_to_expected_ratio=actual_to_expected_ratio,
            actual_duplicate_count=actual_duplicate_count,
            argument_override=argument_override,
        )
        self.list_f1_match_details.append(
            ListF1MatchDetail(
                expected_values=expected_value,
                actual_values=actual_value,
                score_matrix=score_matrix,
                matched_pairs=matched_pairs,
                tp=tp,
                precision=precision,
                recall=recall,
                f1=f1,
                actual_duplicate_count=actual_duplicate_count,
                actual_to_expected_ratio=actual_to_expected_ratio,
                strong_match_failure_reasons=strong_match_failure_reasons,
            )
        )

        if self.config.use_strong_list_reward:
            if strong_match_failure_reasons:
                return 0.0, StepRewardCategory.ARGUMENT_LIST_F1_WEAK_MATCH
            return 1.0, None

        if self.config.use_list_f1_threshold:
            if f1 >= self.config.list_f1_threshold:
                return 1.0, None
            return 0.0, StepRewardCategory.ARGUMENT_LIST_F1_BELOW_THRESHOLD

        if f1 == 1.0:
            return 1.0, None
        if f1 > 0.0:
            return f1, StepRewardCategory.ARGUMENT_LIST_F1_PARTIAL
        return 0.0, StepRewardCategory.ARGUMENT_LIST_F1_BELOW_THRESHOLD

    @classmethod
    def _apply_argument_filter(cls, argument_filter: ToolCallArgumentFilter, argument_value: Any) -> Any:
        included_argument_names = argument_filter.included_argument_names
        if included_argument_names is not None and isinstance(argument_value, dict):
            return {key: value for key, value in argument_value.items() if key in included_argument_names}

        return argument_value

    def _argument_comparison_override(
        self, tool_name: str | None, argument_name: str
    ) -> ToolCallArgumentComparisonOverride | None:
        if tool_name is None or self.config.argument_comparison_overrides is None:
            return None
        return self.config.argument_comparison_overrides.get(tool_name, {}).get(argument_name)

    @staticmethod
    def _maybe_coerce_actual_string_to_list(
        expected_value: Any, actual_value: Any, argument_override: ToolCallArgumentComparisonOverride | None
    ) -> Any:
        """Recover a list argument that the model sent as a string, when the override allows it."""
        if argument_override is None:
            return actual_value
        if not isinstance(expected_value, list) or not isinstance(actual_value, str):
            return actual_value

        if argument_override.coerce_actual_string_to_list:
            try:
                parsed_value = json.loads(actual_value)
            except (JSONDecodeError, TypeError):
                parsed_value = None

            if isinstance(parsed_value, list):
                return parsed_value

        if argument_override.coerce_actual_string_to_singleton_list and len(expected_value) == 1:
            singleton_value = actual_value.strip()
            if singleton_value.startswith("[") and singleton_value.endswith("]"):
                singleton_value = singleton_value[1:-1].strip()
            return [singleton_value]
        return actual_value

    @staticmethod
    def _validate_actual_list_schema(
        expected_value: list[Any],
        actual_value: list[Any],
        argument_override: ToolCallArgumentComparisonOverride | None,
    ) -> StepRewardCategory | None:
        """Reject actual list items whose type differs from the (uniform) expected item type."""
        if argument_override is None or not argument_override.validate_list_item_schema:
            return None
        if not expected_value:
            return None

        expected_element_type = type(expected_value[0])
        if any(type(expected_element) is not expected_element_type for expected_element in expected_value):
            return None

        for actual_element in actual_value:
            if type(actual_element) is not expected_element_type:
                return StepRewardCategory.ARGUMENT_LIST_SCHEMA_MALFORMED
            if (
                argument_override.reject_empty_string_list_items
                and isinstance(actual_element, str)
                and not actual_element.strip()
            ):
                return StepRewardCategory.ARGUMENT_LIST_SCHEMA_MALFORMED
        return None

    def _strong_list_match_failure_reasons(
        self,
        *,
        expected_count: int,
        actual_count: int,
        true_positive_count: int,
        precision: float,
        recall: float,
        f1: float,
        actual_to_expected_ratio: float,
        actual_duplicate_count: int,
        argument_override: ToolCallArgumentComparisonOverride | None,
    ) -> list[str]:
        reasons: list[str] = []
        if self._uses_relaxed_list_guardrails(argument_override, expected_count):
            # Long expected lists replace the precision / recall floors with absolute miss / extra limits.
            assert argument_override is not None
            unmatched_expected = expected_count - true_positive_count
            unmatched_actual = actual_count - true_positive_count
            if (
                argument_override.list_f1_max_unmatched_expected is not None
                and unmatched_expected > argument_override.list_f1_max_unmatched_expected
            ):
                reasons.append("unmatched_expected_above_limit")
            if (
                argument_override.list_f1_max_unmatched_actual is not None
                and unmatched_actual > argument_override.list_f1_max_unmatched_actual
            ):
                reasons.append("unmatched_actual_above_limit")
            if self.config.list_f1_reject_duplicate_actual_values and actual_duplicate_count:
                reasons.append("duplicate_actual_values")
            return reasons

        if f1 < self.config.list_f1_threshold:
            reasons.append("f1_below_threshold")
        if precision < self.config.list_f1_min_precision:
            reasons.append("precision_below_floor")
        if recall < self.config.list_f1_min_recall:
            reasons.append("recall_below_floor")
        if actual_to_expected_ratio > self.config.list_f1_max_actual_to_expected_ratio:
            reasons.append("actual_list_overbroad")
        if self.config.list_f1_reject_duplicate_actual_values and actual_duplicate_count:
            reasons.append("duplicate_actual_values")
        return reasons

    @staticmethod
    def _uses_relaxed_list_guardrails(
        argument_override: ToolCallArgumentComparisonOverride | None, expected_count: int
    ) -> bool:
        if argument_override is None or argument_override.list_f1_relaxed_min_expected_len is None:
            return False
        return expected_count >= argument_override.list_f1_relaxed_min_expected_len

    @staticmethod
    def _count_duplicate_list_values(values: list[Any]) -> int:
        identities = [json.dumps(value, sort_keys=True, default=str) for value in values]
        return len(identities) - len(set(identities))
