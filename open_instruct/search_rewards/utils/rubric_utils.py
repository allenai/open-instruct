"""Utility functions for adaptive rubric generation and management."""

import asyncio
import contextlib
import fcntl
import json
import os
import random
import socket
import tempfile
import time
import uuid
from typing import Any

from open_instruct import logger_utils
from open_instruct.search_rewards.utils.run_utils import extract_json_from_response, run_litellm_async

logger = logger_utils.setup_logger(__name__)


INSTANCE_WISE_RUBRIC_GENERATION_PROMPT = """
You are an expert evaluator generating adaptive rubrics to assess model responses.

## Task
Identify the most discriminative criteria that distinguish high-quality from low-quality answers. Capture subtle quality differences that existing rubrics miss.

## Output Components
- **Description**: Detailed, specific description of what makes a response excellent/problematic
- **Title**: Concise abstract label (general, not question-specific)

## Categories
1. **Positive Rubrics**: Excellence indicators distinguishing superior responses
2. **Negative Rubrics**: Critical flaws definitively degrading quality

## Core Guidelines

### 1. Discriminative Power
- Focus ONLY on criteria meaningfully separating quality levels
- Each rubric must distinguish between otherwise similar responses
- Exclude generic criteria applying equally to all responses

### 2. Novelty & Non-Redundancy
With existing/ground truth rubrics:
- Never duplicate overlapping rubrics in meaning/scope
- Identify uncovered quality dimensions
- Add granular criteria if existing ones are broad
- Return empty lists if existing rubrics are comprehensive

### 3. Avoid Mirror Rubrics
Never create positive/negative versions of same criterion:
- ❌ "Provides clear explanations" + "Lacks clear explanations"
- ✅ Choose only the more discriminative direction

### 4. Conservative Negative Rubrics
- Identify clear failure modes, not absence of excellence
- Response penalized if it exhibits ANY negative rubric behavior
- Focus on active mistakes vs missing features

## Selection Strategy

### Quantity: 1-5 total rubrics (fewer high-quality > many generic)

### Distribution Based on Response Patterns:
- **More positive**: Responses lack sophistication but avoid major errors
- **More negative**: Systematic failure patterns present
- **Balanced**: Both excellence gaps and failure modes exist
- **Empty lists**: Existing rubrics already comprehensive

## Analysis Process
1. Group responses by quality level
2. Find factors separating higher/lower clusters
3. Check if factors covered by existing rubrics
4. Select criteria with highest discriminative value

## Output Format
```json
{
  "question": "<original question verbatim>",
  "positive_rubrics": [
    {"description": "<detailed excellence description>", "title": "<abstract label>"}
  ],
  "negative_rubrics": [
    {"description": "<detailed failure description>", "title": "<abstract label>"}
  ]
}
```

## Examples

**Positive:**
```json
{"description": "Anticipates and addresses potential edge cases or exceptions to the main solution, demonstrating thorough problem understanding", "title": "Edge Case Handling"}
```

**Negative:**
```json
{"description": "Conflates correlation with causation when interpreting data or making recommendations", "title": "Causal Misattribution"}
```

## Inputs
1. **Question**: Original question being answered
2. **Responses**: Multiple model responses (Response 1, Response 2, etc.)
3. **Existing Rubrics** (optional): Previously generated/ground truth rubrics

## Critical Reminders
- Each rubric must distinguish between actual provided responses
- Exclude rubrics applying equally to all responses
- Prefer empty lists over redundancy when existing rubrics are comprehensive
- Focus on observable, objective, actionable criteria
- Quality over quantity: 2 excellent rubrics > 5 mediocre ones

Generate only the most impactful, non-redundant rubrics revealing meaningful quality differences.
"""


async def generate_instance_wise_adaptive_rubrics(
    question: str, response_list: list[str], existing_rubrics: str | None = None, model_name: str | None = None
) -> dict[str, Any] | None:
    """Generate adaptive rubrics for a single question based on multiple responses.

    Args:
        question: The question being answered
        response_list: List of model responses to analyze
        existing_rubrics: JSON string of existing rubrics to avoid duplication
        model_name: LLM model to use for rubric generation

    Returns:
        Dictionary with positive_rubrics and negative_rubrics, or None on failure
    """
    if model_name is None:
        model_name = os.environ.get("RUBRIC_GENERATION_MODEL", "gpt-4.1")

    prompt_suffix = f"Question: {question}\n\nResponses:\n"
    for i, response in enumerate(response_list):
        prompt_suffix += f"Response {i + 1}:\n{response}\n\n"

    if existing_rubrics:
        prompt_suffix += f"\n\nExisting Rubrics:\n{existing_rubrics}"

    prompt = INSTANCE_WISE_RUBRIC_GENERATION_PROMPT + prompt_suffix

    try:
        resp = await run_litellm_async(model_name=model_name, user_prompt=prompt)

        obj = extract_json_from_response(resp)
        logger.debug(f"Generated instance-wise adaptive rubrics: {obj}")
        return obj
    except Exception as e:
        logger.warning(f"Error generating instance-wise adaptive rubrics: {e}")
        return None


async def _generate_instance_wise_adaptive_rubrics(
    responses: list[str],
    ground_truths: list,
    num_samples_per_prompt_rollout: int,
    rubric_buffer: dict[str, Any] | None = None,
    use_full_responses: bool = True,
    answer_length_limit_in_words: int | None = None,
) -> tuple[list[dict[str, Any] | None], list[int]]:
    """Generate adaptive rubrics for all prompts in a batch.

    Args:
        responses: List of all responses (flattened across prompts)
        ground_truths: List of ground truth data for each response
        num_samples_per_prompt_rollout: Number of responses per prompt
        rubric_buffer: Optional buffer of existing rubrics per query
        use_full_responses: Whether to use full responses vs extracted answers
        answer_length_limit_in_words: Optional word limit for answer subsampling

    Returns:
        Tuple of (list of adaptive rubrics per prompt, list of subsampled answer counts)
    """
    ground_truths = [json.loads(ground_truth[0]) for ground_truth in ground_truths]

    num_prompts = len(responses) // num_samples_per_prompt_rollout

    query_key = "query" if "query" in ground_truths[0] else "Question"
    assert query_key in ground_truths[0], f"Query key {query_key} not found in ground truth"

    # Prepare all tasks for parallel execution
    tasks = []
    num_subsampled_answers_list = []
    for i in range(num_prompts):
        start_idx = i * num_samples_per_prompt_rollout
        end_idx = start_idx + num_samples_per_prompt_rollout

        # Get the question from the first ground truth in this group
        question = ground_truths[start_idx][query_key]
        if rubric_buffer is None:
            existing_rubrics = ground_truths[start_idx]["rubrics"]
            existing_rubrics_str = json.dumps(existing_rubrics)
        else:
            existing_rubrics = rubric_buffer[question]["active_rubrics"]
            existing_rubrics_str = json.dumps(existing_rubrics)

        # Get all responses for this question
        response_list = responses[start_idx:end_idx]

        # Create task for parallel execution
        if use_full_responses:
            num_subsampled_answers_list.append(len(response_list))
            task = generate_instance_wise_adaptive_rubrics(
                question,
                response_list,
                existing_rubrics_str,
                model_name=os.environ.get("RUBRIC_GENERATION_MODEL", "gpt-4.1"),
            )
        else:
            # Subsample answers if word limit is specified
            answer_list = response_list.copy()
            if answer_length_limit_in_words is not None:
                # Shuffle answers before selecting to get diverse subset
                random.shuffle(answer_list)
                # Select subset of answers that fits within word limit (minimum 2)
                selected_answers = []
                total_words = 0
                for answer in answer_list:
                    word_count = len(answer.split())
                    if len(selected_answers) < 2 or total_words + word_count <= answer_length_limit_in_words:
                        selected_answers.append(answer)
                        total_words += word_count
                    else:
                        break
                answer_list = selected_answers if selected_answers else answer_list[:2]
            num_subsampled_answers_list.append(len(answer_list))
            task = generate_instance_wise_adaptive_rubrics(
                question,
                answer_list,
                existing_rubrics_str,
                model_name=os.environ.get("RUBRIC_GENERATION_MODEL", "gpt-4.1"),
            )
        tasks.append(task)

    # Execute all tasks in parallel
    adaptive_rubrics = await asyncio.gather(*tasks)

    return adaptive_rubrics, num_subsampled_answers_list


def update_ground_truths_with_adaptive_rubrics(
    ground_truths: list,
    all_adaptive_rubrics: list[dict[str, Any] | None],
    num_samples_per_prompt_rollout: int,
    rubric_buffer: dict[str, Any] | None = None,
) -> tuple[list, float, float, float, float, dict[str, Any] | None, int]:
    """Update ground truths with newly generated adaptive rubrics.

    Args:
        ground_truths: List of ground truth data (may be wrapped in lists)
        all_adaptive_rubrics: List of adaptive rubrics per prompt
        num_samples_per_prompt_rollout: Number of responses per prompt
        rubric_buffer: Optional buffer to update with new rubrics

    Returns:
        Tuple of:
        - Updated ground truths
        - Valid adaptive rubric rate
        - Average number of ground truth rubrics
        - Average number of adaptive rubrics
        - Average number of active buffer rubrics
        - Updated rubric buffer
        - Count of skipped rubrics
    """
    valid_adaptive_rubric_rate = 0.0
    num_ground_truths = []
    num_adaptive_rubrics = []
    num_active_buffer_rubrics = []

    # Expand adaptive_rubrics to match ground_truths structure
    # Each adaptive rubric applies to num_samples_per_prompt_rollout ground truths
    expanded_adaptive_rubrics = []
    for rubric in all_adaptive_rubrics:
        for _ in range(num_samples_per_prompt_rollout):
            expanded_adaptive_rubrics.append(rubric)

    # Track processed queries to avoid duplicate buffer updates
    processed_queries = set()
    skipped_count = 0

    for i, (ground_truth, adaptive_rubrics) in enumerate(zip(ground_truths, expanded_adaptive_rubrics)):
        if adaptive_rubrics is None:
            skipped_count += 1
            continue

        # Handle the case where ground_truth is wrapped in a list
        is_wrapped_in_list = isinstance(ground_truth, list)
        ground_truth_str = ground_truth[0] if is_wrapped_in_list else ground_truth

        ground_truth_obj = json.loads(ground_truth_str)
        query = ground_truth_obj["query"]

        positive_rubrics = adaptive_rubrics.get("positive_rubrics", [])
        negative_rubrics = adaptive_rubrics.get("negative_rubrics", [])

        num_ground_truths.append(len(ground_truth_obj["rubrics"]))
        num_adaptive_rubrics.append(len(positive_rubrics) + len(negative_rubrics))

        # Update rubric buffer with newly generated adaptive rubrics (only once per query)
        if rubric_buffer is not None and query in rubric_buffer and query not in processed_queries:
            logger.debug(
                f"Updating rubric buffer for query {query}; "
                f"before update: {len(rubric_buffer[query]['active_rubrics'])} active, "
                f"{len(rubric_buffer[query]['inactive_rubrics'])} inactive"
            )
            # Convert new adaptive rubrics to the buffer format
            new_active_rubrics = []
            for rubric in positive_rubrics:
                new_active_rubrics.append(
                    {"description": rubric["description"], "weight": 1.0, "title": rubric["title"]}
                )
            for rubric in negative_rubrics:
                new_active_rubrics.append(
                    {"description": rubric["description"], "weight": -1.0, "title": rubric["title"]}
                )

            # Append new rubrics to active_rubrics in buffer
            rubric_buffer[query]["active_rubrics"].extend(new_active_rubrics)
            num_active_buffer_rubrics.append(len(rubric_buffer[query]["active_rubrics"]))
            processed_queries.add(query)  # Mark this query as processed

        # Always use rubrics from buffer if available (for all rollouts of this query)
        if rubric_buffer is not None and query in rubric_buffer:
            # Keep original rubrics and append active rubrics from buffer
            ground_truth_obj["rubrics"] = (
                rubric_buffer[query]["persistent_rubrics"] + rubric_buffer[query]["active_rubrics"]
            )
            ground_truth_obj["rubrics_types"] = ["persistent"] * len(rubric_buffer[query]["persistent_rubrics"]) + [
                "adaptive"
            ] * len(rubric_buffer[query]["active_rubrics"])
        else:
            logger.debug(f"No buffer found for query {query}, using newly generated rubrics")
            # Keep original rubrics and append newly generated adaptive rubrics
            original_rubrics = ground_truth_obj["rubrics"].copy()
            additional_rubrics = []
            for rubric in positive_rubrics:
                additional_rubrics.append(
                    {"description": rubric["description"], "weight": 1.0, "title": rubric["title"]}
                )
            for rubric in negative_rubrics:
                additional_rubrics.append(
                    {"description": rubric["description"], "weight": -1.0, "title": rubric["title"]}
                )
            ground_truth_obj["rubrics"] = original_rubrics + additional_rubrics
            ground_truth_obj["rubrics_types"] = ["persistent"] * len(original_rubrics) + ["adaptive"] * len(
                additional_rubrics
            )

        # Convert back to JSON string and update the original list
        updated_ground_truth_str = json.dumps(ground_truth_obj)
        if is_wrapped_in_list:
            ground_truths[i] = [updated_ground_truth_str]
        else:
            ground_truths[i] = updated_ground_truth_str

        valid_adaptive_rubric_rate += 1.0

    # Log warning if all adaptive rubrics were skipped (likely due to generation failures)
    if skipped_count > 0:
        logger.warning(
            f"Skipped {skipped_count}/{len(expanded_adaptive_rubrics)} adaptive rubrics "
            "(None values - likely JSON parsing failures)"
        )

    # Handle empty lists to avoid division by zero
    valid_adaptive_rubric_rate = valid_adaptive_rubric_rate / len(ground_truths) if len(ground_truths) > 0 else 0.0
    avg_num_ground_truths = sum(num_ground_truths) / len(num_ground_truths) if len(num_ground_truths) > 0 else 0.0
    avg_num_adaptive_rubrics = (
        sum(num_adaptive_rubrics) / len(num_adaptive_rubrics) if len(num_adaptive_rubrics) > 0 else 0.0
    )
    avg_num_active_buffer_rubrics = (
        sum(num_active_buffer_rubrics) / len(num_active_buffer_rubrics) if num_active_buffer_rubrics else 0.0
    )
    return (
        ground_truths,
        valid_adaptive_rubric_rate,
        avg_num_ground_truths,
        avg_num_adaptive_rubrics,
        avg_num_active_buffer_rubrics,
        rubric_buffer,
        skipped_count,
    )


def save_adaptive_rubric_cache_safe(
    cache_dir: str,
    training_step: int,
    decoded_responses: list[str],
    ground_truths: list,
    all_adaptive_rubrics: list,
    num_subsampled_answers_list: list[int],
    num_samples_per_prompt_rollout: int,
    use_full_responses: bool,
    answer_length_limit_in_words: int | None,
) -> str:
    """Safely save adaptive rubric generation inputs and outputs for future training use.

    Uses atomic writes and file locking for multi-thread/multi-node safety.

    Args:
        cache_dir: Directory to save cache files
        training_step: Current training step number
        decoded_responses: List of decoded response strings (inputs)
        ground_truths: List of ground truth data (inputs)
        all_adaptive_rubrics: List of generated adaptive rubrics (outputs)
        num_subsampled_answers_list: List of subsampled answer counts (outputs)
        num_samples_per_prompt_rollout: Number of samples per prompt (config)
        use_full_responses: Whether full responses were used (config)
        answer_length_limit_in_words: Word limit for answers (config)

    Returns:
        Path to the saved cache file
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Create unique identifier for this save operation
    hostname = socket.gethostname()
    pid = os.getpid()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]

    # Prepare cache data
    cache_data = {
        "training_step": training_step,
        "timestamp": timestamp,
        "hostname": hostname,
        "pid": pid,
        # Inputs to _generate_instance_wise_adaptive_rubrics
        "inputs": {
            "decoded_responses": decoded_responses,
            "ground_truths": ground_truths,
            "num_samples_per_prompt_rollout": num_samples_per_prompt_rollout,
            "use_full_responses": use_full_responses,
            "answer_length_limit_in_words": answer_length_limit_in_words,
        },
        # Outputs from _generate_instance_wise_adaptive_rubrics
        "outputs": {
            "all_adaptive_rubrics": all_adaptive_rubrics,
            "num_subsampled_answers_list": num_subsampled_answers_list,
        },
    }

    # Generate final filename
    final_filename = f"adaptive_rubric_cache_step{training_step}_{hostname}_{pid}_{timestamp}_{unique_id}.json"
    final_path = os.path.join(cache_dir, final_filename)

    # Use atomic write: write to temp file, then rename
    fd, temp_path = tempfile.mkstemp(suffix=".tmp", prefix="adaptive_rubric_cache_", dir=cache_dir)

    try:
        # Write data to temp file with file locking
        with os.fdopen(fd, "w") as f:
            # Acquire exclusive lock for writing
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(cache_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Atomic rename (guaranteed atomic on POSIX systems within same filesystem)
        os.rename(temp_path, final_path)
        logger.info(f"Saved adaptive rubric cache to: {final_path}")
        return final_path

    except Exception as e:
        # Clean up temp file if something went wrong
        if os.path.exists(temp_path):
            with contextlib.suppress(OSError):
                os.remove(temp_path)
        logger.error(f"Failed to save adaptive rubric cache: {e}")
        raise


def initialize_rubric_buffer(ground_truths: list, use_static_rubrics_as_persistent: bool = True) -> dict[str, Any]:
    """Initialize rubric buffer from ground truths.

    Args:
        ground_truths: List of ground truth data containing queries and rubrics
        use_static_rubrics_as_persistent: If True, static rubrics become persistent;
            if False, static rubrics become initial active rubrics

    Returns:
        Dictionary mapping query strings to rubric buffer entries
    """
    rubric_buffer: dict[str, Any] = {}

    for gt in ground_truths:
        if isinstance(gt, list):
            gt = gt[0]
        if isinstance(gt, str):
            gt = json.loads(gt)

        query = gt.get("query", gt.get("Question"))
        if query is None:
            continue

        if query not in rubric_buffer:
            rubrics = gt.get("rubrics", [])
            if use_static_rubrics_as_persistent:
                rubric_buffer[query] = {
                    "active_rubrics": [],
                    "inactive_rubrics": [],
                    "persistent_rubrics": rubrics,  # Static rubrics used every time
                }
            else:
                rubric_buffer[query] = {
                    "active_rubrics": rubrics,  # Static rubrics start as active
                    "inactive_rubrics": [],
                    "persistent_rubrics": [],
                }

    return rubric_buffer
