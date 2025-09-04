import re
from open_instruct.search_rewards.utils.format_utils import extract_search_tool_calls


def score_num_in_context_search_turns(context: str, upper_bound: int = 3) -> float:
    """
    Score the number of search turns in the response.
    This function extracts all strings wrapped within <search> and </search> tags,
    then computes the number of valid (non-empty) extracted queries.
    """
    if not context:
        return 0.0, 0

    # Use re.findall to extract all substrings within <search> tags
    # The re.DOTALL flag allows '.' to match newline characters, in case a query spans multiple lines.
    queries = re.findall(r"<search>(.*?)</search>", context, re.DOTALL)

    # A valid query must not be empty or contain only whitespace.
    num_valid_queries = sum(1 for q in queries if q.strip())

    return min(float(num_valid_queries) / upper_bound, 1.0), num_valid_queries


def score_query_redundancy(context: str) -> float:
    """
    Score the redundancy of a query.
    Score is between 0 and 1, which is unique_queries / total_queries
    """
    queries = extract_search_tool_calls(context)
    
    return 1.0 if len(queries) == 0 else len(set(queries)) / len(queries)


def score_query_quality_in_a_row(context: str) -> float:
    """
    Score the quality of a query.
    """
    # TODO
    pass
