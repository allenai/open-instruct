import re
from open_instruct.search_rewards.utils.format_utils import extract_search_tool_calls
from typing import Optional


def score_num_in_context_search_turns(context: str, upper_bound: int = 3, mcp_parser_name: Optional[str] = None) -> float:
    """
    Score the number of search turns in the response.
    This function extracts all strings wrapped within <search> and </search> tags,
    then computes the number of valid (non-empty) extracted queries.
    """
    if not context:
        return 0.0, 0

    queries = extract_search_tool_calls(context, mcp_parser_name=mcp_parser_name)

    # A valid query must not be empty or contain only whitespace.
    num_valid_queries = sum(1 for q in queries)

    return min(float(num_valid_queries) / upper_bound, 1.0), num_valid_queries


def score_query_redundancy(context: str, mcp_parser_name: Optional[str] = None) -> float:
    """
    Score the redundancy of a query.
    Score is between 0 and 1, which is unique_queries / total_queries
    """
    queries = extract_search_tool_calls(context, mcp_parser_name=mcp_parser_name)
    
    return 1.0 if len(queries) == 0 else len(set(queries)) / len(queries)


def score_query_quality_in_a_row(context: str, mcp_parser_name: Optional[str] = None) -> float:
    """
    Score the quality of a query.
    """
    # TODO
    pass
