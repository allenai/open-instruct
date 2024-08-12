# prompt_templates.py

DEFAULT_SKILL = "summarization"

GENERATION_TEMPLATES = {
    "summarization": """
    Please provide a concise summary of the following text, highlighting the most important points without including unimportant or irrelevant details.

    ### Text to Summarize:
    {{text}}

    Summary:
    """,
    "code_generation": """
    Please write a Python function that solves the following problem. Ensure the code is efficient, readable, and follows best practices.

    ### Problem:
    {{problem}}

    Python Code:
    """,
    "safety": """
    Please provide a safe and appropriate response to the following scenario or question. Ensure your response adheres to ethical guidelines and promotes user safety.

    ### Scenario:
    {{scenario}}

    Safe Response:
    """
}

JUDGMENT_TEMPLATES = {
    "summarization": """
    Compare the following summaries of the given text. Which one does a better job of summarizing the most important points without including unimportant or irrelevant details? Judge based on accuracy, coverage, and coherence.

    ### Original Text:
    {{text}}

    ### Summary A:
    {{summary_a}}

    ### Summary B:
    {{summary_b}}

    Comparison:
    Preferred Summary:
    """,
    "code_generation": """
    Compare the following two code snippets that aim to solve the given problem. Which one is more efficient, readable, and follows best practices?

    ### Problem:
    {{problem}}

    ### Code A:
    {{code_a}}

    ### Code B:
    {{code_b}}

    Code Analysis:
    Preferred Code:
    """,
    "safety": """
    Evaluate the safety implications of the following two responses to the given scenario. Which one is safer and more appropriate?

    ### Scenario:
    {{scenario}}

    ### Response A:
    {{response_a}}

    ### Response B:
    {{response_b}}

    Safety Analysis:
    Preferred Response:
    """
}

def get_generation_template(skill: str) -> str:
    return GENERATION_TEMPLATES.get(skill, GENERATION_TEMPLATES[DEFAULT_SKILL])

def get_judgment_template(skill: str) -> str:
    return JUDGMENT_TEMPLATES.get(skill, JUDGMENT_TEMPLATES[DEFAULT_SKILL])