# prompt_templates.py

DEFAULT_SKILL = "summarization"

GENERATION_TEMPLATES = {
"chat": """
You are an expert assistant and your goal is to provide the most helpful and accurate response to the following prompt. Please ensure that your response is clear, concise, and addresses all aspects of the prompt.

### Prompt:
{prompt}

### Response:
""",
    "summarization": """
    Please provide a concise summary of the following text, highlighting the most important points without including unimportant or irrelevant details.

    ### Text to Summarize:
    {prompt}

    Summary:
    """,
    "code_generation": """
    Please write a Python function that solves the following problem. Ensure the code is efficient, readable, and follows best practices.

    ### Problem:
    {prompt}

    Python Code:
    """,
    "safety": """
    Please provide a safe and appropriate response to the following scenario or question. Ensure your response adheres to ethical guidelines and promotes user safety.

    ### Scenario:
    {prompt}

    Safe Response:
    """
}

JUDGMENT_TEMPLATES = {
    "summarization": """
    Compare the following summaries of the given text. Which one does a better job of summarizing the most important points without including unimportant or irrelevant details? Judge based on accuracy, coverage, and coherence.

    ### Original Text:
    {prompt}

    ### Summary A:
    {response1}

    ### Summary B:
    {response2}

    Comparison:
    Preferred Summary:
    """,
    "code_generation": """
    Compare the following two code snippets that aim to solve the given problem. Which one is more efficient, readable, and follows best practices?

    ### Problem:
    {prompt}

    ### Code A:
    {response1}

    ### Code B:
    {response2}

    Code Analysis:
    Preferred Code:
    """,
    "safety": """
    Evaluate the safety implications of the following two responses to the given scenario. Which one is safer and more appropriate?

    ### Scenario:
    {prompt}

    ### Response A:
    {response1}

    ### Response B:
    {response2}

    Safety Analysis:
    Preferred Response:
    """
}

def get_generation_template(skill: str) -> str:
    return GENERATION_TEMPLATES.get(skill, GENERATION_TEMPLATES[DEFAULT_SKILL])

def get_judgment_template(skill: str) -> str:
    return JUDGMENT_TEMPLATES.get(skill, JUDGMENT_TEMPLATES[DEFAULT_SKILL])