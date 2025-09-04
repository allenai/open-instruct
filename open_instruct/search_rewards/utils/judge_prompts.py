from typing import Dict, Any

HLE_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""

def extract_hle_judge_response_from_response(response: str) -> Dict[str, Any]:
    """
    Extract reward score from judge response.
    The response format is:
    extracted_final_answer: ...
    reasoning: ...
    correct: yes/no
    confidence: ...
    
    Returns:
        Dictionary with 'reward' field: 1.0 if correct is 'yes', 0.0 if 'no'
        Returns None if parsing fails
    """
    if not response or not isinstance(response, str):
        return None
    
    lines = response.strip().split('\n')
    reward = 0.0
    found_correct = False
    
    for line in lines:
        # Handle different possible formats and whitespace
        line = line.strip()
        if not line:
            continue
            
        # Try different possible formats for the correct field
        if line.lower().startswith('correct:'):
            parts = line.split(':', 1)
            if len(parts) >= 2:
                correct_value = parts[1].strip().lower()
                if correct_value in ['yes', 'true', '1', 'correct']:
                    reward = 1.0
                elif correct_value in ['no', 'false', '0', 'incorrect']:
                    reward = 0.0
                else:
                    # If we can't parse the value, default to 0.0
                    reward = 0.0
                found_correct = True
                break
        elif line.lower().startswith('correct'):
            # Handle cases where there might be extra whitespace or formatting
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    correct_value = parts[1].strip().lower()
                    if correct_value in ['yes', 'true', '1', 'correct']:
                        reward = 1.0
                    elif correct_value in ['no', 'false', '0', 'incorrect']:
                        reward = 0.0
                    else:
                        reward = 0.0
                    found_correct = True
                    break
    
    # If we didn't find a 'correct' field, try to infer from the response
    if not found_correct:
        # Look for positive/negative indicators in the reasoning
        response_lower = response.lower()
        positive_indicators = ['correct', 'accurate', 'matches', 'equivalent', 'yes']
        negative_indicators = ['incorrect', 'wrong', 'does not match', 'inconsistent', 'no']
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in response_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in response_lower)
        
        if positive_count > negative_count:
            reward = 1.0
        else:
            reward = 0.0
    
    return {"reward": reward}


HLE_JUDGE_SCORE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match and the fraction of the statement that is correct.

score: return a score between 0 and 10 based on the fraction of the statement that is correct.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


HLE_JUDGE_SCORE_NO_REASONING_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

You must directly return a score between 0 and 10 based on the fraction of the statement that is correct.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


def extract_hle_score_judge_response_from_response(response: str) -> Dict[str, Any]:
    """
    Extract reward score from judge response.
    The response format is:
    extracted_final_answer: ...
    reasoning: ...
    score: 0-10
    confidence: ...
    
    Returns:
        Dictionary with 'reward' field: normalized score between 0.0 and 1.0
        Returns None if parsing fails
    """
    if not response or not isinstance(response, str):
        return None
    
    lines = response.strip().split('\n')
    reward = 0.0
    found_score = False
    
    for line in lines:
        # Handle different possible formats and whitespace
        line = line.strip()
        if not line:
            continue
            
        # Try different possible formats for the score field
        if line.lower().startswith('score:'):
            parts = line.split(':', 1)
            if len(parts) >= 2:
                try:
                    score_value = float(parts[1].strip())
                    # Normalize score from 0-10 to 0-1 range
                    reward = max(0.0, min(1.0, score_value / 10.0))
                    found_score = True
                    break
                except (ValueError, TypeError):
                    # If we can't parse the value, default to 0.0
                    reward = 0.0
                    found_score = True
                    break
        elif line.lower().startswith('score'):
            # Handle cases where there might be extra whitespace or formatting
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    try:
                        score_value = float(parts[1].strip())
                        # Normalize score from 0-10 to 0-1 range
                        reward = max(0.0, min(1.0, score_value / 10.0))
                        found_score = True
                        break
                    except (ValueError, TypeError):
                        reward = 0.0
                        found_score = True
                        break
    
    # If we didn't find a 'score' field, try to infer from the response
    if not found_score:
        # Look for positive/negative indicators in the reasoning
        response_lower = response.lower()
        positive_indicators = ['correct', 'accurate', 'matches', 'equivalent', 'yes', 'good', 'appropriate']
        negative_indicators = ['incorrect', 'wrong', 'does not match', 'inconsistent', 'no', 'bad', 'inappropriate']
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in response_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in response_lower)
        
        if positive_count > negative_count:
            reward = 0.7  # Default positive score
        else:
            reward = 0.3  # Default negative score
    
    return {"reward": reward}