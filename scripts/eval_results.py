#!/usr/bin/env python3
"""
Evaluate inference results from tool-use models.

Supports:
- AIME: Math verification using math-verify with majority voting
- SimpleQA: LLM-as-judge grading (OpenAI style)
- GPQA: Simple letter comparison

Usage:
    python scripts/eval_results.py --results_dir /path/to/results
    python scripts/eval_results.py --aime_file results/aime_results.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# For math verification
try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: math-verify not installed. Install with: pip install math-verify")


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...}, handling nested braces."""
    # Find \boxed{ and then match braces
    match = re.search(r'\\boxed\s*\{', text)
    if not match:
        return None
    
    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    
    if depth == 0:
        return text[start:i-1].strip()
    return None


def extract_letter_answer(text: str) -> str | None:
    """Extract letter answer (A, B, C, D) from response."""
    # Look for patterns like "The answer is A", "Answer: B", "(C)", etc.
    patterns = [
        r'(?:the\s+)?answer\s+is\s*[:\s]*([A-D])',
        r'(?:^|\s)([A-D])\s*[\.\):]?\s*$',
        r'\*\*([A-D])\*\*',
        r'\\boxed\{([A-D])\}',
        r'(?:choose|select|pick)\s+([A-D])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    
    # Last resort: find the last standalone letter
    matches = re.findall(r'(?:^|\s)([A-D])(?:\s|$|\.|\))', text, re.MULTILINE)
    if matches:
        return matches[-1].upper()
    
    return None


def eval_aime(results: list[dict]) -> dict:
    """Evaluate AIME results using math-verify with majority voting."""
    if not MATH_VERIFY_AVAILABLE:
        print("Skipping AIME eval: math-verify not installed")
        return {"error": "math-verify not installed"}
    
    correct = 0
    total = 0
    details = []
    
    for sample in results:
        total += 1
        ground_truth = sample.get("answer", sample.get("solution", ""))
        
        # Handle multi-sample (majority voting)
        if "generated_responses" in sample:
            responses = sample["generated_responses"]
            extracted_answers = []
            for resp in responses:
                ans = extract_boxed(resp)
                if ans:
                    extracted_answers.append(ans)
            
            # Majority vote
            if extracted_answers:
                counter = Counter(extracted_answers)
                predicted, count = counter.most_common(1)[0]
            else:
                predicted = None
                count = 0
        else:
            resp = sample.get("generated_response", "")
            predicted = extract_boxed(resp)
            count = 1
        
        # Verify using math-verify
        is_correct = False
        if predicted is not None:
            try:
                pred_parsed = parse(predicted)
                gt_parsed = parse(str(ground_truth))
                is_correct = verify(pred_parsed, gt_parsed)
            except Exception as e:
                # Fallback to string comparison
                is_correct = str(predicted).strip() == str(ground_truth).strip()
        
        if is_correct:
            correct += 1
        
        details.append({
            "id": sample.get("id", total - 1),
            "ground_truth": ground_truth,
            "predicted": predicted,
            "vote_count": count if "generated_responses" in sample else None,
            "correct": is_correct,
        })
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nAIME Results:")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": details,
    }


# SimpleQA grading template from OpenAI
SIMPLEQA_GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.

The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target.

The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()


def eval_simpleqa(results: list[dict], grader_model: str = "gpt-4o-mini") -> dict:
    """Evaluate SimpleQA results using LLM-as-judge."""
    try:
        from openai import OpenAI
        client = OpenAI()
    except ImportError:
        print("Skipping SimpleQA eval: openai not installed")
        return {"error": "openai not installed"}
    except Exception as e:
        print(f"Skipping SimpleQA eval: {e}")
        return {"error": str(e)}
    
    grades = {"CORRECT": 0, "INCORRECT": 0, "NOT_ATTEMPTED": 0}
    details = []
    
    for i, sample in enumerate(results):
        question = sample.get("problem", "")
        ground_truth = sample.get("answer", "")
        predicted = sample.get("generated_response", "")
        
        # Grade using LLM
        prompt = SIMPLEQA_GRADER_TEMPLATE.format(
            question=question,
            target=ground_truth,
            predicted_answer=predicted,
        )
        
        try:
            response = client.chat.completions.create(
                model=grader_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            grade_letter = response.choices[0].message.content.strip()
            
            grade_map = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}
            grade = grade_map.get(grade_letter, "NOT_ATTEMPTED")
        except Exception as e:
            print(f"  Error grading sample {i}: {e}")
            grade = "NOT_ATTEMPTED"
        
        grades[grade] += 1
        details.append({
            "id": sample.get("id", i),
            "question": question[:100] + "..." if len(question) > 100 else question,
            "ground_truth": ground_truth,
            "grade": grade,
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Graded {i + 1}/{len(results)} samples...")
    
    total = len(results)
    accuracy = grades["CORRECT"] / total if total > 0 else 0
    attempted = grades["CORRECT"] + grades["INCORRECT"]
    accuracy_given_attempted = grades["CORRECT"] / attempted if attempted > 0 else 0
    
    print(f"\nSimpleQA Results:")
    print(f"  Accuracy: {accuracy:.2%} ({grades['CORRECT']}/{total})")
    print(f"  Accuracy (given attempted): {accuracy_given_attempted:.2%}")
    print(f"  Correct: {grades['CORRECT']}, Incorrect: {grades['INCORRECT']}, Not Attempted: {grades['NOT_ATTEMPTED']}")
    
    return {
        "accuracy": accuracy,
        "accuracy_given_attempted": accuracy_given_attempted,
        "correct": grades["CORRECT"],
        "incorrect": grades["INCORRECT"],
        "not_attempted": grades["NOT_ATTEMPTED"],
        "total": total,
        "details": details,
    }


def eval_gpqa(results: list[dict]) -> dict:
    """Evaluate GPQA results by comparing letter answers."""
    correct = 0
    total = 0
    details = []
    
    for sample in results:
        total += 1
        ground_truth = sample.get("answer", "")  # Should be A, B, C, or D
        predicted_text = sample.get("generated_response", "")
        
        predicted = extract_letter_answer(predicted_text)
        is_correct = predicted is not None and predicted.upper() == ground_truth.upper()
        
        if is_correct:
            correct += 1
        
        details.append({
            "id": sample.get("id", total - 1),
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
        })
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nGPQA Results:")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate inference results")
    parser.add_argument("--results_dir", type=str, help="Directory containing result files")
    parser.add_argument("--aime_file", type=str, help="AIME results JSONL file")
    parser.add_argument("--simpleqa_file", type=str, help="SimpleQA results JSONL file")
    parser.add_argument("--gpqa_file", type=str, help="GPQA results JSONL file")
    parser.add_argument("--grader_model", type=str, default="gpt-4o-mini", help="Model for SimpleQA grading")
    parser.add_argument("--output_file", type=str, help="Output file for detailed results (JSON)")
    args = parser.parse_args()
    
    all_results = {}
    
    # Find files
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not args.aime_file and (results_dir / "aime_results.jsonl").exists():
            args.aime_file = str(results_dir / "aime_results.jsonl")
        if not args.simpleqa_file and (results_dir / "simpleqa_results.jsonl").exists():
            args.simpleqa_file = str(results_dir / "simpleqa_results.jsonl")
        if not args.gpqa_file and (results_dir / "gpqa_results.jsonl").exists():
            args.gpqa_file = str(results_dir / "gpqa_results.jsonl")
    
    # Evaluate AIME
    if args.aime_file:
        print(f"Evaluating AIME: {args.aime_file}")
        results = load_jsonl(args.aime_file)
        all_results["aime"] = eval_aime(results)
    
    # Evaluate SimpleQA
    if args.simpleqa_file:
        print(f"Evaluating SimpleQA: {args.simpleqa_file}")
        results = load_jsonl(args.simpleqa_file)
        all_results["simpleqa"] = eval_simpleqa(results, args.grader_model)
    
    # Evaluate GPQA
    if args.gpqa_file:
        print(f"Evaluating GPQA: {args.gpqa_file}")
        results = load_jsonl(args.gpqa_file)
        all_results["gpqa"] = eval_gpqa(results)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for dataset, result in all_results.items():
        if "error" not in result:
            print(f"{dataset.upper()}: {result['accuracy']:.2%}")
    
    # Save detailed results
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
