#!/usr/bin/env python3

import json
import logging
import os
import sys
from typing import Dict, List, Any
from pathlib import Path
import numpy as np
from scipy import stats

# Add the current directory to the path for imports
# We're already in the open_instruct root directory
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

from open_instruct.search_rewards.reasoning_model_rewards import compute_hle_reward
from open_instruct.search_rewards.openscholar_rewards import compute_paper_reward

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_generated_answers(file_path: str) -> List[Dict[str, Any]]:
    """
    Load generated answers from JSONL file.
    
    Args:
        file_path: Path to the JSONL file containing generated answers
        
    Returns:
        List of dictionaries containing question and answers
    """
    answers_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                answers_data.append(data)
    
    return answers_data


def load_test_cases(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load test cases from JSON file.
    """
    all_test_cases = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_test_cases = json.load(f)
        for test_case in raw_test_cases:
            all_test_cases[test_case['initial_prompt']] = test_case
    return all_test_cases


def create_correct_answer_for_question(question: str) -> str:
    """
    Create a simple correct answer template for the given question.
    This is a placeholder - in a real scenario, you would have ground truth answers.
    
    Args:
        question: The question to create an answer for
        
    Returns:
        A template correct answer
    """
    # This is a simple template - in practice, you'd have actual correct answers
    return f"Correct answer for: {question}"


def load_existing_evaluations(file_path: str) -> set:
    """
    Load existing evaluation results to avoid duplicate processing.
    
    Args:
        file_path: Path to the JSONL file containing existing evaluations
        
    Returns:
        Set of questions that have already been evaluated
    """
    evaluated_questions = set()
    
    if not os.path.exists(file_path):
        return evaluated_questions
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    question = data.get('question', '')
                    if question:
                        evaluated_questions.add(question)
    except Exception as e:
        print(f"Warning: Could not load existing evaluations: {e}")
    
    return evaluated_questions


def load_reference_answers(file_path: str) -> Dict[str, str]:
    """
    Load reference answers from JSONL file.
    
    Args:
        file_path: Path to the JSONL file containing reference answers
        
    Returns:
        Dictionary mapping questions to reference answers
    """
    reference_answers = {}
    
    if not os.path.exists(file_path):
        print(f"Warning: Reference answers file not found at {file_path}")
        return reference_answers
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    question = data.get('question', '')
                    comprehensive_answer = data.get('comprehensive_answer', '')
                    if question and comprehensive_answer:
                        reference_answers[question] = comprehensive_answer
        
        print(f"Loaded {len(reference_answers)} reference answers from {file_path}")
    except Exception as e:
        print(f"Error loading reference answers: {e}")
    
    return reference_answers


def evaluate_answers_with_hle(answers_data: List[Dict[str, Any]], 
                             reference_answers: Dict[str, str] = None,
                             skip_evaluated: bool = True,
                             evaluated_questions: set = None,
                             no_reasoning: bool = False,
                             allow_missing_question: bool = False) -> List[Dict[str, Any]]:
    """
    Evaluate all answers using the HLE reward system.
    
    Args:
        answers_data: List of dictionaries containing questions and answers
        reference_answers: Dictionary mapping questions to reference answers (used as correct answers)
        skip_evaluated: Whether to skip questions that have already been evaluated
        evaluated_questions: Set of questions that have already been evaluated
        
    Returns:
        List of evaluation results
    """
    if reference_answers is None:
        reference_answers = {}
    
    if evaluated_questions is None:
        evaluated_questions = set()
    
    evaluation_results = []
    
    for entry in answers_data:
        question = entry.get('question', 'Unknown question')
        if not allow_missing_question:
            assert question in reference_answers, f"Question {question} not found in reference answers"
        
        # Skip if already evaluated
        if skip_evaluated and question in evaluated_questions:
            print(f"Skipping already evaluated question: {question[:50]}...")
            continue
        
        all_answers = entry.get('all_answers', [])
        
        # Get correct answer for this question (use reference answer if available, otherwise create template)
        correct_answer = reference_answers.get(question, create_correct_answer_for_question(question))
        
        entry_results = {
            'question': question,
            'correct_answer': correct_answer,
            'has_reference_answer': question in reference_answers,
            'answer_evaluations': []
        }
        
        for answer_info in all_answers:
            num_rubrics = answer_info.get('num_rubrics_provided', 0)
            answer_text = answer_info.get('answer', '')
            
            print(f"Evaluating question: {question[:50]}...")
            print(f"  Number of rubrics: {num_rubrics}")
            print(f"  Answer length: {len(answer_text)} characters")
            print(f"  Reference answer available: {'Yes' if question in reference_answers else 'No'}")
            
            try:
                # Score the answer using HLE reward
                result = compute_hle_reward('<answer>'+answer_text+'</answer>', correct_answer, question, no_reasoning)
                
                evaluation = {
                    'num_rubrics_provided': num_rubrics,
                    'answer_length': len(answer_text),
                    'hle_score': result.get('reward', 0.0),
                    'extraction_success': result.get('extraction_success', False),
                    'scoring_results': result.get('scoring_results', {}),
                    'error': result.get('error'),
                    'citations': result.get('citations', {}),
                    'has_reference_answer': question in reference_answers
                }
                
                print(f"  HLE Score: {evaluation['hle_score']:.4f}")
                if evaluation['error']:
                    print(f"  Error: {evaluation['error']}")
                
            except Exception as e:
                print(f"  Error evaluating answer: {e}")
                breakpoint()
                evaluation = {
                    'num_rubrics_provided': num_rubrics,
                    'answer_length': len(answer_text),
                    'hle_score': 0.0,
                    'extraction_success': False,
                    'scoring_results': {},
                    'error': str(e),
                    'citations': {},
                    'has_reference_answer': question in reference_answers
                }
            
            entry_results['answer_evaluations'].append(evaluation)
        
        evaluation_results.append(entry_results)
        print("-" * 80)
    
    return evaluation_results


def evaluate_answers_with_rubric_reward(answers_data: List[Dict[str, Any]], 
                                      test_cases_path: str = "./open_instruct/search_rewards/data/test_configs_snippets.json",
                                      skip_evaluated: bool = True,
                                      evaluated_questions: set = None,
                                      allow_missing_question: bool = False) -> List[Dict[str, Any]]:
    """
    Evaluate all answers using the paper reward system, focusing only on properties score.
    
    Args:
        answers_data: List of dictionaries containing questions and answers
        test_cases_path: Path to the test case configurations
        skip_evaluated: Whether to skip questions that have already been evaluated
        evaluated_questions: Set of questions that have already been evaluated
        
    Returns:
        List of evaluation results
    """
    test_cases = load_test_cases(test_cases_path)
    
    if evaluated_questions is None:
        evaluated_questions = set()
    
    evaluation_results = []
    
    for entry in answers_data:
        question = entry.get('question', 'Unknown question')
        if not allow_missing_question:
            assert question in test_cases, f"Question {question} not found in test cases"
        
        # Skip if already evaluated
        if skip_evaluated and question in evaluated_questions:
            print(f"Skipping already evaluated question: {question[:50]}...")
            continue
        
        all_answers = entry.get('all_answers', [])
        
        # Get test case for this question
        test_case = test_cases.get(question, {})
        
        entry_results = {
            'question': question,
            'test_case': test_case,
            'answer_evaluations': []
        }
        
        for answer_info in all_answers:
            num_rubrics = answer_info.get('num_rubrics_provided', 0)
            answer_text = answer_info.get('answer', '')
            
            print(f"Evaluating question: {question[:50]}...")
            print(f"  Number of rubrics: {num_rubrics}")
            print(f"  Answer length: {len(answer_text)} characters")
            
            try:
                # Score the answer using paper reward
                result = compute_paper_reward('<answer>'+answer_text+'</answer>', test_case)
                
                # Extract only the properties scores from scoring_results
                scoring_results = result.get('scoring_results', {})
                properties_scores = {}
                
                if scoring_results:
                    # Get the other_properties from the test case config
                    other_properties = test_case.get('metric_config', {}).get('config', {}).get('other_properties', [])
                    
                    for prop in other_properties:
                        prop_name = prop.get('name', '')
                        if prop_name in scoring_results:
                            properties_scores[prop_name] = scoring_results[prop_name]
                        if f"{prop_name}_evidence" in scoring_results:
                            properties_scores[f"{prop_name}_evidence"] = scoring_results[f"{prop_name}_evidence"]
                
                # Calculate properties-only score (average of all property scores)
                properties_score = 0.0
                if properties_scores:
                    properties_score = sum(properties_scores.values()) / len(properties_scores)
                
                evaluation = {
                    'num_rubrics_provided': num_rubrics,
                    'answer_length': len(answer_text),
                    'paper_score': result.get('reward', 0.0),
                    'properties_score': properties_score,
                    'properties_scores': properties_scores,
                    'extraction_success': result.get('extraction_success', False),
                    'scoring_results': scoring_results,
                    'error': result.get('error'),
                    'citations': result.get('citations', {}),
                }
                
                print(f"  Paper Score: {evaluation['paper_score']:.4f}")
                print(f"  Properties Score: {evaluation['properties_score']:.4f}")
                if evaluation['error']:
                    print(f"  Error: {evaluation['error']}")
                
            except Exception as e:
                print(f"  Error evaluating answer: {e}")
                evaluation = {
                    'num_rubrics_provided': num_rubrics,
                    'answer_length': len(answer_text),
                    'paper_score': 0.0,
                    'properties_score': 0.0,
                    'properties_scores': {},
                    'extraction_success': False,
                    'scoring_results': {},
                    'error': str(e),
                    'citations': {},
                }
            
            entry_results['answer_evaluations'].append(evaluation)
        
        evaluation_results.append(entry_results)
        print("-" * 80)
    
    return evaluation_results


def save_evaluation_results(results: List[Dict[str, Any]], output_file: str):
    """
    Save evaluation results to a JSONL file in appending mode.
    
    Args:
        results: List of evaluation results
        output_file: Path to save the results
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save each result as a separate line in JSONL format
    with open(output_file, 'a', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Evaluation results appended to: {output_file}")


def load_evaluation_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation results from JSONL file.
    
    Args:
        file_path: Path to the JSONL file containing evaluation results
        
    Returns:
        List of evaluation results
    """
    results = []
    
    if not os.path.exists(file_path):
        print(f"Warning: Evaluation results file not found at {file_path}")
        return results
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    results.append(data)
        print(f"Loaded {len(results)} evaluation results from {file_path}")
    except Exception as e:
        print(f"Error loading evaluation results: {e}")
    
    return results


def calculate_correlation_analysis(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate correlation between reward scores and number of rubrics.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary containing correlation statistics
    """
    # Extract all score-rubric pairs
    scores = []
    rubrics = []
    
    for entry in results:
        for eval_info in entry['answer_evaluations']:
            if eval_info['extraction_success']:  # Only include successful evaluations
                scores.append(eval_info['hle_score'] if 'hle_score' in eval_info else eval_info['properties_score'])
                rubrics.append(eval_info['num_rubrics_provided'])
    
    if len(scores) < 2:
        return {
            'pearson_correlation': 0.0,
            'pearson_p_value': 1.0,
            'spearman_correlation': 0.0,
            'spearman_p_value': 1.0,
            'sample_size': len(scores)
        }
    
    # Calculate correlations
    pearson_corr, pearson_p = stats.pearsonr(rubrics, scores)
    spearman_corr, spearman_p = stats.spearmanr(rubrics, scores)
    
    return {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'sample_size': len(scores)
    }


def print_paper_reward_summary_statistics(results: List[Dict[str, Any]]):
    """
    Print summary statistics of the paper reward evaluation results.
    
    Args:
        results: List of evaluation results
    """
    print("\n" + "=" * 80)
    print("PAPER REWARD SUMMARY STATISTICS")
    print("=" * 80)
    
    total_questions = len(results)
    total_answers = sum(len(entry['answer_evaluations']) for entry in results)
    
    print(f"Total questions evaluated: {total_questions}")
    print(f"Total answers evaluated: {total_answers}")
    
    # Calculate average scores by number of rubrics
    rubric_paper_scores = {}
    rubric_properties_scores = {}
    successful_evaluations = 0
    
    for entry in results:
        for eval_info in entry['answer_evaluations']:
            num_rubrics = eval_info['num_rubrics_provided']
            paper_score = eval_info['paper_score']
            properties_score = eval_info['properties_score']
            
            if num_rubrics not in rubric_paper_scores:
                rubric_paper_scores[num_rubrics] = []
                rubric_properties_scores[num_rubrics] = []
            
            rubric_paper_scores[num_rubrics].append(paper_score)
            rubric_properties_scores[num_rubrics].append(properties_score)
            
            if eval_info['extraction_success']:
                successful_evaluations += 1
    
    print(f"Successful evaluations: {successful_evaluations}/{total_answers}")
    print(f"Success rate: {successful_evaluations/total_answers*100:.1f}%")
    
    print("\nAverage paper scores by number of rubrics:")
    for num_rubrics in sorted(rubric_paper_scores.keys()):
        scores = rubric_paper_scores[num_rubrics]
        avg_score = sum(scores) / len(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.0
        print(f"  {num_rubrics} rubrics: {avg_score:.4f} ± {std_score:.4f} (n={len(scores)})")
    
    print("\nAverage properties scores by number of rubrics:")
    for num_rubrics in sorted(rubric_properties_scores.keys()):
        scores = rubric_properties_scores[num_rubrics]
        avg_score = sum(scores) / len(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.0
        print(f"  {num_rubrics} rubrics: {avg_score:.4f} ± {std_score:.4f} (n={len(scores)})")
    
    # Overall averages
    all_paper_scores = [score for scores in rubric_paper_scores.values() for score in scores]
    all_properties_scores = [score for scores in rubric_properties_scores.values() for score in scores]
    
    overall_paper_avg = sum(all_paper_scores) / len(all_paper_scores) if all_paper_scores else 0.0
    overall_properties_avg = sum(all_properties_scores) / len(all_properties_scores) if all_properties_scores else 0.0
    
    print(f"\nOverall average paper score: {overall_paper_avg:.4f}")
    print(f"Overall average properties score: {overall_properties_avg:.4f}")
    
    # Correlation analysis for ann scores
    print("\n" + "=" * 80)
    print("ANNOTATION SCORE CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Extract all ann score-rubric pairs
    ann_scores = []
    rubrics = []
    
    for entry in results:
        for eval_info in entry['answer_evaluations']:
            if eval_info['extraction_success']:  # Only include successful evaluations
                # Extract ann_score from scoring_results, fall back to properties_score if not available
                scoring_results = eval_info.get('scoring_results', {})
                ann_score = scoring_results.get('ann_score', eval_info.get('properties_score', 0.0))
                ann_scores.append(ann_score)
                rubrics.append(eval_info['num_rubrics_provided'])
    
    if len(ann_scores) < 2:
        print("Insufficient data for correlation analysis (need at least 2 data points)")
        return
    
    # Calculate correlations
    pearson_corr, pearson_p = stats.pearsonr(rubrics, ann_scores)
    spearman_corr, spearman_p = stats.spearmanr(rubrics, ann_scores)
    
    print(f"Sample size for correlation analysis: {len(ann_scores)}")
    print(f"Pearson correlation (r): {pearson_corr:.4f}")
    print(f"Pearson p-value: {pearson_p:.4f}")
    print(f"Spearman correlation (ρ): {spearman_corr:.4f}")
    print(f"Spearman p-value: {spearman_p:.4f}")
    
    # Interpret correlation strength
    pearson_abs = abs(pearson_corr)
    spearman_abs = abs(spearman_corr)
    
    print("\nCorrelation interpretation:")
    if pearson_abs >= 0.7:
        print("  Pearson: Strong correlation")
    elif pearson_abs >= 0.3:
        print("  Pearson: Moderate correlation")
    else:
        print("  Pearson: Weak correlation")
        
    if spearman_abs >= 0.7:
        print("  Spearman: Strong correlation")
    elif spearman_abs >= 0.3:
        print("  Spearman: Moderate correlation")
    else:
        print("  Spearman: Weak correlation")
    
    # Significance interpretation
    alpha = 0.05
    if pearson_p < alpha:
        print(f"  Pearson correlation is statistically significant (p < {alpha})")
    else:
        print(f"  Pearson correlation is not statistically significant (p >= {alpha})")
        
    if spearman_p < alpha:
        print(f"  Spearman correlation is statistically significant (p < {alpha})")
    else:
        print(f"  Spearman correlation is not statistically significant (p >= {alpha})")


def print_summary_statistics(results: List[Dict[str, Any]], evaluation_type: str = "hle"):
    """
    Print summary statistics of the evaluation results.
    
    Args:
        results: List of evaluation results
        evaluation_type: Type of evaluation ("hle" or "paper")
    """
    if evaluation_type == "paper":
        print_paper_reward_summary_statistics(results)
    else:
        # Original HLE summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        total_questions = len(results)
        total_answers = sum(len(entry['answer_evaluations']) for entry in results)
        
        print(f"Total questions evaluated: {total_questions}")
        print(f"Total answers evaluated: {total_answers}")
        
        # Calculate average scores by number of rubrics
        rubric_scores = {}
        successful_evaluations = 0
        
        for entry in results:
            for eval_info in entry['answer_evaluations']:
                num_rubrics = eval_info['num_rubrics_provided']
                score = eval_info['hle_score'] if 'hle_score' in eval_info else eval_info['properties_score']
                
                if num_rubrics not in rubric_scores:
                    rubric_scores[num_rubrics] = []
                
                rubric_scores[num_rubrics].append(score)
                
                if eval_info['extraction_success']:
                    successful_evaluations += 1
        
        print(f"Successful evaluations: {successful_evaluations}/{total_answers}")
        print(f"Success rate: {successful_evaluations/total_answers*100:.1f}%")
        
        print("\nAverage scores by number of rubrics:")
        for num_rubrics in sorted(rubric_scores.keys()):
            scores = rubric_scores[num_rubrics]
            avg_score = sum(scores) / len(scores)
            std_score = np.std(scores) if len(scores) > 1 else 0.0
            print(f"  {num_rubrics} rubrics: {avg_score:.4f} ± {std_score:.4f} (n={len(scores)})")
        
        # Overall average
        all_scores = [score for scores in rubric_scores.values() for score in scores]
        overall_avg = sum(all_scores) / len(all_scores)
        print(f"\nOverall average score: {overall_avg:.4f}")
        
        # Correlation analysis
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        
        correlation_stats = calculate_correlation_analysis(results)
        
        print(f"Sample size for correlation analysis: {correlation_stats['sample_size']}")
        
        if correlation_stats['sample_size'] < 2:
            print("Insufficient data for correlation analysis (need at least 2 data points)")
            return
        
        print(f"Pearson correlation (r): {correlation_stats['pearson_correlation']:.4f}")
        print(f"Pearson p-value: {correlation_stats['pearson_p_value']:.4f}")
        print(f"Spearman correlation (ρ): {correlation_stats['spearman_correlation']:.4f}")
        print(f"Spearman p-value: {correlation_stats['spearman_p_value']:.4f}")
        
        # Interpret correlation strength
        pearson_abs = abs(correlation_stats['pearson_correlation'])
        spearman_abs = abs(correlation_stats['spearman_correlation'])
        
        print("\nCorrelation interpretation:")
        if pearson_abs >= 0.7:
            print("  Pearson: Strong correlation")
        elif pearson_abs >= 0.3:
            print("  Pearson: Moderate correlation")
        else:
            print("  Pearson: Weak correlation")
            
        if spearman_abs >= 0.7:
            print("  Spearman: Strong correlation")
        elif spearman_abs >= 0.3:
            print("  Spearman: Moderate correlation")
        else:
            print("  Spearman: Weak correlation")
        
        # Significance interpretation
        alpha = 0.05
        if correlation_stats['pearson_p_value'] < alpha:
            print(f"  Pearson correlation is statistically significant (p < {alpha})")
        else:
            print(f"  Pearson correlation is not statistically significant (p >= {alpha})")
            
        if correlation_stats['spearman_p_value'] < alpha:
            print(f"  Spearman correlation is statistically significant (p < {alpha})")
        else:
            print(f"  Spearman correlation is not statistically significant (p >= {alpha})")


def analyze_existing_paper_results(file_path: str):
    """
    Analyze existing paper reward evaluation results from JSONL file.
    
    Args:
        file_path: Path to the JSONL file containing evaluation results
    """
    print(f"Analyzing existing paper reward results from: {file_path}")
    results = load_evaluation_results(file_path)
    
    if results:
        print_paper_reward_summary_statistics(results)
    else:
        print("No results found to analyze.")


def analyze_existing_results(file_path: str, evaluation_type: str = "hle"):
    """
    Analyze existing evaluation results from JSONL file.
    
    Args:
        file_path: Path to the JSONL file containing evaluation results
        evaluation_type: Type of evaluation ("hle" or "paper")
    """
    if evaluation_type == "paper":
        analyze_existing_paper_results(file_path)
    else:
        print(f"Analyzing existing results from: {file_path}")
        results = load_evaluation_results(file_path)
        
        if results:
            print_summary_statistics(results, evaluation_type)
        else:
            print("No results found to analyze.")


def main():
    """Main function to evaluate generated answers with HLE reward."""
    # File paths
    no_reasoning = False
    generated_answers_file = "./generated_answers/generated_answers.jsonl"
    reference_answers_file = "./generated_reference_answers/reference_answers.jsonl"
    hle_output_file = "./evaluation_results/hle_evaluation_results.jsonl" if not no_reasoning else "./evaluation_results/hle_evaluation_results_direct_socre.jsonl"
    rubric_output_file = "./evaluation_results/rubric_evaluation_results.jsonl"
    
    # Check if input file exists
    if not os.path.exists(generated_answers_file):
        print(f"Error: Generated answers file not found at {generated_answers_file}")
        print("Please run the sample_os_responses.py script first to generate answers.")
        return
    
    # Choose evaluation type
    evaluation_type = input("Choose evaluation type (hle/rubric): ").strip().lower()
    if evaluation_type not in ["hle", "rubric"]:
        print("Invalid evaluation type. Using 'hle' as default.")
        evaluation_type = "hle"
    
    output_file = hle_output_file if evaluation_type == "hle" else rubric_output_file
    
    try:
        # Load reference answers if using HLE reward
        reference_answers = {}
        if evaluation_type == "hle":
            print("Loading reference answers...")
            reference_answers = load_reference_answers(reference_answers_file)
            print(f"Loaded {len(reference_answers)} reference answers")
        
        # Load existing evaluations to avoid duplicates
        print("Loading existing evaluation results...")
        evaluated_questions = load_existing_evaluations(output_file)
        print(f"Found {len(evaluated_questions)} already evaluated questions")
        
        # Load generated answers
        print("Loading generated answers...")
        answers_data = load_generated_answers(generated_answers_file)
        print(f"Loaded {len(answers_data)} questions with answers")
        
        # Filter out already evaluated questions
        new_answers_data = [entry for entry in answers_data 
                           if entry.get('question', '') not in evaluated_questions]
        
        if not new_answers_data:
            print("All questions have already been evaluated!")
            # Analyze existing results
            analyze_existing_results(output_file, evaluation_type)
            return
        
        print(f"Evaluating {len(new_answers_data)} new questions...")
        
        if evaluation_type == "rubric":
            # Evaluate answers with paper reward
            print("\nEvaluating answers with paper reward...")
            evaluation_results = evaluate_answers_with_rubric_reward(
                new_answers_data, 
                evaluated_questions=evaluated_questions
            )
        else:
            # Evaluate answers with HLE reward
            print("\nEvaluating answers with HLE reward...")
            evaluation_results = evaluate_answers_with_hle(
                new_answers_data, 
                reference_answers=reference_answers,
                evaluated_questions=evaluated_questions,
                no_reasoning=no_reasoning,
            )
        
        # Save results
        if evaluation_results:
            save_evaluation_results(evaluation_results, output_file)
            
            # Print summary statistics for new evaluations
            print_summary_statistics(evaluation_results, evaluation_type)
        else:
            print("No new evaluations to save.")
        
        # Analyze all results (including existing ones)
        print("\nAnalyzing all results...")
        analyze_existing_results(output_file, evaluation_type)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
