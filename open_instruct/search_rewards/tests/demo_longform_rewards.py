#!/usr/bin/env python3

"""
Demonstration script for longform_averaged_outcome_rewards.py

This script shows how the format reward function works with various input examples.
It demonstrates the scoring of different format elements (answer tags, citation tags, query tags).
"""

import re
import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
search_rewards_dir = os.path.dirname(current_dir)
sys.path.insert(0, search_rewards_dir)

def compute_format_reward_demo(response: str) -> float:
    """
    Demo version of compute_format_reward function.
    
    This function evaluates the format quality of a response by checking for:
    1. Answer tags: <answer>...</answer> (weight: 0.5)
    2. Citation tags: <cite id="...">...</cite> (weight: 0.3)  
    3. Query tags: <query>...</query> (weight: 0.2)
    
    Returns a score between 0.0 and 1.0.
    """
    # Check for answer tags
    answer_pattern = r"<answer>.*?</answer>"
    answer_match = re.search(answer_pattern, response, re.DOTALL)
    answer_format_reward = 1.0 if answer_match else 0.0
    
    # Check for citation tags
    citation_pattern = r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>"
    citation_match = re.search(citation_pattern, response, re.DOTALL)
    citation_format_reward = 1.0 if citation_match else 0.0
    
    # Check for query tags
    query_pattern = r"<query>.*?</query>"
    query_match = re.search(query_pattern, response, re.DOTALL)
    query_format_reward = 1.0 if query_match else 0.0
    
    # Compute weighted average
    format_reward = 0.5 * answer_format_reward + 0.3 * citation_format_reward + 0.2 * query_format_reward
    
    return format_reward


def demo_format_scoring():
    """Demonstrate format scoring with various examples."""
    
    print("=" * 80)
    print("LONGFORM AVERAGED OUTCOME REWARDS - FORMAT SCORING DEMO")
    print("=" * 80)
    print()
    
    # Test cases with different format combinations
    test_cases = [
        {
            "name": "Perfect Format (All Elements)",
            "response": """
            <query>What is machine learning?</query>
            
            <snippets id="ml123">Machine learning is a subset of artificial intelligence that focuses on algorithms.</snippets>
            
            <answer>
            Machine learning is a powerful approach to artificial intelligence. 
            <cite id="ml123">It involves training algorithms on data to make predictions or decisions without being explicitly programmed for each task.</cite>
            This field has revolutionized many industries and continues to grow rapidly.
            </answer>
            """,
            "expected_components": {"answer": True, "citation": True, "query": True},
            "expected_score": 1.0
        },
        
        {
            "name": "Answer and Citation Only",
            "response": """
            <answer>
            Artificial intelligence is the simulation of human intelligence in machines.
            <cite id="ai456">AI systems are designed to perform tasks that typically require human intelligence, such as visual perception, speech recognition, and decision-making.</cite>
            </answer>
            """,
            "expected_components": {"answer": True, "citation": True, "query": False},
            "expected_score": 0.8  # 0.5 + 0.3 + 0.0
        },
        
        {
            "name": "Answer Only",
            "response": """
            <answer>
            Deep learning is a subset of machine learning that uses neural networks with multiple layers.
            It has been particularly successful in image recognition and natural language processing.
            </answer>
            """,
            "expected_components": {"answer": True, "citation": False, "query": False},
            "expected_score": 0.5  # 0.5 + 0.0 + 0.0
        },
        
        {
            "name": "Citation Only",
            "response": """
            Natural language processing enables computers to understand human language.
            <cite id="nlp789">Recent advances in transformer models have significantly improved NLP capabilities.</cite>
            This technology powers many applications we use daily.
            """,
            "expected_components": {"answer": False, "citation": True, "query": False},
            "expected_score": 0.3  # 0.0 + 0.3 + 0.0
        },
        
        {
            "name": "Query Only",
            "response": """
            <query>How does computer vision work in autonomous vehicles?</query>
            
            Computer vision in autonomous vehicles uses cameras and sensors to interpret the environment.
            """,
            "expected_components": {"answer": False, "citation": False, "query": True},
            "expected_score": 0.2  # 0.0 + 0.0 + 0.2
        },
        
        {
            "name": "No Format Elements",
            "response": """
            This is just plain text without any special formatting.
            It discusses robotics and automation but has no structured elements.
            """,
            "expected_components": {"answer": False, "citation": False, "query": False},
            "expected_score": 0.0  # 0.0 + 0.0 + 0.0
        },
        
        {
            "name": "Malformed Citations",
            "response": """
            <answer>
            This response has an answer but malformed citations.
            <cite>Missing id attribute</cite>
            <cite id="">Empty id attribute</cite>
            </answer>
            """,
            "expected_components": {"answer": True, "citation": False, "query": False},
            "expected_score": 0.5  # 0.5 + 0.0 + 0.0
        },
        
        {
            "name": "Multiple Valid Citations",
            "response": """
            <answer>
            This response demonstrates multiple citation formats.
            <cite id="ref1">First citation with double quotes.</cite>
            <cite id='ref2'>Second citation with single quotes.</cite>
            <cite id=ref3>Third citation without quotes.</cite>
            </answer>
            """,
            "expected_components": {"answer": True, "citation": True, "query": False},
            "expected_score": 0.8  # 0.5 + 0.3 + 0.0
        }
    ]
    
    # Run tests and display results
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print("-" * 60)
        
        # Compute the score
        score = compute_format_reward_demo(test_case['response'])
        
        # Display the response (truncated for readability)
        response_preview = test_case['response'].strip()[:200] + "..." if len(test_case['response'].strip()) > 200 else test_case['response'].strip()
        print(f"Response Preview: {response_preview}")
        print()
        
        # Show component detection
        answer_detected = bool(re.search(r"<answer>.*?</answer>", test_case['response'], re.DOTALL))
        citation_detected = bool(re.search(r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>", test_case['response'], re.DOTALL))
        query_detected = bool(re.search(r"<query>.*?</query>", test_case['response'], re.DOTALL))
        
        print("Format Element Detection:")
        print(f"  ✓ Answer tags:    {answer_detected} (weight: 0.5)")
        print(f"  ✓ Citation tags:  {citation_detected} (weight: 0.3)")
        print(f"  ✓ Query tags:     {query_detected} (weight: 0.2)")
        print()
        
        # Show score calculation
        answer_score = 0.5 if answer_detected else 0.0
        citation_score = 0.3 if citation_detected else 0.0
        query_score = 0.2 if query_detected else 0.0
        
        print("Score Calculation:")
        print(f"  Answer component:   {answer_score:.1f}")
        print(f"  Citation component: {citation_score:.1f}")
        print(f"  Query component:    {query_score:.1f}")
        print(f"  Total Score:        {score:.1f}")
        print(f"  Expected Score:     {test_case['expected_score']:.1f}")
        
        # Verify the result
        if abs(score - test_case['expected_score']) < 0.001:
            print("  ✅ PASS - Score matches expected value")
        else:
            print("  ❌ FAIL - Score does not match expected value")
        
        print()
        print("=" * 80)
        print()


def demo_reward_weights():
    """Demonstrate the reward weight configuration."""
    
    print("REWARD WEIGHTS CONFIGURATION")
    print("=" * 40)
    print()
    
    REWARD_WEIGHTS = {
        "rubric": 0.5,      # Rubric-based content quality scoring
        "citation": 0.2,    # Citation accuracy and relevance
        "format": 0.2,      # Format compliance (answer/cite/query tags)
        "num_search_turns": 0.1,  # Number of search operations performed
    }
    
    print("The longform averaged outcome reward combines multiple components:")
    print()
    for component, weight in REWARD_WEIGHTS.items():
        percentage = weight * 100
        print(f"  {component.replace('_', ' ').title():<20} {weight:.1f} ({percentage:4.1f}%)")
    
    total_weight = sum(REWARD_WEIGHTS.values())
    print(f"  {'Total':<20} {total_weight:.1f} ({total_weight*100:4.1f}%)")
    print()
    
    print("Component Descriptions:")
    print("  • Rubric:         Quality of the answer content based on ground truth")
    print("  • Citation:       Accuracy and relevance of citations to claims")
    print("  • Format:         Proper use of <answer>, <cite>, and <query> tags")
    print("  • Search Turns:   Appropriate number of search operations performed")
    print()


if __name__ == "__main__":
    print("Longform Averaged Outcome Rewards - Demonstration")
    print("=" * 60)
    print()
    
    # Run the format scoring demo
    demo_format_scoring()
    
    # Show the reward weights
    demo_reward_weights()
    
    print("Demo completed successfully!")
    print("=" * 60) 