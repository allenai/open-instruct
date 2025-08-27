#!/usr/bin/env python3
"""
Utility functions for visualizing FGRPO advantage computation and token-level application.

This module provides functions to visualize how advantages are computed and applied
to tokens during FGRPO training, helping users understand the reward normalization
and token-level reinforcement process.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizer
import logging

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'      # Positive advantage
    BLUE = '\033[94m'     # Negative advantage  
    GREEN = '\033[92m'    # Info/success
    YELLOW = '\033[93m'   # Warning/header
    PURPLE = '\033[95m'   # Headers
    CYAN = '\033[96m'     # Emphasis
    WHITE = '\033[97m'    # Default
    BOLD = '\033[1m'
    END = '\033[0m'       # Reset

def print_colored(text: str, color: str = Colors.WHITE):
    """Print text with color"""
    print(f"{color}{text}{Colors.END}")

def convert_string_span_to_token_span(effective_spans: List[Tuple[int, int]], 
                                    decoded_resp: str, 
                                    token_resp: List[int], 
                                    tokenizer: PreTrainedTokenizer) -> List[int]:
    """
    Convert character spans to token spans and create a mask for training.
    This is a copy of the function from FGRPO to ensure consistency.
    
    Args:
        effective_spans: List of tuples (start_char, end_char) in decoded response
        decoded_resp: Decoded string response
        token_resp: Tokenized response (list of token IDs)
        tokenizer: Tokenizer used to decode the responses
        
    Returns:
        List of integers where 1 means the token should be trained on, 0 means masked
    """
    if not effective_spans or len(token_resp) == 0:
        # If no effective spans or no tokens, mask everything except last EOS
        mask = [0] * len(token_resp)
        if len(token_resp) > 0 and token_resp[-1] == tokenizer.eos_token_id:
            mask[-1] = 1
        return mask
    
    # Build character-to-token mapping by decoding each token
    char_to_token = {}
    current_pos = 0
    
    for token_idx, token_id in enumerate(token_resp):
        # Decode individual token
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        
        if token_text:
            # Find where this token appears in the decoded response
            token_start = decoded_resp.find(token_text, current_pos)
            if token_start != -1:
                token_end = token_start + len(token_text)
                # Map all characters in this token to the token index
                for char_idx in range(token_start, token_end):
                    if char_idx < len(decoded_resp):
                        char_to_token[char_idx] = token_idx
                current_pos = token_end
            else:
                # Token not found at expected position, map current position
                if current_pos < len(decoded_resp):
                    char_to_token[current_pos] = token_idx
                    current_pos += 1
        else:
            # Empty token text, map current position if valid
            if current_pos < len(decoded_resp):
                char_to_token[current_pos] = token_idx
                current_pos += 1
    
    # Fill any remaining unmapped characters with the last token
    if len(token_resp) > 0:
        for char_idx in range(current_pos, len(decoded_resp)):
            char_to_token[char_idx] = len(token_resp) - 1
    
    # Initialize mask - everything masked by default
    mask = [0] * len(token_resp)
    
    # Unmask tokens that fall within effective spans
    for span_idx, (start_char, end_char) in enumerate(effective_spans):
        # Clamp character indices to valid range
        start_char = max(0, min(start_char, len(decoded_resp)))
        end_char = max(start_char, min(end_char, len(decoded_resp)))
        
        # Find token boundaries for this character span
        if start_char < len(decoded_resp) and start_char in char_to_token:
            token_start = char_to_token[start_char]
        else:
            token_start = 0
        
        if end_char > 0 and (end_char - 1) in char_to_token:
            token_end = char_to_token[end_char - 1] + 1
        else:
            token_end = token_start
        
        # Unmask tokens in this range
        for token_idx in range(token_start, min(token_end, len(token_resp))):
            mask[token_idx] = 1
    
    return mask

def compute_token_advantages(response: str, 
                           finegrained_rewards: Any, 
                           tokenizer: PreTrainedTokenizer) -> Tuple[np.ndarray, List[int], str]:
    """
    Compute token-level advantages exactly like FGRPO does.
    
    Args:
        response: The response string
        finegrained_rewards: FinegrainedRewardOutput object with finegrained_scores
        tokenizer: The tokenizer used
        
    Returns:
        Tuple of (token_advantages, token_ids, decoded_response)
    """
    # Tokenize the response
    token_ids = tokenizer.encode(response, add_special_tokens=False)
    decoded_resp = tokenizer.decode(token_ids, skip_special_tokens=True)
    
    # Initialize per-token advantage array (same as FGRPO line 1292)
    token_advantage = np.zeros(len(token_ids), dtype=np.float32)
    token_advantage_count = np.zeros(len(token_ids), dtype=np.int32)
    
    # Process each finegrained score (same as FGRPO lines 1295-1305)
    for score_obj in finegrained_rewards.finegrained_scores:
        # Get token mask for this finegrained score's spans
        span_mask = convert_string_span_to_token_span(
            score_obj.effective_spans, decoded_resp, token_ids, tokenizer
        )
        
        # Add this advantage to all tokens covered by the spans
        for token_idx, is_covered in enumerate(span_mask):
            if is_covered == 1:
                token_advantage[token_idx] += score_obj.advantage
                token_advantage_count[token_idx] += 1
    
    # Calculate average advantages (same as FGRPO lines 1307-1310)
    for token_idx in range(len(token_ids)):
        if token_advantage_count[token_idx] > 0:
            token_advantage[token_idx] /= token_advantage_count[token_idx]
    
    return token_advantage, token_ids, decoded_resp

def visualize_advantage_application(response: str,
                                  finegrained_rewards: Any,
                                  tokenizer: PreTrainedTokenizer,
                                  response_idx: int = 0,
                                  max_length: int = 200,
                                  show_token_details: bool = False) -> str:
    """
    Visualize how advantages are applied to tokens in a response.
    
    Args:
        response: The response string to visualize
        finegrained_rewards: FinegrainedRewardOutput object with finegrained_scores
        tokenizer: The tokenizer used
        response_idx: Index of this response (for logging)
        max_length: Maximum length to display (truncate if longer)
        show_token_details: Whether to show detailed token breakdown
        
    Returns:
        Colored string representation of the response
    """
    
    # Compute token advantages
    token_advantages, token_ids, decoded_resp = compute_token_advantages(
        response, finegrained_rewards, tokenizer
    )
    
    # Truncate if too long
    display_response = response
    if len(response) > max_length:
        display_response = response[:max_length] + "..."
    
    # Create character-level advantage mapping for visualization
    char_advantages = [0.0] * len(display_response)
    
    # Map token advantages back to characters for coloring
    current_pos = 0
    for token_idx, token_id in enumerate(token_ids):
        if token_idx >= len(token_advantages):
            break
            
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        advantage = token_advantages[token_idx]
        
        if token_text and current_pos < len(display_response):
            # Find token in the display response
            token_start = display_response.find(token_text, current_pos)
            if token_start != -1:
                token_end = min(token_start + len(token_text), len(display_response))
                # Apply advantage to all characters in this token
                for char_idx in range(token_start, token_end):
                    char_advantages[char_idx] = advantage
                current_pos = token_end
            else:
                # Fallback: apply to current position
                if current_pos < len(display_response):
                    char_advantages[current_pos] = advantage
                    current_pos += 1
    
    # Create colored output
    colored_response = ""
    for i, char in enumerate(display_response):
        advantage = char_advantages[i] if i < len(char_advantages) else 0.0
        if advantage > 0.01:  # Positive advantage threshold
            colored_response += f"{Colors.RED}{char}{Colors.END}"
        elif advantage < -0.01:  # Negative advantage threshold
            colored_response += f"{Colors.BLUE}{char}{Colors.END}"
        else:
            colored_response += char
    
    # Statistics
    positive_tokens = sum(1 for adv in token_advantages if adv > 0.01)
    negative_tokens = sum(1 for adv in token_advantages if adv < -0.01)
    zero_tokens = len(token_advantages) - positive_tokens - negative_tokens
    
    # Log the visualization
    print_colored(f"🎨 Response {response_idx} Advantage Visualization:", Colors.PURPLE)
    print(f"   {colored_response}")
    print_colored(f"   📊 Tokens: 🔴{positive_tokens} positive, 🔵{negative_tokens} negative, ⚪{zero_tokens} neutral", Colors.CYAN)
    
    # Show reward group breakdown
    group_info = {}
    for score_obj in finegrained_rewards.finegrained_scores:
        group_id = score_obj.reward_group_id
        advantage = score_obj.advantage
        if group_id not in group_info:
            group_info[group_id] = []
        group_info[group_id].append(advantage)
    
    group_summary = ", ".join([f"Group {gid}: {advs[0]:.2f}" for gid, advs in group_info.items()])
    print_colored(f"   🎯 {group_summary}", Colors.CYAN)
    
    # Show detailed token breakdown if requested
    if show_token_details:
        print_colored(f"   🔤 Token Details:", Colors.YELLOW)
        tokens = [tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids[:10]]  # Show first 10
        for i, (token_text, advantage) in enumerate(zip(tokens, token_advantages[:10])):
            advantage_symbol = "🔴" if advantage > 0.01 else "🔵" if advantage < -0.01 else "⚪"
            print_colored(f"      Token {i:2d}: '{token_text:8s}' → {advantage:6.3f} {advantage_symbol}", Colors.WHITE)
        if len(token_ids) > 10:
            print_colored(f"      ... and {len(token_ids) - 10} more tokens", Colors.WHITE)
    
    return colored_response

def log_advantage_examples(responses: List[str],
                         decoded_responses: List[str], 
                         all_finegrained_rewards: List[Any],
                         tokenizer: PreTrainedTokenizer,
                         step: int,
                         num_examples: int = 5,
                         show_token_details: bool = False):
    """
    Log advantage visualization examples during training.
    
    Args:
        responses: List of response token tensors
        decoded_responses: List of decoded response strings  
        all_finegrained_rewards: List of FinegrainedRewardOutput objects
        tokenizer: The tokenizer used
        step: Current training step
        num_examples: Number of examples to show
        show_token_details: Whether to show detailed token breakdown
    """
    
    logger = logging.getLogger(__name__)
    
    print_colored(f"\n🎨 ADVANTAGE VISUALIZATION - Step {step}", Colors.BOLD + Colors.PURPLE)
    print_colored("=" * 60, Colors.PURPLE)
    print_colored("🔴 Red = Positive Advantage | 🔵 Blue = Negative Advantage", Colors.WHITE)
    
    # Show examples
    num_to_show = min(num_examples, len(decoded_responses))
    
    for i in range(num_to_show):
        try:
            response = decoded_responses[i]
            finegrained_rewards = all_finegrained_rewards[i]
            
            # Skip if no finegrained scores
            if not hasattr(finegrained_rewards, 'finegrained_scores') or not finegrained_rewards.finegrained_scores:
                continue
                
            visualize_advantage_application(
                response=response,
                finegrained_rewards=finegrained_rewards,
                tokenizer=tokenizer,
                response_idx=i,
                max_length=150,  # Shorter for training logs
                show_token_details=show_token_details
            )
            
        except Exception as e:
            logger.warning(f"Failed to visualize response {i}: {e}")
    
    print_colored("=" * 60, Colors.PURPLE)

def log_advantage_statistics(all_finegrained_rewards: List[Any], step: int):
    """
    Log aggregate statistics about advantage computation.
    
    Args:
        all_finegrained_rewards: List of FinegrainedRewardOutput objects
        step: Current training step
    """
    
    logger = logging.getLogger(__name__)
    
    # Collect statistics
    total_scores = 0
    group_advantages = {}
    group_counts = {}
    
    for finegrained_reward in all_finegrained_rewards:
        if not hasattr(finegrained_reward, 'finegrained_scores'):
            continue
            
        for score_obj in finegrained_reward.finegrained_scores:
            total_scores += 1
            group_id = score_obj.reward_group_id
            advantage = score_obj.advantage
            
            if group_id not in group_advantages:
                group_advantages[group_id] = []
                group_counts[group_id] = 0
                
            group_advantages[group_id].append(advantage)
            group_counts[group_id] += 1
    
    # Log statistics
    print_colored(f"\n📊 ADVANTAGE STATISTICS - Step {step}", Colors.BOLD + Colors.GREEN)
    print_colored(f"   Total finegrained scores: {total_scores}", Colors.WHITE)
    
    for group_id in sorted(group_advantages.keys()):
        advantages = group_advantages[group_id]
        mean_adv = np.mean(advantages)
        std_adv = np.std(advantages)
        count = group_counts[group_id]
        
        print_colored(f"   Group {group_id}: {count} scores, mean={mean_adv:.4f}, std={std_adv:.4f}", Colors.CYAN)
    
    print()
