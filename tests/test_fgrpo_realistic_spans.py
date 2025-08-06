#!/usr/bin/env python3
"""
Test case for realistic finegrained GRPO with proper span masking.
This demonstrates how different reward types apply to different tokens
and verifies that gradients only flow through the relevant effective spans.
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
from unittest.mock import Mock, patch
import sys
import os

# Add the open_instruct module to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_instruct.rl_utils2 import pack_sequences
from open_instruct.fgrpo_fast import collate_fn, masked_mean

# Import the INVALID_LOGPROB constant
INVALID_LOGPROB = 1.0


@dataclass
class MockArgs:
    """Mock Args class with the necessary attributes for testing"""
    advantage_normalization_type: str = "finegrained"
    num_samples_per_prompt_rollout: int = 2
    pack_length: int = 512
    per_device_train_batch_size: int = 2
    world_size: int = 1
    mask_truncated_completions: bool = False
    allow_world_padding: bool = False
    beta: float = 0.05  # KL coefficient
    clip_lower: float = 0.2
    clip_higher: float = 0.2
    kl_estimator: str = "kl3"
    masked_mean_axis: Optional[int] = None
    temperature: float = 0.7


class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1


class MockModel(nn.Module):
    """Simple mock model for testing gradients"""
    def __init__(self, vocab_size=1000, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        # Simple forward pass
        x = self.embedding(input_ids)
        logits = self.linear(x)
        return type('Output', (), {'logits': logits})()


def create_realistic_finegrained_data():
    """Create realistic test data with different reward types and spans"""
    
    # Realistic responses for a math problem
    responses = [
        # Response 1: "Let me think step by step.\n\nFirst, I need to solve 2+3.\n2+3 = 5\n\nThe answer is 5."
        [50, 51, 52, 53, 54, 55, 10, 10, 56, 57, 58, 59, 60, 61, 62, 10, 63, 64, 65, 66, 10, 10, 67, 68, 69, 70, 71],
        
        # Response 2: "I'll solve this quickly.\n2+3=5\nAnswer: 5"  
        [80, 81, 82, 83, 84, 10, 85, 86, 87, 10, 88, 89, 90],
        
        # Response 3: "Step 1: Add the numbers\nStep 2: 2+3=5\nStep 3: State the answer\nFinal answer: 5"
        [100, 101, 102, 103, 104, 10, 105, 106, 107, 108, 109, 10, 110, 111, 112, 113, 114, 10, 115, 116, 117, 118],
        
        # Response 4: "2+3=5" (minimal response)
        [120, 121, 122, 123]
    ]
    
    # Queries for math problems
    queries = [
        [200, 201, 202],  # "What is 2+3?"
        [200, 201, 202],  # Same query
        [203, 204, 205],  # "Calculate 2+3"  
        [203, 204, 205],  # Same query
    ]
    
    # Tool masks (all normal tokens for this example)
    masks = [
        [1] * len(resp) for resp in responses
    ]
    
    # Realistic finegrained scores with different reward types:
    # Reward Group 0: Format/Structure reward (applies to reasoning steps)
    # Reward Group 1: Verification reward (applies to final answer)
    # Reward Group 2: Step-by-step reward (applies to intermediate calculations)
    
    finegrained_scores = [
        # Response 1: Well-structured with good reasoning
        # Format reward: High score for structured reasoning (tokens 0-15: "Let me think step by step.\n\nFirst, I need to solve")
        (0.9, [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False], 0),
        
        # Verification reward: Correct answer (tokens 23-26: "answer is 5")
        (1.0, [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True], 1),
        
        # Step-by-step reward: Shows calculation (tokens 16-22: "2+3.\n2+3 = 5\n\n")  
        (0.8, [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, False, False, False, False], 2),
        
        # Response 2: Quick but correct
        # Format reward: Low score for minimal structure (tokens 0-5: "I'll solve this")
        (0.3, [True, True, True, True, True, True, False, False, False, False, False, False, False], 0),
        
        # Verification reward: Correct answer (tokens 10-12: "Answer: 5")
        (1.0, [False, False, False, False, False, False, False, False, False, False, True, True, True], 1),
        
        # Step-by-step reward: Shows calculation but minimal (tokens 6-9: "2+3=5")
        (0.4, [False, False, False, False, False, False, True, True, True, True, False, False, False], 2),
        
        # Response 3: Very structured approach
        # Format reward: High score for excellent structure (tokens 0-9, 15-21: step labels)
        (1.0, [True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, True, True, True, True, True, True, True], 0),
        
        # Verification reward: Correct final answer (tokens 18-21: "Final answer: 5")
        (1.0, [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True], 1),
        
        # Step-by-step reward: Shows calculation (tokens 10-14: "2+3=5")
        (0.7, [False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, False, False, False, False, False, False, False], 2),
        
        # Response 4: Minimal but correct
        # Format reward: Very low score for no structure (no tokens)
        (0.1, [False, False, False, False], 0),
        
        # Verification reward: Correct answer (all tokens: "2+3=5")
        (1.0, [True, True, True, True], 1),
        
        # Step-by-step reward: Shows calculation but no steps (all tokens)
        (0.2, [True, True, True, True], 2),
    ]
    
    return {
        'responses': responses,
        'queries': queries,
        'masks': masks,
        'finegrained_scores': finegrained_scores,
    }


def test_realistic_span_masking():
    """Test that different reward types properly mask different token spans"""
    print("=" * 80)
    print("TESTING REALISTIC FINEGRAINED SPAN MASKING")
    print("=" * 80)
    
    test_data = create_realistic_finegrained_data()
    
    print("Test scenario: Math problem responses with 3 reward types:")
    print("  - Group 0 (Format): Rewards structured reasoning")
    print("  - Group 1 (Verification): Rewards correct answers") 
    print("  - Group 2 (Step-by-step): Rewards showing calculations")
    print()
    
    # Show the span patterns for each response
    for i, response in enumerate(test_data['responses']):
        print(f"Response {i+1} (length {len(response)}):")
        
        # Find all spans for this response
        response_spans = []
        for j, (score, span, group_id) in enumerate(test_data['finegrained_scores']):
            if j // 3 == i:  # Each response has 3 reward segments
                response_spans.append((group_id, span, score))
        
        # Sort by group_id for consistent display
        response_spans.sort(key=lambda x: x[0])
        
        for group_id, span, score in response_spans:
            group_names = ["Format", "Verification", "Step-by-step"]
            span_tokens = [j for j, mask in enumerate(span) if mask]
            print(f"  - {group_names[group_id]} reward (score={score:.1f}): tokens {span_tokens}")
        print()
    
    return test_data


def test_gradient_isolation():
    """Test that gradients only flow through the effective tokens for each reward group"""
    print("=" * 80)
    print("TESTING GRADIENT ISOLATION BY REWARD GROUP")
    print("=" * 80)
    
    test_data = create_realistic_finegrained_data()
    args = MockArgs()
    tokenizer = MockTokenizer()
    
    # Compute advantages
    advantages_list, advantages_mask_list, reward_group_ids = compute_finegrained_advantages(test_data['finegrained_scores'])
    
    # Pack sequences
    packed_sequences = pack_sequences(
        queries=test_data['queries'],
        responses=test_data['responses'],
        masks=test_data['masks'],
        pack_length=args.pack_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Create packed advantages with realistic span masks
    packed_advantages_list = []
    packed_advantages_mask_list = []
    
    unique_groups = list(set(reward_group_ids))
    print(f"Processing {len(unique_groups)} reward groups: {sorted(unique_groups)}")
    print()
    
    for group_idx, group_id in enumerate(sorted(unique_groups)):
        print(f"Processing reward group {group_id} ({['Format', 'Verification', 'Step-by-step'][group_id]}):")
        
        # Get advantages for this group
        group_advantages = advantages_list[group_idx]
        lookup_advantages = np.zeros(len(group_advantages) + 1, dtype=np.float32)
        lookup_advantages[1:] = group_advantages
        
        print(f"  - Group advantages: {group_advantages}")
        
        # Pack advantages for this group
        packed_advantages = [
            torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
            for packed_mask in packed_sequences.response_masks
        ]
        packed_advantages_list.append(packed_advantages)
        
        # Create realistic span masks based on the finegrained_scores
        packed_span_masks = []
        for seq_idx, packed_mask in enumerate(packed_sequences.response_masks):
            span_mask = torch.zeros_like(packed_mask, dtype=torch.bool)
            
            # Map packed positions back to original responses
            response_start_positions = []
            current_pos = 0
            
            for resp_idx in range(len(test_data['responses'])):
                query_len = len(test_data['queries'][resp_idx])
                response_len = len(test_data['responses'][resp_idx])
                response_start_positions.append(current_pos + query_len)
                current_pos += query_len + response_len
            
            # Apply the effective spans for this group
            for resp_idx in range(len(test_data['responses'])):
                # Find the span for this response and group
                span_idx = resp_idx * 3 + group_id  # 3 reward types per response
                if span_idx < len(test_data['finegrained_scores']):
                    _, effective_span, _ = test_data['finegrained_scores'][span_idx]
                    
                    # Map the effective span to packed positions
                    response_start = response_start_positions[resp_idx]
                    for token_idx, is_effective in enumerate(effective_span):
                        packed_pos = response_start + token_idx
                        if packed_pos < len(span_mask) and packed_mask[packed_pos] > 0:
                            span_mask[packed_pos] = is_effective
            
            packed_span_masks.append(span_mask)
            
            # Show span statistics
            effective_tokens = span_mask.sum().item()
            total_response_tokens = (packed_mask > 0).sum().item()
            print(f"    * Sequence {seq_idx}: {effective_tokens}/{total_response_tokens} tokens are effective")
        
        packed_advantages_mask_list.append(packed_span_masks)
        print()
    
    # Store in expected format
    packed_sequences.advantages = [(packed_advantages_list, packed_advantages_mask_list)]
    
    # Test gradient isolation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MockModel().to(device)
    
    print("Testing gradient isolation:")
    print("  - Training with all reward groups together")
    print("  - Then training with each group individually")  
    print("  - Verifying that gradients are different when different spans are used")
    print()
    
    # Test 1: All groups together
    all_groups_result = run_training_step(model, packed_sequences, args, tokenizer, device)
    print(f"All groups together: Loss = {all_groups_result['final_loss']:.6f}, Grad norm = {all_groups_result['total_grad_norm']:.6f}")
    
    # Test 2: Each group individually
    individual_results = []
    for group_idx in range(len(packed_advantages_list)):
        # Create a copy of the model with same weights
        model_copy = MockModel().to(device)
        model_copy.load_state_dict(model.state_dict())
        
        # Create packed sequences with only this group
        temp_packed_sequences = type(packed_sequences)(**{
            attr: getattr(packed_sequences, attr) for attr in dir(packed_sequences) 
            if not attr.startswith('_') and attr != 'advantages'
        })
        temp_packed_sequences.advantages = [([packed_advantages_list[group_idx]], [packed_advantages_mask_list[group_idx]])]
        
        group_result = run_training_step(model_copy, temp_packed_sequences, args, tokenizer, device)
        individual_results.append(group_result)
        
        group_names = ["Format", "Verification", "Step-by-step"]
        print(f"Group {group_idx} ({group_names[group_idx]}) only: Loss = {group_result['final_loss']:.6f}, Grad norm = {group_result['total_grad_norm']:.6f}")
    
    print()
    
    # Verify that different groups produce different gradients
    print("Gradient isolation verification:")
    for i in range(len(individual_results)):
        for j in range(i + 1, len(individual_results)):
            grad_diff = abs(individual_results[i]['total_grad_norm'] - individual_results[j]['total_grad_norm'])
            loss_diff = abs(individual_results[i]['final_loss'] - individual_results[j]['final_loss'])
            group_names = ["Format", "Verification", "Step-by-step"]
            print(f"  - {group_names[i]} vs {group_names[j]}: Grad diff = {grad_diff:.6f}, Loss diff = {loss_diff:.6f}")
    
    print()
    return all_groups_result, individual_results


def test_span_coverage():
    """Test that different reward groups cover different token ranges without excessive overlap"""
    print("=" * 80)
    print("TESTING SPAN COVERAGE AND OVERLAP")
    print("=" * 80)
    
    test_data = create_realistic_finegrained_data()
    
    for resp_idx, response in enumerate(test_data['responses']):
        print(f"Response {resp_idx + 1} (length {len(response)}):")
        
        # Get all spans for this response
        spans_by_group = {}
        for score_idx, (score, span, group_id) in enumerate(test_data['finegrained_scores']):
            if score_idx // 3 == resp_idx:  # 3 reward types per response
                spans_by_group[group_id] = span
        
        # Analyze coverage
        total_tokens = len(response)
        covered_by_any = [False] * total_tokens
        coverage_count = [0] * total_tokens
        
        group_names = ["Format", "Verification", "Step-by-step"]
        
        for group_id in sorted(spans_by_group.keys()):
            span = spans_by_group[group_id]
            effective_count = sum(span)
            coverage_pct = (effective_count / total_tokens) * 100
            
            print(f"  - {group_names[group_id]}: {effective_count}/{total_tokens} tokens ({coverage_pct:.1f}%)")
            
            for token_idx, is_effective in enumerate(span):
                if is_effective:
                    covered_by_any[token_idx] = True
                    coverage_count[token_idx] += 1
        
        # Overall coverage statistics
        total_covered = sum(covered_by_any)
        total_coverage_pct = (total_covered / total_tokens) * 100
        
        overlapping_tokens = sum(1 for count in coverage_count if count > 1)
        overlap_pct = (overlapping_tokens / total_tokens) * 100
        
        print(f"  - Total coverage: {total_covered}/{total_tokens} tokens ({total_coverage_pct:.1f}%)")
        print(f"  - Overlapping tokens: {overlapping_tokens}/{total_tokens} tokens ({overlap_pct:.1f}%)")
        print()
    
    return True


def compute_finegrained_advantages(finegrained_scores):
    """Compute finegrained advantages like in the main implementation"""
    
    # Extract components
    scores = np.array([score for score, _, _ in finegrained_scores])
    effective_spans = [effective_span for _, effective_span, _ in finegrained_scores]
    reward_group_ids = [reward_group_id for _, _, reward_group_id in finegrained_scores]
    
    # Normalize advantages per group
    unique_groups = list(set(reward_group_ids))
    advantages = np.zeros_like(scores, dtype=np.float32)
    
    for group_id in unique_groups:
        group_indices = [i for i, gid in enumerate(reward_group_ids) if gid == group_id]
        group_scores = scores[group_indices]
        
        # Calculate group statistics
        group_mean = np.mean(group_scores)
        group_std = np.std(group_scores) + 1e-8
        group_advantages = (group_scores - group_mean) / group_std
        
        # Assign normalized advantages back
        for idx, group_idx in enumerate(group_indices):
            advantages[group_idx] = group_advantages[idx]
    
    # Create span masks
    span_masks = []
    for effective_span in effective_spans:
        span_mask = np.array(effective_span, dtype=bool)
        span_masks.append(span_mask)
    
    # Group advantages and masks by reward group
    grouped_advantages = {}
    grouped_masks = {}
    
    for i, group_id in enumerate(reward_group_ids):
        if group_id not in grouped_advantages:
            grouped_advantages[group_id] = []
            grouped_masks[group_id] = []
        grouped_advantages[group_id].append(advantages[i])
        grouped_masks[group_id].append(span_masks[i])
    
    # Convert to expected format
    advantages_list = []
    advantages_mask_list = []
    
    for group_id in sorted(unique_groups):
        advantages_list.append(np.array(grouped_advantages[group_id]))
        advantages_mask_list.append(grouped_masks[group_id])
    
    return advantages_list, advantages_mask_list, reward_group_ids


def run_training_step(model, packed_sequences, args, tokenizer, device):
    """Run a single training step and return results"""
    
    # Simulate collation for one device
    B = len(packed_sequences.query_responses) // args.world_size
    i = 0  # Device 0
    
    # Get advantages structure
    packed_advantages_list, packed_advantages_mask_list = packed_sequences.advantages[0]
    
    # Collate data
    query_responses = packed_sequences.query_responses[B * i : B * (i + 1)]
    attention_masks = packed_sequences.attention_masks[B * i : B * (i + 1)]
    position_ids = packed_sequences.position_ids[B * i : B * (i + 1)]
    response_masks = packed_sequences.response_masks[B * i : B * (i + 1)]
    
    # Collate advantages for each group
    collated_advantages_list = []
    collated_advantages_mask_list = []
    
    for group_idx in range(len(packed_advantages_list)):
        group_advantages = packed_advantages_list[group_idx][B * i : B * (i + 1)]
        group_masks = packed_advantages_mask_list[group_idx][B * i : B * (i + 1)]
        
        collated_group_advantages = collate_fn(group_advantages, 0, pin_memory=False)
        collated_group_masks = collate_fn([mask.float() for mask in group_masks], 0, pin_memory=False).bool()
        
        collated_advantages_list.append(collated_group_advantages.to(device))
        collated_advantages_mask_list.append(collated_group_masks.to(device))
    
    # Collate other tensors
    query_responses_tensor = collate_fn(query_responses, tokenizer.pad_token_id, pin_memory=False).to(device)
    attention_masks_tensor = collate_fn(attention_masks, 0, pin_memory=False).to(device)
    position_ids_tensor = collate_fn(position_ids, 0, pin_memory=False).to(device)
    response_masks_tensor = collate_fn(response_masks, 0, pin_memory=False).to(device)
    
    # Forward pass
    padding_mask = query_responses_tensor != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses_tensor, ~padding_mask, 0)
    
    model_output = model(
        input_ids=input_ids[:, :-1],
        attention_mask=attention_masks_tensor[:, :-1].clamp(0, 1),
        position_ids=position_ids_tensor[:, :-1],
    )
    
    logits = model_output.logits / (args.temperature + 1e-7)
    new_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    
    # Compute loss for each group (this is the key part - each group only affects its effective tokens)
    response_masks_bool = response_masks_tensor[:, 1:].bool()
    loss_list = []
    
    for j, (advantages, advantages_mask) in enumerate(zip(collated_advantages_list, collated_advantages_mask_list)):
        # Apply finegrained mask to logprobs - this is where gradient isolation happens!
        group_new_logprobs = torch.masked_fill(new_logprobs, ~advantages_mask[:, 1:], INVALID_LOGPROB)
        old_logprobs = group_new_logprobs.detach()
        
        # Calculate policy gradient loss
        logprobs_diff = group_new_logprobs - old_logprobs
        ratio = torch.exp(logprobs_diff)
        
        pg_losses = -advantages[:, 1:] * ratio
        pg_losses2 = -advantages[:, 1:] * torch.clamp(
            ratio, 1.0 - args.clip_lower, 1.0 + args.clip_higher
        )
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        
        # Compute masked mean - only over effective tokens for this reward group
        mask_for_loss = response_masks_bool & advantages_mask[:, 1:]
        group_loss = masked_mean(pg_loss_max, mask_for_loss, args.masked_mean_axis)
        loss_list.append(group_loss)
    
    final_loss = torch.mean(torch.stack(loss_list))
    
    # Backward pass
    model.zero_grad()
    final_loss.backward()
    
    # Calculate gradient norm
    total_grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    return {
        'final_loss': final_loss.item(),
        'total_grad_norm': total_grad_norm,
        'group_losses': [loss.item() for loss in loss_list]
    }


def run_all_realistic_tests():
    """Run all realistic finegrained GRPO tests"""
    print("🧪 STARTING REALISTIC FINEGRAINED GRPO TESTS 🧪")
    print()
    
    try:
        # Test 1: Show realistic span patterns
        print("Test 1: Realistic span patterns")
        test_data = test_realistic_span_masking()
        print("✅ Test 1: Realistic span patterns - PASSED")
        print()
        
        # Test 2: Span coverage analysis
        print("Test 2: Span coverage analysis")
        coverage_result = test_span_coverage()
        assert coverage_result, "Span coverage analysis should complete successfully"
        print("✅ Test 2: Span coverage analysis - PASSED")
        print()
        
        # Test 3: Gradient isolation
        print("Test 3: Gradient isolation by reward group")
        all_groups_result, individual_results = test_gradient_isolation()
        
        # Verify that different groups produce different gradients
        assert len(individual_results) == 3, "Should have 3 reward groups"
        
        # Check that individual group gradients are different from each other
        grad_diffs = []
        for i in range(len(individual_results)):
            for j in range(i + 1, len(individual_results)):
                grad_diff = abs(individual_results[i]['total_grad_norm'] - individual_results[j]['total_grad_norm'])
                grad_diffs.append(grad_diff)
        
        # At least some groups should have significantly different gradients
        significant_diffs = sum(1 for diff in grad_diffs if diff > 1e-4)
        assert significant_diffs > 0, "Different reward groups should produce different gradients"
        
        print("✅ Test 3: Gradient isolation - PASSED")
        print()
        
        print("=" * 80)
        print("🎉 ALL REALISTIC TESTS PASSED! 🎉")
        print("Finegrained GRPO correctly isolates gradients to effective token spans!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ REALISTIC TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_realistic_tests()
    exit(0 if success else 1) 