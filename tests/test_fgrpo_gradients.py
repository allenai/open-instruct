#!/usr/bin/env python3
"""
Test case for finegrained GRPO gradient computation.
This tests that gradients flow correctly through the finegrained advantage computation
without running actual model training.
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


def log_softmax_and_gather(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Simple implementation of log_softmax and gather for testing"""
    log_probs = torch.log_softmax(logits, dim=-1)
    # Gather the log probabilities for the labels
    gathered = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return gathered


def create_gradient_test_data():
    """Create test data for gradient computation"""
    
    # Mock responses (token IDs) - smaller for easier testing
    responses = [
        [10, 11, 12],  # Response 1 for prompt 1, sample 1
        [15, 16, 17],  # Response 2 for prompt 1, sample 2
        [20, 21, 22],  # Response 1 for prompt 2, sample 1
        [25, 26, 27],  # Response 2 for prompt 2, sample 2
    ]
    
    # Mock queries (token IDs)
    queries = [
        [5, 6],   # Query for prompt 1, sample 1
        [5, 6],   # Query for prompt 1, sample 2 (same query)
        [8, 9],   # Query for prompt 2, sample 1
        [8, 9],   # Query for prompt 2, sample 2 (same query)
    ]
    
    # Mock tool masks (1 means normal token, 0 means tool token to mask)
    masks = [
        [1, 1, 1],  # All tokens are normal
        [1, 1, 1],  # All tokens are normal
        [1, 1, 1],  # All tokens are normal
        [1, 1, 1],  # All tokens are normal
    ]
    
    # Mock finegrained scores: (score, effective_span, reward_group_id)
    # Each response has multiple reward segments
    finegrained_scores = [
        # Response 1: Two segments with different rewards
        (1.0, [True, True, False], 0),   # First segment, group 0
        (0.5, [False, False, True], 1),  # Second segment, group 1
        
        # Response 2: Two segments
        (0.8, [True, True, False], 0),   # First segment, group 0
        (0.2, [False, False, True], 1),  # Second segment, group 1
        
        # Response 3: Two segments
        (0.9, [True, True, False], 0),   # First segment, group 0
        (0.7, [False, False, True], 1),  # Second segment, group 1
        
        # Response 4: Two segments
        (0.3, [True, True, False], 0),   # First segment, group 0
        (0.6, [False, False, True], 1),  # Second segment, group 1
    ]
    
    return {
        'responses': responses,
        'queries': queries,
        'masks': masks,
        'finegrained_scores': finegrained_scores,
    }


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


def prepare_mock_training_data():
    """Prepare mock training data similar to the actual training pipeline"""
    test_data = create_gradient_test_data()
    args = MockArgs()
    tokenizer = MockTokenizer()
    
    responses = test_data['responses']
    queries = test_data['queries']
    masks = test_data['masks']
    finegrained_scores = test_data['finegrained_scores']
    
    # Compute advantages
    advantages_list, advantages_mask_list, reward_group_ids = compute_finegrained_advantages(finegrained_scores)
    
    # Pack sequences
    packed_sequences = pack_sequences(
        queries=queries,
        responses=responses,
        masks=masks,
        pack_length=args.pack_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Create packed advantages for each group
    packed_advantages_list = []
    packed_advantages_mask_list = []
    
    unique_groups = list(set(reward_group_ids))
    for group_idx, group_id in enumerate(sorted(unique_groups)):
        # Create lookup for this group's advantages
        group_advantages = advantages_list[group_idx]
        lookup_advantages = np.zeros(len(group_advantages) + 1, dtype=np.float32)
        lookup_advantages[1:] = group_advantages
        
        # Pack advantages for this group
        packed_advantages = [
            torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
            for packed_mask in packed_sequences.response_masks
        ]
        packed_advantages_list.append(packed_advantages)
        
        # Create span masks based on effective spans
        packed_span_masks = []
        for i, packed_mask in enumerate(packed_sequences.response_masks):
            # Simplified span mask creation for testing
            span_mask = torch.zeros_like(packed_mask, dtype=torch.bool)
            
            # For testing, create a simple pattern based on the group
            response_token_count = 0
            for j, mask_val in enumerate(packed_mask):
                if mask_val > 0:  # This is a response token
                    # Alternate between groups for different token positions
                    if group_id == 0:
                        # Group 0: first half of response tokens
                        span_mask[j] = response_token_count % 2 == 0
                    else:
                        # Group 1: second half of response tokens  
                        span_mask[j] = response_token_count % 2 == 1
                    response_token_count += 1
            
            # Ensure we have at least some True values
            if not span_mask.any():
                # Fallback: set every other response token to True
                for j, mask_val in enumerate(packed_mask):
                    if mask_val > 0 and j % 2 == group_id:
                        span_mask[j] = True
            
            packed_span_masks.append(span_mask)
        packed_advantages_mask_list.append(packed_span_masks)
    
    # Store in expected format
    packed_sequences.advantages = [(packed_advantages_list, packed_advantages_mask_list)]
    
    return packed_sequences, args, tokenizer


def test_gradient_flow():
    """Test that gradients flow correctly through the finegrained loss computation"""
    print("=" * 60)
    print("TESTING GRADIENT FLOW WITH FINEGRAINED ADVANTAGES")
    print("=" * 60)
    
    # Prepare data
    packed_sequences, args, tokenizer = prepare_mock_training_data()
    
    # Create mock models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MockModel().to(device)
    ref_model = MockModel().to(device)
    
    # Copy weights so ref model is identical initially
    ref_model.load_state_dict(model.state_dict())
    
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print()
    
    # Simulate collation for one device
    B = len(packed_sequences.query_responses) // args.world_size
    i = 0  # Device 0
    
    # Get advantages structure
    packed_advantages_list, packed_advantages_mask_list = packed_sequences.advantages[0]
    
    # Collate data (simplified for testing)
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
    
    print("Step 1: Forward pass through models")
    print(f"  - Input shape: {query_responses_tensor.shape}")
    print(f"  - Number of advantage groups: {len(collated_advantages_list)}")
    
    # Forward pass through current model
    padding_mask = query_responses_tensor != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses_tensor, ~padding_mask, 0)
    
    model_output = model(
        input_ids=input_ids[:, :-1],
        attention_mask=attention_masks_tensor[:, :-1].clamp(0, 1),
        position_ids=position_ids_tensor[:, :-1],
    )
    
    logits = model_output.logits / (args.temperature + 1e-7)
    new_logprobs = log_softmax_and_gather(logits, input_ids[:, 1:])
    
    print(f"  - Logits shape: {logits.shape}")
    print(f"  - New logprobs shape: {new_logprobs.shape}")
    
    # Forward pass through reference model
    with torch.no_grad():
        ref_output = ref_model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_masks_tensor[:, :-1].clamp(0, 1),
            position_ids=position_ids_tensor[:, :-1],
        )
        ref_logits = ref_output.logits / (args.temperature + 1e-7)
        ref_logprobs = log_softmax_and_gather(ref_logits, input_ids[:, 1:])
    
    print(f"  - Ref logprobs shape: {ref_logprobs.shape}")
    print()
    
    print("Step 2: Compute loss for each advantage group")
    response_masks_bool = response_masks_tensor[:, 1:].bool()
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    loss_list = []
    
    for j, (advantages, advantages_mask) in enumerate(zip(collated_advantages_list, collated_advantages_mask_list)):
        print(f"  - Processing group {j}")
        print(f"    * Advantages shape: {advantages.shape}")
        print(f"    * Advantages mask shape: {advantages_mask.shape}")
        
        # Apply finegrained mask to logprobs
        group_new_logprobs = torch.masked_fill(new_logprobs, ~advantages_mask[:, 1:], INVALID_LOGPROB)
        group_ref_logprobs = torch.masked_fill(ref_logprobs, ~advantages_mask[:, 1:], INVALID_LOGPROB)
        
        # For testing, use simple old_logprobs (in practice these would be cached)
        old_logprobs = group_new_logprobs.detach()
        
        # Calculate policy gradient loss
        logprobs_diff = group_new_logprobs - old_logprobs
        ratio = torch.exp(logprobs_diff)
        
        pg_losses = -advantages[:, 1:] * ratio
        pg_losses2 = -advantages[:, 1:] * torch.clamp(
            ratio, 1.0 - args.clip_lower, 1.0 + args.clip_higher
        )
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        
        # Calculate KL divergence
        ref_logprobs_diff = (group_new_logprobs - group_ref_logprobs).clamp(-40.0, 40.0)
        
        if args.kl_estimator == "kl1":
            kl = ref_logprobs_diff
        elif args.kl_estimator == "kl2":
            kl = ref_logprobs_diff ** 2 / 2
        elif args.kl_estimator == "kl3":
            kl = torch.expm1(-ref_logprobs_diff) + ref_logprobs_diff
        elif args.kl_estimator == "kl4":
            kl = ratio * ref_logprobs_diff
        else:
            kl = ref_logprobs_diff
        
        # Combine policy loss and KL
        combined_loss = pg_loss_max + (args.beta * kl)
        
        # Compute masked mean
        mask_for_loss = response_masks_bool & advantages_mask[:, 1:]
        group_loss = masked_mean(combined_loss, mask_for_loss, args.masked_mean_axis)
        
        print(f"    * Group loss: {group_loss.item():.6f}")
        print(f"    * PG loss mean: {masked_mean(pg_loss_max, mask_for_loss, args.masked_mean_axis).item():.6f}")
        print(f"    * KL loss mean: {masked_mean(args.beta * kl, mask_for_loss, args.masked_mean_axis).item():.6f}")
        print(f"    * Ratio mean: {masked_mean(ratio, mask_for_loss, args.masked_mean_axis).item():.6f}")
        print(f"    * Advantages mean: {masked_mean(advantages[:, 1:], mask_for_loss, args.masked_mean_axis).item():.6f}")
        
        loss_list.append(group_loss)
    
    # Final loss is mean across groups
    final_loss = torch.mean(torch.stack(loss_list))
    print(f"\nFinal loss: {final_loss.item():.6f}")
    print()
    
    print("Step 3: Backward pass and gradient check")
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass
    final_loss.backward()
    
    # Check gradients
    total_grad_norm = 0.0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            param_count += 1
            print(f"  - {name}: grad_norm = {grad_norm:.6f}, shape = {param.shape}")
        else:
            print(f"  - {name}: NO GRADIENT")
    
    total_grad_norm = total_grad_norm ** 0.5
    print(f"\nTotal gradient norm: {total_grad_norm:.6f}")
    print(f"Parameters with gradients: {param_count}")
    print()
    
    # Verify gradients are non-zero and finite
    has_gradients = total_grad_norm > 0
    gradients_finite = torch.isfinite(torch.tensor(total_grad_norm))
    
    print("Step 4: Gradient verification")
    print(f"  - Has non-zero gradients: {has_gradients}")
    print(f"  - Gradients are finite: {gradients_finite}")
    print(f"  - Loss is finite: {torch.isfinite(final_loss)}")
    print()
    
    return {
        'final_loss': final_loss.item(),
        'total_grad_norm': total_grad_norm,
        'has_gradients': has_gradients,
        'gradients_finite': gradients_finite,
        'loss_finite': torch.isfinite(final_loss).item(),
        'group_losses': [loss.item() for loss in loss_list]
    }


def test_gradient_differences():
    """Test that different advantage groups produce different gradients"""
    print("=" * 60)
    print("TESTING GRADIENT DIFFERENCES BETWEEN GROUPS")
    print("=" * 60)
    
    # Create both datasets upfront
    original_data = create_gradient_test_data()
    
    # Create modified data with different scores
    modified_data = create_gradient_test_data()
    modified_scores = []
    for i, (score, effective_span, reward_group_id) in enumerate(modified_data['finegrained_scores']):
        # For group 0, swap the first two scores to change relative ordering
        if reward_group_id == 0 and i == 0:  # First group 0 score
            # Find the second group 0 score and use its value
            second_group_0_score = None
            for j, (s2, _, gid2) in enumerate(modified_data['finegrained_scores']):
                if gid2 == 0 and j > i:
                    second_group_0_score = s2
                    break
            new_score = second_group_0_score if second_group_0_score is not None else score * 0.5
        elif reward_group_id == 0 and i == 2:  # Second group 0 score  
            # Use the first group 0 score
            new_score = modified_data['finegrained_scores'][0][0]
        else:
            new_score = score
        modified_scores.append((new_score, effective_span, reward_group_id))
    modified_data['finegrained_scores'] = modified_scores
    
    # Create models with same initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)  # Fix seed for reproducible results
    
    print("Run 1: Original advantages")
    results1 = test_gradient_flow_with_data(original_data, device)
    
    print("\n" + "=" * 60)
    print("Run 2: Modified advantages (group 0 scores swapped)")
    
    # Reset seed to get same model initialization
    torch.manual_seed(42)
    results2 = test_gradient_flow_with_data(modified_data, device)
    
    print("=" * 60)
    print("COMPARING RESULTS")
    print("=" * 60)
    
    print(f"Run 1 - Loss: {results1['final_loss']:.6f}, Grad norm: {results1['total_grad_norm']:.6f}")
    print(f"Run 2 - Loss: {results2['final_loss']:.6f}, Grad norm: {results2['total_grad_norm']:.6f}")
    
    loss_diff = abs(results1['final_loss'] - results2['final_loss'])
    grad_diff = abs(results1['total_grad_norm'] - results2['total_grad_norm'])
    
    print(f"Loss difference: {loss_diff:.6f}")
    print(f"Gradient norm difference: {grad_diff:.6f}")
    
    # Check that we get meaningful differences
    loss_changed = loss_diff > 1e-6
    gradients_changed = grad_diff > 1e-6
    
    print(f"Loss changed significantly: {loss_changed}")
    print(f"Gradients changed significantly: {gradients_changed}")
    print()
    
    return {
        'loss_changed': loss_changed,
        'gradients_changed': gradients_changed,
        'loss_diff': loss_diff,
        'grad_diff': grad_diff
    }


def test_gradient_flow_with_data(test_data, device):
    """Test gradient flow with specific test data"""
    args = MockArgs()
    tokenizer = MockTokenizer()
    
    responses = test_data['responses']
    queries = test_data['queries']
    masks = test_data['masks']
    finegrained_scores = test_data['finegrained_scores']
    
    # Compute advantages
    advantages_list, advantages_mask_list, reward_group_ids = compute_finegrained_advantages(finegrained_scores)
    
    # Pack sequences
    packed_sequences = pack_sequences(
        queries=queries,
        responses=responses,
        masks=masks,
        pack_length=args.pack_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Create packed advantages for each group
    packed_advantages_list = []
    packed_advantages_mask_list = []
    
    unique_groups = list(set(reward_group_ids))
    for group_idx, group_id in enumerate(sorted(unique_groups)):
        # Create lookup for this group's advantages
        group_advantages = advantages_list[group_idx]
        lookup_advantages = np.zeros(len(group_advantages) + 1, dtype=np.float32)
        lookup_advantages[1:] = group_advantages
        
        # Pack advantages for this group
        packed_advantages = [
            torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
            for packed_mask in packed_sequences.response_masks
        ]
        packed_advantages_list.append(packed_advantages)
        
        # Create span masks based on effective spans
        packed_span_masks = []
        for i, packed_mask in enumerate(packed_sequences.response_masks):
            # Simplified span mask creation for testing
            span_mask = torch.zeros_like(packed_mask, dtype=torch.bool)
            
            # For testing, create a simple pattern based on the group
            response_token_count = 0
            for j, mask_val in enumerate(packed_mask):
                if mask_val > 0:  # This is a response token
                    # Alternate between groups for different token positions
                    if group_id == 0:
                        # Group 0: first half of response tokens
                        span_mask[j] = response_token_count % 2 == 0
                    else:
                        # Group 1: second half of response tokens  
                        span_mask[j] = response_token_count % 2 == 1
                    response_token_count += 1
            
            # Ensure we have at least some True values
            if not span_mask.any():
                # Fallback: set every other response token to True
                for j, mask_val in enumerate(packed_mask):
                    if mask_val > 0 and j % 2 == group_id:
                        span_mask[j] = True
            
            packed_span_masks.append(span_mask)
        packed_advantages_mask_list.append(packed_span_masks)
    
    # Store in expected format
    packed_sequences.advantages = [(packed_advantages_list, packed_advantages_mask_list)]
    
    # Run the test
    return test_single_forward_pass(MockModel().to(device), packed_sequences, args, tokenizer, device)


def test_gradient_accumulation():
    """Test that gradients accumulate correctly across multiple groups"""
    print("=" * 60)
    print("TESTING GRADIENT ACCUMULATION ACROSS GROUPS")
    print("=" * 60)
    
    # Prepare data
    packed_sequences, args, tokenizer = prepare_mock_training_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test 1: Compute loss for all groups together
    model1 = MockModel().to(device)
    loss_all_groups = test_single_forward_pass(model1, packed_sequences, args, tokenizer, device)
    
    print(f"All groups together - Loss: {loss_all_groups['final_loss']:.6f}")
    print(f"All groups together - Grad norm: {loss_all_groups['total_grad_norm']:.6f}")
    print()
    
    # Test 2: Compute loss for each group separately and sum
    model2 = MockModel().to(device)
    model2.load_state_dict(model1.state_dict())  # Same initialization
    
    packed_advantages_list, packed_advantages_mask_list = packed_sequences.advantages[0]
    individual_losses = []
    
    for group_idx in range(len(packed_advantages_list)):
        print(f"Processing group {group_idx} individually")
        
        # Create temporary packed sequences with only this group
        temp_packed_sequences = type(packed_sequences)(**{
            attr: getattr(packed_sequences, attr) for attr in dir(packed_sequences) 
            if not attr.startswith('_') and attr != 'advantages'
        })
        temp_packed_sequences.advantages = [([packed_advantages_list[group_idx]], [packed_advantages_mask_list[group_idx]])]
        
        group_result = test_single_forward_pass(model2, temp_packed_sequences, args, tokenizer, device)
        individual_losses.append(group_result['final_loss'])
        
        print(f"  Group {group_idx} loss: {group_result['final_loss']:.6f}")
    
    total_individual_loss = sum(individual_losses) / len(individual_losses)  # Mean like in actual implementation
    
    print(f"\nIndividual groups mean loss: {total_individual_loss:.6f}")
    print(f"Difference: {abs(loss_all_groups['final_loss'] - total_individual_loss):.8f}")
    
    # Check that losses are approximately equal (accounting for floating point precision)
    losses_match = abs(loss_all_groups['final_loss'] - total_individual_loss) < 1e-3
    
    print(f"Losses match (within 1e-3): {losses_match}")
    print()
    
    return {
        'all_groups_loss': loss_all_groups['final_loss'],
        'individual_mean_loss': total_individual_loss,
        'losses_match': losses_match,
        'difference': abs(loss_all_groups['final_loss'] - total_individual_loss)
    }


def test_single_forward_pass(model, packed_sequences, args, tokenizer, device):
    """Helper function to run a single forward pass and return results"""
    # Similar to test_gradient_flow but simplified
    B = len(packed_sequences.query_responses) // args.world_size
    i = 0
    
    packed_advantages_list, packed_advantages_mask_list = packed_sequences.advantages[0]
    
    # Collate data
    query_responses = packed_sequences.query_responses[B * i : B * (i + 1)]
    attention_masks = packed_sequences.attention_masks[B * i : B * (i + 1)]
    position_ids = packed_sequences.position_ids[B * i : B * (i + 1)]
    response_masks = packed_sequences.response_masks[B * i : B * (i + 1)]
    
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
    new_logprobs = log_softmax_and_gather(logits, input_ids[:, 1:])
    
    # Simple reference (for testing, just use the current model)
    with torch.no_grad():
        ref_logprobs = new_logprobs.detach()
    
    # Compute loss for each group
    response_masks_bool = response_masks_tensor[:, 1:].bool()
    loss_list = []
    
    for j, (advantages, advantages_mask) in enumerate(zip(collated_advantages_list, collated_advantages_mask_list)):
        group_new_logprobs = torch.masked_fill(new_logprobs, ~advantages_mask[:, 1:], INVALID_LOGPROB)
        old_logprobs = group_new_logprobs.detach()
        
        logprobs_diff = group_new_logprobs - old_logprobs
        ratio = torch.exp(logprobs_diff)
        
        pg_losses = -advantages[:, 1:] * ratio
        pg_losses2 = -advantages[:, 1:] * torch.clamp(
            ratio, 1.0 - args.clip_lower, 1.0 + args.clip_higher
        )
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        
        # Simple KL for testing
        kl = torch.zeros_like(pg_loss_max)
        combined_loss = pg_loss_max + (args.beta * kl)
        
        mask_for_loss = response_masks_bool & advantages_mask[:, 1:]
        group_loss = masked_mean(combined_loss, mask_for_loss, args.masked_mean_axis)
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


def run_all_gradient_tests():
    """Run all gradient tests"""
    print("🧪 STARTING FINEGRAINED GRPO GRADIENT TESTS 🧪")
    print()
    
    try:
        # Test 1: Basic gradient flow
        print("Test 1: Basic gradient flow")
        results1 = test_gradient_flow()
        
        # Verify basic requirements
        assert results1['has_gradients'], "Model should have non-zero gradients"
        assert results1['gradients_finite'], "Gradients should be finite"
        assert results1['loss_finite'], "Loss should be finite"
        
        print("✅ Test 1: Basic gradient flow - PASSED")
        print()
        
        # Test 2: Gradient differences
        print("Test 2: Gradient differences between advantage patterns")
        results2 = test_gradient_differences()
        
        # Verify that different advantages produce different results
        assert results2['loss_changed'], "Different advantages should produce different losses"
        assert results2['gradients_changed'], "Different advantages should produce different gradients"
        
        print("✅ Test 2: Gradient differences - PASSED")
        print()
        
        # Test 3: Gradient accumulation
        print("Test 3: Gradient accumulation across groups")
        results3 = test_gradient_accumulation()
        
        # Verify that group-wise computation is consistent
        assert results3['losses_match'], "Individual group losses should sum to total loss"
        
        print("✅ Test 3: Gradient accumulation - PASSED")
        print()
        
        print("=" * 60)
        print("🎉 ALL GRADIENT TESTS PASSED! 🎉")
        print("The finegrained GRPO gradient computation is working correctly!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ GRADIENT TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_gradient_tests()
    exit(0 if success else 1) 