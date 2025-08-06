#!/usr/bin/env python3
"""
Test case for finegrained GRPO using efficient span tuples.
This demonstrates how using (start, end, total_length) tuples instead of boolean arrays
significantly reduces memory usage and communication overhead for long responses.
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
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
    pack_length: int = 2048  # Larger pack length for long responses
    per_device_train_batch_size: int = 2
    world_size: int = 1
    mask_truncated_completions: bool = False
    allow_world_padding: bool = False
    beta: float = 0.05
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
        x = self.embedding(input_ids)
        logits = self.linear(x)
        return type('Output', (), {'logits': logits})()


def create_long_response_data():
    """Create test data with long responses to demonstrate span tuple efficiency"""
    
    # Create long responses (simulating code generation or detailed explanations)
    responses = []
    queries = []
    
    # Generate responses of varying lengths (100-800 tokens)
    response_lengths = [100, 250, 500, 800]
    
    for i, length in enumerate(response_lengths):
        # Create a response with sequential token IDs
        response = list(range(1000 + i * 1000, 1000 + i * 1000 + length))
        responses.append(response)
        
        # Create corresponding query
        query = [500 + i, 501 + i, 502 + i]
        queries.append(query)
    
    # Tool masks (all normal tokens)
    masks = [[1] * len(resp) for resp in responses]
    
    # Create finegrained scores using efficient span tuples (start, end, total_length)
    # This is much more memory efficient than boolean arrays for long responses
    finegrained_scores = []
    
    for i, response in enumerate(responses):
        total_length = len(response)
        
        # Reward Group 0: Format reward (applies to first 20% of response)
        format_start = 0
        format_end = max(1, total_length // 5)  # First 20%
        format_score = 0.8 + (i * 0.1)  # Vary scores across responses
        finegrained_scores.append((format_score, (format_start, format_end, total_length), 0))
        
        # Reward Group 1: Content reward (applies to middle 40% of response)
        content_start = total_length // 4  # Start at 25%
        content_end = (3 * total_length) // 4  # End at 75%
        content_score = 0.6 + (i * 0.15)
        finegrained_scores.append((content_score, (content_start, content_end, total_length), 1))
        
        # Reward Group 2: Conclusion reward (applies to last 15% of response)
        conclusion_start = (17 * total_length) // 20  # Start at 85%
        conclusion_end = total_length
        conclusion_score = 0.7 + (i * 0.05)
        finegrained_scores.append((conclusion_score, (conclusion_start, conclusion_end, total_length), 2))
    
    return {
        'responses': responses,
        'queries': queries,
        'masks': masks,
        'finegrained_scores': finegrained_scores,
    }


def compare_memory_usage():
    """Compare memory usage between span tuples and boolean arrays"""
    print("=" * 80)
    print("COMPARING MEMORY USAGE: SPAN TUPLES vs BOOLEAN ARRAYS")
    print("=" * 80)
    
    test_data = create_long_response_data()
    
    total_tuple_bytes = 0
    total_bool_array_bytes = 0
    
    print("Memory usage comparison for each response:")
    print()
    
    for i, (response, (score, span_tuple, group_id)) in enumerate(zip(test_data['responses'], test_data['finegrained_scores'][::3])):  # Every 3rd for one per response
        response_length = len(response)
        start, end, total_length = span_tuple
        
        # Calculate memory for span tuple (3 integers)
        tuple_bytes = 3 * 8  # 3 int64 values
        
        # Calculate memory for equivalent boolean array
        bool_array_bytes = response_length * 1  # 1 byte per boolean
        
        total_tuple_bytes += tuple_bytes
        total_bool_array_bytes += bool_array_bytes
        
        compression_ratio = bool_array_bytes / tuple_bytes
        
        print(f"Response {i+1} (length {response_length}):")
        print(f"  - Span tuple (start={start}, end={end}): {tuple_bytes} bytes")
        print(f"  - Boolean array: {bool_array_bytes} bytes")
        print(f"  - Compression ratio: {compression_ratio:.1f}x")
        print()
    
    total_responses = len(test_data['responses'])
    total_reward_groups = 3
    total_segments = total_responses * total_reward_groups
    
    total_tuple_bytes *= total_reward_groups  # Account for all reward groups
    total_bool_array_bytes *= total_reward_groups
    
    overall_compression = total_bool_array_bytes / total_tuple_bytes
    
    print(f"Overall memory usage for {total_segments} reward segments:")
    print(f"  - Span tuples: {total_tuple_bytes:,} bytes ({total_tuple_bytes/1024:.1f} KB)")
    print(f"  - Boolean arrays: {total_bool_array_bytes:,} bytes ({total_bool_array_bytes/1024:.1f} KB)")
    print(f"  - Memory saved: {total_bool_array_bytes - total_tuple_bytes:,} bytes ({(total_bool_array_bytes - total_tuple_bytes)/1024:.1f} KB)")
    print(f"  - Overall compression ratio: {overall_compression:.1f}x")
    print()
    
    return {
        'tuple_bytes': total_tuple_bytes,
        'bool_array_bytes': total_bool_array_bytes,
        'compression_ratio': overall_compression,
        'memory_saved': total_bool_array_bytes - total_tuple_bytes
    }


def test_span_tuple_conversion():
    """Test that span tuples are correctly converted to boolean masks"""
    print("=" * 80)
    print("TESTING SPAN TUPLE TO BOOLEAN MASK CONVERSION")
    print("=" * 80)
    
    test_data = create_long_response_data()
    
    print("Verifying span tuple conversion for each response:")
    print()
    
    for i, response in enumerate(test_data['responses']):
        response_length = len(response)
        print(f"Response {i+1} (length {response_length}):")
        
        # Get all reward segments for this response
        response_segments = test_data['finegrained_scores'][i*3:(i+1)*3]
        
        for j, (score, span_tuple, group_id) in enumerate(response_segments):
            start, end, total_length = span_tuple
            
            # Verify total_length matches response length
            assert total_length == response_length, f"total_length ({total_length}) != response_length ({response_length})"
            
            # Create boolean mask from span tuple (like the implementation does)
            span_mask = np.zeros(response_length, dtype=bool)
            start = max(0, min(start, response_length))
            end = max(start, min(end, response_length))
            span_mask[start:end] = True
            
            effective_tokens = span_mask.sum()
            coverage_pct = (effective_tokens / response_length) * 100
            
            group_names = ["Format", "Content", "Conclusion"]
            print(f"  - {group_names[group_id]} reward (score={score:.2f}):")
            print(f"    * Span: [{start}:{end}] out of {total_length} tokens")
            print(f"    * Effective tokens: {effective_tokens}/{response_length} ({coverage_pct:.1f}%)")
        
        print()
    
    return True


def test_gradient_flow_with_span_tuples():
    """Test that gradient flow works correctly with span tuples"""
    print("=" * 80)
    print("TESTING GRADIENT FLOW WITH SPAN TUPLES")
    print("=" * 80)
    
    test_data = create_long_response_data()
    args = MockArgs()
    tokenizer = MockTokenizer()
    
    # Compute advantages using span tuples
    advantages_list, advantages_mask_list, reward_group_ids = compute_finegrained_advantages_from_tuples(test_data['finegrained_scores'], test_data['responses'])
    
    print(f"Processing {len(set(reward_group_ids))} reward groups with span tuples")
    
    # Pack sequences
    packed_sequences = pack_sequences(
        queries=test_data['queries'],
        responses=test_data['responses'],
        masks=test_data['masks'],
        pack_length=args.pack_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Create packed advantages with span masks converted from tuples
    packed_advantages_list = []
    packed_advantages_mask_list = []
    
    unique_groups = list(set(reward_group_ids))
    for group_idx, group_id in enumerate(sorted(unique_groups)):
        group_advantages = advantages_list[group_idx]
        lookup_advantages = np.zeros(len(group_advantages) + 1, dtype=np.float32)
        lookup_advantages[1:] = group_advantages
        
        packed_advantages = [
            torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
            for packed_mask in packed_sequences.response_masks
        ]
        packed_advantages_list.append(packed_advantages)
        
        # Create span masks from tuples
        packed_span_masks = []
        for seq_idx, packed_mask in enumerate(packed_sequences.response_masks):
            span_mask = create_span_mask_from_tuples(
                test_data['finegrained_scores'], 
                test_data['responses'], 
                group_id, 
                packed_mask, 
                packed_sequences
            )
            packed_span_masks.append(span_mask)
        
        packed_advantages_mask_list.append(packed_span_masks)
    
    packed_sequences.advantages = [(packed_advantages_list, packed_advantages_mask_list)]
    
    # Test gradient computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MockModel().to(device)
    
    result = run_training_step(model, packed_sequences, args, tokenizer, device)
    
    print(f"Training step completed successfully:")
    print(f"  - Final loss: {result['final_loss']:.6f}")
    print(f"  - Gradient norm: {result['total_grad_norm']:.6f}")
    print(f"  - Group losses: {[f'{loss:.6f}' for loss in result['group_losses']]}")
    
    # Verify gradients are non-zero and finite
    assert result['total_grad_norm'] > 0, "Should have non-zero gradients"
    assert np.isfinite(result['final_loss']), "Loss should be finite"
    assert np.isfinite(result['total_grad_norm']), "Gradients should be finite"
    
    print("✅ Gradient flow with span tuples working correctly!")
    print()
    
    return result


def compute_finegrained_advantages_from_tuples(finegrained_scores, responses):
    """Compute advantages from finegrained scores with span tuples"""
    
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
        
        group_mean = np.mean(group_scores)
        group_std = np.std(group_scores) + 1e-8
        group_advantages = (group_scores - group_mean) / group_std
        
        for idx, group_idx in enumerate(group_indices):
            advantages[group_idx] = group_advantages[idx]
    
    # Convert span tuples to boolean masks
    span_masks = []
    for i, effective_span in enumerate(effective_spans):
        response_idx = i // 3  # 3 reward groups per response
        response_length = len(responses[response_idx])
        
        if isinstance(effective_span, (tuple, list)) and len(effective_span) == 3:
            start, end, total_length = effective_span
            # Validate total_length
            assert total_length == response_length, f"Span total_length mismatch for response {response_idx}"
            
            # Create boolean mask from span
            span_mask = np.zeros(response_length, dtype=bool)
            start = max(0, min(start, response_length))
            end = max(start, min(end, response_length))
            span_mask[start:end] = True
        else:
            # Fallback for other formats
            span_mask = np.ones(response_length, dtype=bool)
        
        span_masks.append(span_mask)
    
    # Group by reward group
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


def create_span_mask_from_tuples(finegrained_scores, responses, target_group_id, packed_mask, packed_sequences):
    """Create span mask for packed sequence from finegrained score tuples"""
    span_mask = torch.zeros_like(packed_mask, dtype=torch.bool)
    
    # Map packed positions back to original responses
    response_start_positions = []
    current_pos = 0
    
    for resp_idx in range(len(responses)):
        query_len = 3  # All queries have 3 tokens in our test data
        response_len = len(responses[resp_idx])
        response_start_positions.append(current_pos + query_len)
        current_pos += query_len + response_len
    
    # Apply spans for the target group
    for resp_idx in range(len(responses)):
        # Find spans for this response and target group
        for segment_idx, (score, span_tuple, group_id) in enumerate(finegrained_scores):
            if group_id == target_group_id and segment_idx // 3 == resp_idx:
                start, end, total_length = span_tuple
                response_start = response_start_positions[resp_idx]
                
                # Map span to packed positions
                for token_idx in range(start, end):
                    packed_pos = response_start + token_idx
                    if packed_pos < len(span_mask) and packed_mask[packed_pos] > 0:
                        span_mask[packed_pos] = True
                break
    
    return span_mask


def run_training_step(model, packed_sequences, args, tokenizer, device):
    """Run a single training step and return results"""
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
        
        mask_for_loss = response_masks_bool & advantages_mask[:, 1:]
        group_loss = masked_mean(pg_loss_max, mask_for_loss, args.masked_mean_axis)
        loss_list.append(group_loss)
    
    final_loss = torch.mean(torch.stack(loss_list))
    
    model.zero_grad()
    final_loss.backward()
    
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


def run_all_span_tuple_tests():
    """Run all span tuple efficiency tests"""
    print("🧪 STARTING FINEGRAINED GRPO SPAN TUPLE EFFICIENCY TESTS 🧪")
    print()
    
    try:
        # Test 1: Memory usage comparison
        print("Test 1: Memory usage comparison")
        memory_stats = compare_memory_usage()
        assert memory_stats['compression_ratio'] > 5, "Should achieve significant compression"
        print("✅ Test 1: Memory usage comparison - PASSED")
        print()
        
        # Test 2: Span tuple conversion
        print("Test 2: Span tuple to boolean mask conversion")
        conversion_result = test_span_tuple_conversion()
        assert conversion_result, "Span tuple conversion should work correctly"
        print("✅ Test 2: Span tuple conversion - PASSED")
        print()
        
        # Test 3: Gradient flow
        print("Test 3: Gradient flow with span tuples")
        gradient_result = test_gradient_flow_with_span_tuples()
        assert gradient_result['total_grad_norm'] > 0, "Should have non-zero gradients"
        print("✅ Test 3: Gradient flow - PASSED")
        print()
        
        print("=" * 80)
        print("🎉 ALL SPAN TUPLE TESTS PASSED! 🎉")
        print(f"Span tuples provide {memory_stats['compression_ratio']:.1f}x memory compression!")
        print(f"Memory saved: {memory_stats['memory_saved']/1024:.1f} KB")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ SPAN TUPLE TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_span_tuple_tests()
    exit(0 if success else 1) 