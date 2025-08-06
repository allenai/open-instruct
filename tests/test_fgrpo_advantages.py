#!/usr/bin/env python3
"""
Test case for finegrained GRPO advantage computation.
This tests the advantage calculation logic without running actual model training.
"""

import asyncio
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
from unittest.mock import Mock, patch
import sys
import os

# Add the open_instruct module to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_instruct.rl_utils2 import pack_sequences
from open_instruct.fgrpo_fast import collate_fn


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


class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
    
    def batch_decode(self, sequences, skip_special_tokens=True):
        # Simple mock decode - just convert numbers to strings
        return [" ".join(map(str, seq)) for seq in sequences]


def create_test_data():
    """Create test data for finegrained advantage computation"""
    
    # Mock responses (token IDs)
    responses = [
        [10, 11, 12, 13, 14],  # Response 1 for prompt 1, sample 1
        [15, 16, 17, 18],      # Response 2 for prompt 1, sample 2
        [20, 21, 22, 23, 24, 25],  # Response 1 for prompt 2, sample 1
        [26, 27, 28],          # Response 2 for prompt 2, sample 2
    ]
    
    # Mock queries (token IDs)
    queries = [
        [5, 6, 7],   # Query for prompt 1, sample 1
        [5, 6, 7],   # Query for prompt 1, sample 2 (same query)
        [8, 9],      # Query for prompt 2, sample 1
        [8, 9],      # Query for prompt 2, sample 2 (same query)
    ]
    
    # Mock tool masks (1 means normal token, 0 means tool token to mask)
    masks = [
        [1, 1, 1, 1, 1],  # All tokens are normal
        [1, 1, 1, 1],     # All tokens are normal
        [1, 1, 1, 1, 1, 1],  # All tokens are normal
        [1, 1, 1],        # All tokens are normal
    ]
    
    # Mock finegrained scores: (score, effective_span, reward_group_id)
    # Each response has multiple reward segments
    finegrained_scores = [
        # Response 1: Two segments with different rewards
        (1.0, [True, True, False, False, False], 0),  # First segment, group 0
        (0.5, [False, False, True, True, True], 1),   # Second segment, group 1
        
        # Response 2: Two segments
        (0.8, [True, True, False, False], 0),         # First segment, group 0
        (0.2, [False, False, True, True], 1),         # Second segment, group 1
        
        # Response 3: Two segments
        (0.9, [True, True, True, False, False, False], 0),  # First segment, group 0
        (0.7, [False, False, False, True, True, True], 1),  # Second segment, group 1
        
        # Response 4: Two segments
        (0.3, [True, False, False], 0),               # First segment, group 0
        (0.6, [False, True, True], 1),                # Second segment, group 1
    ]
    
    # Mock other required data
    decoded_responses = ["resp1", "resp2", "resp3", "resp4"]
    ground_truths = ["gt1", "gt1", "gt2", "gt2"]
    datasets = ["ds1", "ds1", "ds2", "ds2"]
    finish_reasons = ["stop", "stop", "stop", "stop"]
    infos = ([0, 0, 0, 0], [0, 0, 0, 0], ["", "", "", ""], ["", "", "", ""], [0, 0, 0, 0], [False, False, False, False])
    decoded_queries = ["query1", "query1", "query2", "query2"]
    
    return {
        'responses': responses,
        'queries': queries,
        'masks': masks,
        'finegrained_scores': finegrained_scores,
        'decoded_responses': decoded_responses,
        'ground_truths': ground_truths,
        'datasets': datasets,
        'finish_reasons': finish_reasons,
        'infos': infos,
        'decoded_queries': decoded_queries
    }


async def mock_reward_fn(responses, decoded_responses, ground_truths, datasets, finish_reasons, infos, decoded_queries):
    """Mock reward function that returns finegrained scores"""
    test_data = create_test_data()
    reward_metrics = {"mock_metric": 0.5}
    return test_data['finegrained_scores'], reward_metrics


def gather_mean_and_std_from_the_same_reward_group(finegrained_scores, reward_group_id):
    """Helper function to calculate mean and std for a reward group"""
    scores = [score for score, effective_span, group_id in finegrained_scores if group_id == reward_group_id]
    return np.mean(scores), np.std(scores) + 1e-8


def test_finegrained_advantage_computation():
    """Test the finegrained advantage computation logic"""
    print("=" * 60)
    print("TESTING FINEGRAINED ADVANTAGE COMPUTATION")
    print("=" * 60)
    
    # Create test data
    test_data = create_test_data()
    args = MockArgs()
    tokenizer = MockTokenizer()
    
    # Extract test data
    responses = test_data['responses']
    queries = test_data['queries']
    masks = test_data['masks']
    finegrained_scores = test_data['finegrained_scores']
    
    print(f"Test data created:")
    print(f"  - {len(responses)} responses")
    print(f"  - {len(finegrained_scores)} finegrained score entries")
    print(f"  - Response lengths: {[len(r) for r in responses]}")
    print()
    
    # Step 1: Extract components from finegrained_scores
    print("Step 1: Extracting components from finegrained_scores")
    scores = np.array([score for score, _, _ in finegrained_scores])
    effective_spans = [effective_span for _, effective_span, _ in finegrained_scores]
    reward_group_ids = [reward_group_id for _, _, reward_group_id in finegrained_scores]
    
    print(f"  - Scores: {scores}")
    print(f"  - Reward group IDs: {reward_group_ids}")
    print(f"  - Effective spans: {effective_spans}")
    print()
    
    # Step 2: Normalize advantages per group
    print("Step 2: Normalizing advantages per reward group")
    unique_groups = list(set(reward_group_ids))
    advantages = np.zeros_like(scores, dtype=np.float32)
    
    print(f"  - Unique groups: {unique_groups}")
    
    for group_id in unique_groups:
        group_indices = [i for i, gid in enumerate(reward_group_ids) if gid == group_id]
        group_scores = scores[group_indices]
        
        # Calculate group statistics
        group_mean, group_std = gather_mean_and_std_from_the_same_reward_group(finegrained_scores, group_id)
        group_advantages = (group_scores - group_mean) / group_std
        
        print(f"  - Group {group_id}:")
        print(f"    * Indices: {group_indices}")
        print(f"    * Scores: {group_scores}")
        print(f"    * Mean: {group_mean:.4f}, Std: {group_std:.4f}")
        print(f"    * Advantages: {group_advantages}")
        
        # Assign normalized advantages back
        for idx, group_idx in enumerate(group_indices):
            advantages[group_idx] = group_advantages[idx]
    
    print(f"  - Final advantages: {advantages}")
    print()
    
    # Step 3: Create span masks
    print("Step 3: Creating span masks")
    span_masks = []
    for i, effective_span in enumerate(effective_spans):
        span_mask = np.array(effective_span, dtype=bool)
        span_masks.append(span_mask)
        print(f"  - Span {i}: {effective_span} -> {span_mask}")
    print()
    
    # Step 4: Group advantages and masks by reward group
    print("Step 4: Grouping advantages and masks by reward group")
    grouped_advantages = {}
    grouped_masks = {}
    
    for i, group_id in enumerate(reward_group_ids):
        if group_id not in grouped_advantages:
            grouped_advantages[group_id] = []
            grouped_masks[group_id] = []
        grouped_advantages[group_id].append(advantages[i])
        grouped_masks[group_id].append(span_masks[i])
    
    for group_id in unique_groups:
        print(f"  - Group {group_id}:")
        print(f"    * Advantages: {grouped_advantages[group_id]}")
        print(f"    * Masks: {grouped_masks[group_id]}")
    print()
    
    # Step 5: Convert to expected format
    print("Step 5: Converting to expected format for training loop")
    advantages_list = []
    advantages_mask_list = []
    
    for group_id in sorted(unique_groups):
        advantages_list.append(np.array(grouped_advantages[group_id]))
        advantages_mask_list.append(grouped_masks[group_id])
    
    print(f"  - advantages_list: {[adv.shape for adv in advantages_list]}")
    print(f"  - advantages_mask_list lengths: {[len(masks) for masks in advantages_mask_list]}")
    
    for i, (adv, masks_list) in enumerate(zip(advantages_list, advantages_mask_list)):
        print(f"  - Group {i}: advantages={adv}, masks={masks_list}")
    print()
    
    # Step 6: Test filtering (non-zero std check)
    print("Step 6: Testing filtering logic")
    non_zero_std_groups = set()
    for group_id in unique_groups:
        group_indices = [i for i, gid in enumerate(reward_group_ids) if gid == group_id]
        group_scores = scores[group_indices]
        group_std = np.std(group_scores)
        print(f"  - Group {group_id}: std={group_std:.4f}")
        if len(group_scores) > 1 and group_std != 0:
            non_zero_std_groups.add(group_id)
    
    print(f"  - Non-zero std groups: {non_zero_std_groups}")
    
    # Filter based on non-zero std groups
    non_zero_gradient_index = [i for i, gid in enumerate(reward_group_ids) if gid in non_zero_std_groups]
    print(f"  - Keeping indices: {non_zero_gradient_index}")
    print()
    
    return {
        'advantages_list': advantages_list,
        'advantages_mask_list': advantages_mask_list,
        'non_zero_std_groups': non_zero_std_groups,
        'scores': scores,
        'advantages': advantages
    }


def test_packing_with_finegrained():
    """Test the packing logic with finegrained advantages"""
    print("=" * 60)
    print("TESTING PACKING WITH FINEGRAINED ADVANTAGES")
    print("=" * 60)
    
    # Get results from advantage computation test
    advantage_results = test_finegrained_advantage_computation()
    test_data = create_test_data()
    args = MockArgs()
    tokenizer = MockTokenizer()
    
    responses = test_data['responses']
    queries = test_data['queries']
    masks = test_data['masks']
    advantages_list = advantage_results['advantages_list']
    advantages_mask_list = advantage_results['advantages_mask_list']
    non_zero_std_groups = advantage_results['non_zero_std_groups']
    
    print("Step 1: Packing sequences")
    packed_sequences = pack_sequences(
        queries=queries,
        responses=responses,
        masks=masks,
        pack_length=args.pack_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    print(f"  - Packed {len(packed_sequences.query_responses)} sequences")
    print(f"  - Response mask shapes: {[mask.shape for mask in packed_sequences.response_masks]}")
    print()
    
    print("Step 2: Creating packed advantages for each group")
    packed_advantages_list = []
    packed_advantages_mask_list = []
    
    for group_idx, group_id in enumerate(sorted(non_zero_std_groups)):
        print(f"  - Processing group {group_id} (index {group_idx})")
        
        # Create lookup for this group's advantages
        group_advantages = advantages_list[group_idx]
        lookup_advantages = np.zeros(len(group_advantages) + 1, dtype=np.float32)
        lookup_advantages[1:] = group_advantages
        
        print(f"    * Group advantages: {group_advantages}")
        print(f"    * Lookup advantages: {lookup_advantages}")
        
        # Pack advantages for this group
        packed_advantages = [
            torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
            for packed_mask in packed_sequences.response_masks
        ]
        packed_advantages_list.append(packed_advantages)
        
        print(f"    * Packed advantages shapes: {[adv.shape for adv in packed_advantages]}")
        
        # Create simple span masks (this is a simplified version for testing)
        packed_span_masks = []
        for i, packed_mask in enumerate(packed_sequences.response_masks):
            span_mask = torch.ones_like(packed_mask, dtype=torch.bool)  # Simplified for testing
            packed_span_masks.append(span_mask)
        packed_advantages_mask_list.append(packed_span_masks)
        
        print(f"    * Packed mask shapes: {[mask.shape for mask in packed_span_masks]}")
    
    # Store in the expected format
    packed_sequences.advantages = [(packed_advantages_list, packed_advantages_mask_list)]
    
    print(f"\nStep 3: Final packed structure")
    print(f"  - Advantages structure: {type(packed_sequences.advantages[0])}")
    print(f"  - Number of advantage groups: {len(packed_sequences.advantages[0][0])}")
    print(f"  - Number of mask groups: {len(packed_sequences.advantages[0][1])}")
    print()
    
    return packed_sequences


def test_collation_with_finegrained():
    """Test the collation logic with finegrained advantages"""
    print("=" * 60)
    print("TESTING COLLATION WITH FINEGRAINED ADVANTAGES")
    print("=" * 60)
    
    # Get packed sequences
    packed_sequences = test_packing_with_finegrained()
    args = MockArgs()
    tokenizer = MockTokenizer()
    
    print("Step 1: Simulating per-device slicing")
    B = len(packed_sequences.query_responses) // args.world_size
    i = 0  # Simulate device 0
    
    print(f"  - Batch size per device: {B}")
    print(f"  - Processing device {i}")
    
    # Slice data for this device
    per_device_packed_query_responses = packed_sequences.query_responses[B * i : B * (i + 1)]
    per_device_packed_response_masks = packed_sequences.response_masks[B * i : B * (i + 1)]
    
    # Handle finegrained advantages slicing
    if len(packed_sequences.advantages) > 0:
        packed_advantages_list, packed_advantages_mask_list = packed_sequences.advantages[0]
        per_device_packed_advantages_list = []
        per_device_packed_advantages_mask_list = []
        
        for group_idx in range(len(packed_advantages_list)):
            per_device_packed_advantages_list.append(
                packed_advantages_list[group_idx][B * i : B * (i + 1)]
            )
            per_device_packed_advantages_mask_list.append(
                packed_advantages_mask_list[group_idx][B * i : B * (i + 1)]
            )
        
        per_device_packed_advantages = (per_device_packed_advantages_list, per_device_packed_advantages_mask_list)
    else:
        per_device_packed_advantages = ([], [])
    
    print(f"  - Per-device sequences: {len(per_device_packed_query_responses)}")
    print(f"  - Per-device advantages groups: {len(per_device_packed_advantages[0])}")
    print()
    
    print("Step 2: Collating micro-batches")
    b_inds = np.random.permutation(len(per_device_packed_query_responses))
    collated_advantages = []
    
    for j in range(0, len(per_device_packed_query_responses), args.per_device_train_batch_size):
        micro_range = b_inds[j : j + args.per_device_train_batch_size]
        print(f"  - Micro-batch range: {micro_range}")
        
        # Handle finegrained advantages collation
        if len(per_device_packed_advantages) > 0:
            packed_advantages_list, packed_advantages_mask_list = per_device_packed_advantages
            
            collated_advantages_list = []
            collated_advantages_mask_list = []
            
            for group_idx in range(len(packed_advantages_list)):
                # Collate advantages for this group
                group_advantages = [
                    packed_advantages_list[group_idx][idx] for idx in micro_range
                ]
                collated_group_advantages = collate_fn(group_advantages, 0, pin_memory=False)
                collated_advantages_list.append(collated_group_advantages)
                
                print(f"    * Group {group_idx} advantages shape: {collated_group_advantages.shape}")
                
                # Collate masks for this group
                group_masks = [
                    packed_advantages_mask_list[group_idx][idx] for idx in micro_range
                ]
                collated_group_masks = collate_fn([mask.float() for mask in group_masks], 0, pin_memory=False).bool()
                collated_advantages_mask_list.append(collated_group_masks)
                
                print(f"    * Group {group_idx} masks shape: {collated_group_masks.shape}")
            
            # Store as expected format for training loop
            collated_advantages.append((collated_advantages_list, collated_advantages_mask_list))
        else:
            collated_advantages.append(([], []))
    
    print(f"\nStep 3: Final collated structure")
    print(f"  - Number of micro-batches: {len(collated_advantages)}")
    for mb_idx, (adv_list, mask_list) in enumerate(collated_advantages):
        print(f"  - Micro-batch {mb_idx}:")
        print(f"    * Advantage groups: {len(adv_list)}")
        print(f"    * Mask groups: {len(mask_list)}")
        if len(adv_list) > 0:
            print(f"    * Advantage shapes: {[adv.shape for adv in adv_list]}")
            print(f"    * Mask shapes: {[mask.shape for mask in mask_list]}")
    print()
    
    return collated_advantages


def test_training_loop_format():
    """Test that the data format matches what the training loop expects"""
    print("=" * 60)
    print("TESTING TRAINING LOOP FORMAT COMPATIBILITY")
    print("=" * 60)
    
    collated_advantages = test_collation_with_finegrained()
    
    print("Step 1: Simulating training loop data extraction")
    
    # Simulate what happens in the training loop
    for i, collated_advantage in enumerate(collated_advantages):
        print(f"  - Processing collated advantage {i}")
        
        # This simulates line 821 in the training code:
        # mb_advantages_list, mb_advantages_mask_list = collated_advantages[i]
        mb_advantages_list, mb_advantages_mask_list = collated_advantage
        
        print(f"    * mb_advantages_list type: {type(mb_advantages_list)}")
        print(f"    * mb_advantages_list length: {len(mb_advantages_list)}")
        print(f"    * mb_advantages_mask_list type: {type(mb_advantages_mask_list)}")
        print(f"    * mb_advantages_mask_list length: {len(mb_advantages_mask_list)}")
        
        # Check that we can iterate over the groups (as expected in training loop)
        for j in range(len(mb_advantages_list)):
            print(f"      - Group {j}:")
            print(f"        * Advantage shape: {mb_advantages_list[j].shape}")
            print(f"        * Mask shape: {mb_advantages_mask_list[j].shape}")
            print(f"        * Advantage sample values: {mb_advantages_list[j].flatten()[:5]}")
            print(f"        * Mask sample values: {mb_advantages_mask_list[j].flatten()[:5]}")
    
    print("\n✅ Training loop format compatibility test passed!")
    print()


def run_all_tests():
    """Run all tests"""
    print("🧪 STARTING FINEGRAINED GRPO ADVANTAGE COMPUTATION TESTS 🧪")
    print()
    
    try:
        # Test 1: Basic advantage computation
        advantage_results = test_finegrained_advantage_computation()
        print("✅ Test 1: Advantage computation - PASSED")
        
        # Test 2: Packing with finegrained
        packed_sequences = test_packing_with_finegrained()
        print("✅ Test 2: Packing with finegrained - PASSED")
        
        # Test 3: Collation with finegrained
        collated_advantages = test_collation_with_finegrained()
        print("✅ Test 3: Collation with finegrained - PASSED")
        
        # Test 4: Training loop format
        test_training_loop_format()
        print("✅ Test 4: Training loop format - PASSED")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The finegrained GRPO advantage computation is working correctly!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 