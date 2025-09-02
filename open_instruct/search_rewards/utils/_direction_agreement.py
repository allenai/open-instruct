from typing import Dict, Any, List
import numpy as np


def normalize_rewards(rewards: List[float]) -> List[float]:
    return (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

def compute_direction_agreement(log_values: Dict[str, Any], verifiable_rewards: List[float], filter_responses_with_all_zero_rewards: bool = True) -> float:
    fine_grained_keys = [
        "format_reward",
        "num_search_turns_reward",
        "rubric_reward",
        "citation_reward",
    ]
    
    # Normalize first
    global_rewards = normalize_rewards(verifiable_rewards)
    
    # Determine filter indices based on ORIGINAL values
    if filter_responses_with_all_zero_rewards:
        response_indices = np.array(verifiable_rewards) != 0
        filtered_global_rewards = global_rewards[response_indices]
        
        # Edge case: if all rewards are zero or only one example left
        if len(filtered_global_rewards) <= 1:
            print(f"Warning: Only {len(filtered_global_rewards)} examples left after filtering. Returning NaN correlations.")
            direction_agreement_dict = {}
            for key in fine_grained_keys:
                direction_agreement_dict[f'global_vs_{key}'] = np.nan
            direction_agreement_dict['averaged_global_vs_fine_grained'] = np.nan
            for i, key1 in enumerate(fine_grained_keys):
                for key2 in fine_grained_keys[i+1:]:
                    direction_agreement_dict[f'pairwise_{key1}_vs_{key2}'] = np.nan
            return direction_agreement_dict
    else:
        response_indices = np.ones(len(verifiable_rewards), dtype=bool)
        filtered_global_rewards = global_rewards
    
    direction_agreement_dict = {}
    for key in fine_grained_keys:
        # Normalize first, then filter
        finegrained_rewards = normalize_rewards(np.array(log_values[key]))
        filtered_finegrained_rewards = finegrained_rewards[response_indices]
        
        direction_agreement = np.corrcoef(filtered_finegrained_rewards, filtered_global_rewards)[0, 1]
        print(f"Direction agreement between {key} and global: {direction_agreement}")
        direction_agreement_dict[f'global_vs_{key}'] = direction_agreement
    
    direction_agreement_dict['averaged_global_vs_fine_grained'] = np.mean(list(direction_agreement_dict.values()))
    
    for i, key1 in enumerate(fine_grained_keys):
        for key2 in fine_grained_keys[i+1:]:
            # Normalize first, then filter
            r1 = normalize_rewards(np.array(log_values[key1]))
            r2 = normalize_rewards(np.array(log_values[key2]))
            filtered_r1 = r1[response_indices]
            filtered_r2 = r2[response_indices]
            
            pairwise_corr = np.corrcoef(filtered_r1, filtered_r2)[0, 1]
            direction_agreement_dict[f'pairwise_{key1}_vs_{key2}'] = pairwise_corr
            
    return direction_agreement_dict