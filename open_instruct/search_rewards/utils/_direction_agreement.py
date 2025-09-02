from typing import Dict, Any, List
import numpy as np


def normalize_rewards(rewards: List[float]) -> List[float]:
    return (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

def compute_direction_agreement(log_values: Dict[str, Any], verifiable_rewards: List[float]) -> float:
    fine_grained_keys = [
        "format_reward",
        "num_search_turns_reward",
        "rubric_reward",
        "citation_reward",
    ]
    
    global_rewards = normalize_rewards(verifiable_rewards)
    direction_agreement_dict = {}
    for key in fine_grained_keys:
        finegrained_rewards = normalize_rewards(np.array(log_values[key]))
        direction_agreement = np.corrcoef(finegrained_rewards, global_rewards)[0, 1]
        print(f"Direction agreement between {key} and global: {direction_agreement}")
        direction_agreement_dict[f'global_vs_{key}'] = direction_agreement
    direction_agreement_dict['averaged_global_vs_fine_grained'] = np.mean(list(direction_agreement_dict.values()))
    
    for i, key1 in enumerate(fine_grained_keys):
        for key2 in fine_grained_keys[i+1:]:
            r1 = normalize_rewards(np.array(log_values[key1]))
            r2 = normalize_rewards(np.array(log_values[key2]))
            pairwise_corr = np.corrcoef(r1, r2)[0, 1]
            direction_agreement_dict[f'pairwise_{key1}_vs_{key2}'] = pairwise_corr
            
    return direction_agreement_dict
