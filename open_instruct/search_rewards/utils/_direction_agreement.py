from typing import Dict, Any, List
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict


def normalize_rewards(rewards: List[float]) -> List[float]:
    return (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

def compute_direction_agreement(log_values: Dict[str, Any], verifiable_rewards: List[float], filter_responses_with_all_zero_rewards: bool = True) -> Dict[str, Any]:
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

def compute_direction_agreement_per_prompt(
    prompts: List[str],
    persistent_rubric_rewards: List[float],
    adaptive_rubric_rewards: List[float],
) -> Dict[str, Any]:
    """
    Computes the directional agreement between persistent rubric rewards and adaptive rubric rewards
    for each unique prompt. The agreement is measured by Spearman's rank correlation coefficient.
    
    This function groups responses by prompt and computes how well the rankings agree between
    the two reward types. A high correlation means both reward types rank responses similarly
    for that prompt.

    Args:
        prompts: A list of prompts (one per response).
        persistent_rubric_rewards: A list of persistent rubric rewards for each response.
        adaptive_rubric_rewards: A list of adaptive rubric rewards for each response.

    Returns:
        A dictionary containing:
        - spearman_mean: Mean Spearman correlation across all prompts
        - spearman_median: Median Spearman correlation
        - spearman_std: Standard deviation of correlations
        - spearman_min: Minimum correlation
        - spearman_max: Maximum correlation
        - kendall_tau_mean: Mean Kendall's tau correlation
        - num_prompts: Number of unique prompts
        - num_prompts_with_single_response: Number of prompts with only 1 response (skipped)
        - num_prompts_with_variance_zero: Number of prompts where rewards have zero variance
        - fraction_positive_correlation: Fraction of prompts with positive correlation
        - fraction_strong_agreement: Fraction of prompts with correlation > 0.7
    """
    if not (len(prompts) == len(persistent_rubric_rewards) == len(adaptive_rubric_rewards)):
        raise ValueError(f"All input lists must have the same length. Got: {len(prompts)=}, {len(persistent_rubric_rewards)=}, {len(adaptive_rubric_rewards)=}")

    # Group by prompt
    prompt_data = defaultdict(lambda: {"persistent_rewards": [], "adaptive_rewards": []})
    
    for i, prompt in enumerate(prompts):
        prompt_data[prompt]["persistent_rewards"].append(persistent_rubric_rewards[i])
        prompt_data[prompt]["adaptive_rewards"].append(adaptive_rubric_rewards[i])

    # Compute correlations per prompt
    spearman_correlations = []
    kendall_correlations = []
    num_single_response = 0
    num_zero_variance = 0
    
    for prompt, data in prompt_data.items():
        n_responses = len(data["persistent_rewards"])
        
        if n_responses < 2:
            # Need at least two data points to compute correlation
            num_single_response += 1
            continue
        
        persistent = np.array(data["persistent_rewards"])
        adaptive = np.array(data["adaptive_rewards"])
        
        # Check for zero variance (all rewards are the same)
        if np.std(persistent) < 1e-8 or np.std(adaptive) < 1e-8:
            num_zero_variance += 1
            continue
        
        # Calculate Spearman's rank correlation
        try:
            spearman_corr, _ = spearmanr(persistent, adaptive)
            if not np.isnan(spearman_corr):
                spearman_correlations.append(spearman_corr)
        except (ValueError, Exception) as e:
            # Handle cases where correlation cannot be computed
            pass
        
        # Calculate Kendall's tau (another rank correlation measure)
        try:
            from scipy.stats import kendalltau
            kendall_corr, _ = kendalltau(persistent, adaptive)
            if not np.isnan(kendall_corr):
                kendall_correlations.append(kendall_corr)
        except (ValueError, Exception) as e:
            pass
    
    # Compile results
    results = {
        "num_prompts": len(prompt_data),
        "num_prompts_with_single_response": num_single_response,
        "num_prompts_with_variance_zero": num_zero_variance,
        "num_prompts_analyzed": len(spearman_correlations),
    }
    
    if spearman_correlations:
        spearman_arr = np.array(spearman_correlations)
        results["spearman_mean"] = np.mean(spearman_arr)
        results["spearman_median"] = np.median(spearman_arr)
        results["spearman_std"] = np.std(spearman_arr)
        results["spearman_min"] = np.min(spearman_arr)
        results["spearman_max"] = np.max(spearman_arr)
        results["fraction_positive_correlation"] = np.mean(spearman_arr > 0)
        results["fraction_strong_agreement"] = np.mean(spearman_arr > 0.7)
        results["fraction_disagreement"] = np.mean(spearman_arr < -0.3)
    else:
        # No valid correlations computed
        for key in ["spearman_mean", "spearman_median", "spearman_std", "spearman_min", "spearman_max",
                    "fraction_positive_correlation", "fraction_strong_agreement", "fraction_disagreement"]:
            results[key] = np.nan
    
    if kendall_correlations:
        kendall_arr = np.array(kendall_correlations)
        results["kendall_tau_mean"] = np.mean(kendall_arr)
        results["kendall_tau_median"] = np.median(kendall_arr)
    else:
        results["kendall_tau_mean"] = np.nan
        results["kendall_tau_median"] = np.nan
    
    return results