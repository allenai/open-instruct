#!/usr/bin/env python3
"""
Toy Training Script for Fine-Grained Token-Level Reward Control

This script demonstrates how to control token-level rewards and backpropagation
similar to grpo_fast and fgrpo_fast, but with pre-defined rollouts and rewards
for educational and debugging purposes.

Key Features:
- Pre-defined training rollouts (prompts and responses)
- Configurable fine-grained rewards at token level
- Control over which tokens receive which rewards
- Simplified training loop for experimentation
- Support for multiple reward groups (e.g., format, content, reasoning)
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Try to import from the package, fall back to relative imports
try:
    from open_instruct.search_rewards.toy_rewards import compute_combined_rubric_citation_reward
    from open_instruct.search_rewards.finegrained_rewards import compute_finegrained_reward
except ImportError:
    # Add current directory to path for standalone execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    try:
        from search_rewards.toy_rewards import compute_combined_rubric_citation_reward
        from search_rewards.finegrained_rewards import compute_finegrained_reward
    except ImportError:
        print("Warning: Could not import reward functions. Using dummy rewards.")
        def compute_combined_rubric_citation_reward(prediction, label, query):
            return {"finegrained_scores": [(0.5, (0, len(prediction)), 0, 0)], "log_values": {}}
        def compute_finegrained_reward(prediction, label, query):
            return {"finegrained_scores": [(0.5, (0, len(prediction)), 0, 0)], "log_values": {}}


@dataclass
class ToyTrainingConfig:
    """Configuration for toy training script"""
    
    # Model settings
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"  # Use Qwen model
    """The model to use for training"""
    
    # Training settings
    learning_rate: float = 1e-5
    """Learning rate for training"""
    
    num_training_steps: int = 10
    """Number of training steps to run"""
    
    beta: float = 0.01
    """KL penalty coefficient"""
    
    # Reward settings
    reward_function: str = "combined"
    """Which reward function to use: 'combined', 'finegrained', or 'custom'"""
    
    # Fine-grained reward settings
    advantage_normalization: str = "standard"
    """How to normalize advantages: 'standard', 'centered', or 'none'"""
    
    # Logging
    verbose: bool = True
    """Whether to print detailed logs"""
    
    device: str = "auto"
    """Device to use for training"""


class ToyTrainingData:
    """Container for pre-defined training data"""
    
    def __init__(self):
        # Pre-defined prompts and responses for training
        self.training_examples = [
            {
                "prompt": "What is the capital of France?",
                "responses": [
                    "The capital of France is Paris. It's a beautiful city known for the Eiffel Tower.",
                    "Paris is the capital city of France, located in the northern part of the country.",
                    "France's capital is Paris, which is also its largest city and cultural center.",
                ],
                "ground_truth": "Paris",
                "query": "What is the capital of France?"
            },
            {
                "prompt": "Solve: 2 + 3 * 4",
                "responses": [
                    "Following order of operations: 2 + 3 * 4 = 2 + 12 = 14",
                    "First multiply: 3 * 4 = 12, then add: 2 + 12 = 14. The answer is 14.",
                    "2 + 3 * 4 = 5 * 4 = 20",  # Incorrect response
                ],
                "ground_truth": "14",
                "query": "Solve: 2 + 3 * 4"
            },
            {
                "prompt": "Explain photosynthesis briefly.",
                "responses": [
                    "Photosynthesis is the process where plants convert sunlight into energy using chlorophyll.",
                    "Plants use photosynthesis to make food from sunlight, water, and carbon dioxide.",
                    "Photosynthesis: 6CO2 + 6H2O + light → C6H12O6 + 6O2. Plants make glucose and oxygen.",
                ],
                "ground_truth": "Photosynthesis is the process by which plants convert light energy into chemical energy",
                "query": "Explain photosynthesis briefly."
            }
        ]
    
    def get_custom_finegrained_scores(self, example_idx: int, response_idx: int, response_text: str) -> List[Tuple[float, Tuple[int, int], int, int]]:
        """
        Generate custom fine-grained scores for specific examples.
        Returns list of (score, (start_char, end_char), reward_group_id, response_idx) tuples.
        
        Reward groups:
        - 0: Format/Structure
        - 1: Content Accuracy  
        - 2: Reasoning/Explanation
        - 3: Completeness
        """
        
        if example_idx == 0:  # France capital question
            if response_idx == 0:  # "The capital of France is Paris. It's a beautiful city..."
                return [
                    (0.9, (0, 30), 0, response_idx),   # Format: "The capital of France is Paris"
                    (0.95, (4, 30), 1, response_idx),  # Accuracy: "capital of France is Paris"
                    (0.7, (31, len(response_text)), 2, response_idx),  # Additional info
                ]
            elif response_idx == 1:  # "Paris is the capital city..."
                return [
                    (0.85, (0, 35), 0, response_idx),  # Format: "Paris is the capital city of France"
                    (0.95, (0, 35), 1, response_idx),  # Accuracy: same span
                    (0.8, (36, len(response_text)), 2, response_idx),  # Location info
                ]
            elif response_idx == 2:  # "France's capital is Paris..."
                return [
                    (0.8, (0, 25), 0, response_idx),   # Format: "France's capital is Paris"
                    (0.95, (0, 25), 1, response_idx),  # Accuracy: same span
                    (0.85, (26, len(response_text)), 2, response_idx),  # Additional context
                ]
                
        elif example_idx == 1:  # Math problem
            if response_idx == 0:  # Correct with explanation
                return [
                    (0.9, (0, 25), 0, response_idx),   # Format: "Following order of operations"
                    (0.95, (26, 45), 1, response_idx), # Accuracy: "2 + 3 * 4 = 2 + 12"
                    (0.9, (46, len(response_text)), 2, response_idx),  # Final answer
                ]
            elif response_idx == 1:  # Correct with step-by-step
                return [
                    (0.85, (0, 20), 0, response_idx),  # Format: "First multiply:"
                    (0.95, (21, 35), 1, response_idx), # Accuracy: "3 * 4 = 12"
                    (0.9, (36, 55), 2, response_idx),  # Reasoning: "then add: 2 + 12"
                    (0.95, (56, len(response_text)), 1, response_idx),  # Final answer
                ]
            elif response_idx == 2:  # Incorrect response
                return [
                    (0.3, (0, 15), 0, response_idx),   # Format: "2 + 3 * 4 = 5"
                    (0.1, (0, 15), 1, response_idx),   # Accuracy: incorrect calculation
                    (0.2, (16, len(response_text)), 2, response_idx),  # Wrong reasoning
                ]
                
        elif example_idx == 2:  # Photosynthesis
            if response_idx == 0:  # Simple explanation
                return [
                    (0.8, (0, len(response_text)), 0, response_idx),   # Format: complete sentence
                    (0.85, (0, len(response_text)), 1, response_idx),  # Accuracy: correct concept
                    (0.6, (0, len(response_text)), 3, response_idx),   # Completeness: basic
                ]
            elif response_idx == 1:  # Practical explanation
                return [
                    (0.85, (0, len(response_text)), 0, response_idx),  # Format: clear structure
                    (0.9, (0, len(response_text)), 1, response_idx),   # Accuracy: correct
                    (0.8, (0, len(response_text)), 3, response_idx),   # Completeness: good
                ]
            elif response_idx == 2:  # Chemical equation
                return [
                    (0.9, (0, 15), 0, response_idx),   # Format: "Photosynthesis:"
                    (0.95, (16, 50), 1, response_idx), # Accuracy: chemical equation
                    (0.85, (51, len(response_text)), 2, response_idx),  # Explanation
                    (0.9, (0, len(response_text)), 3, response_idx),   # Completeness: comprehensive
                ]
        
        # Default fallback
        return [(0.5, (0, len(response_text)), 0, response_idx)]


class ToyTrainer:
    """Main trainer class for toy fine-grained training"""
    
    def __init__(self, config: ToyTrainingConfig):
        self.config = config
        self.data = ToyTrainingData()
        
        # Set up device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Load model and tokenizer with fallback
        print(f"Loading model: {config.model_name_or_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                torch_dtype=torch.float32,  # Use float32 for stability in toy example
            ).to(self.device)
            
        except Exception as e:
            print(f"Failed to load {config.model_name_or_path}: {e}")
            print("Falling back to GPT2...")
            
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=torch.float32,
            ).to(self.device)
        
        # Set up optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        print(f"Model loaded on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text and return input_ids and attention_mask"""
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def compute_rewards(self, example: Dict, response_idx: int, response_text: str) -> Tuple[List, Dict]:
        """Compute fine-grained rewards for a response"""
        
        if self.config.reward_function == "custom":
            # Use pre-defined custom rewards
            example_idx = self.data.training_examples.index(example)
            finegrained_scores = self.data.get_custom_finegrained_scores(
                example_idx, response_idx, response_text
            )
            log_values = {"reward_type": "custom"}
            
        elif self.config.reward_function == "combined":
            # Use the combined rubric + citation reward
            result = compute_combined_rubric_citation_reward(
                response_text, 
                json.dumps({"Answer": example["ground_truth"]}),
                example["query"]
            )
            finegrained_scores = result.get("finegrained_scores", [])
            log_values = result.get("log_values", {})
            
        elif self.config.reward_function == "finegrained":
            # Use the simple finegrained reward
            result = compute_finegrained_reward(
                response_text,
                example["ground_truth"], 
                example["query"]
            )
            finegrained_scores = result.get("finegrained_scores", [])
            log_values = result.get("log_values", {})
            
        else:
            raise ValueError(f"Unknown reward function: {self.config.reward_function}")
        
        return finegrained_scores, log_values
    
    def convert_char_spans_to_token_spans(self, text: str, char_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Convert character spans to token spans"""
        # Tokenize the text
        tokens = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offset_mapping = tokens["offset_mapping"]
        
        token_spans = []
        for start_char, end_char in char_spans:
            start_token = None
            end_token = None
            
            # Find start token
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start <= start_char < token_end:
                    start_token = i
                    break
                elif start_char <= token_start:
                    start_token = i
                    break
            
            # Find end token
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start < end_char <= token_end:
                    end_token = i + 1
                    break
                elif end_char <= token_start:
                    end_token = i
                    break
            
            # Handle edge cases
            if start_token is None:
                start_token = 0
            if end_token is None:
                end_token = len(offset_mapping)
            
            # Ensure valid span
            start_token = max(0, min(start_token, len(offset_mapping)))
            end_token = max(start_token, min(end_token, len(offset_mapping)))
            
            token_spans.append((start_token, end_token))
        
        return token_spans
    
    def compute_advantages(self, finegrained_scores: List[Tuple]) -> np.ndarray:
        """Compute normalized advantages from fine-grained scores"""
        if not finegrained_scores:
            return np.array([])
        
        # Extract scores and group IDs
        scores = np.array([score for score, _, group_id, _ in finegrained_scores])
        group_ids = [group_id for _, _, group_id, _ in finegrained_scores]
        
        # Normalize advantages per reward group
        advantages = np.zeros_like(scores, dtype=np.float32)
        unique_groups = list(set(group_ids))
        
        for group_id in unique_groups:
            group_indices = [i for i, gid in enumerate(group_ids) if gid == group_id]
            group_scores = scores[group_indices]
            
            # Calculate group statistics
            group_mean = np.mean(group_scores)
            group_std = np.std(group_scores) + 1e-8
            
            # Apply normalization
            if self.config.advantage_normalization == "standard":
                if len(group_scores) == 1:
                    # For single scores, use the score relative to 0.5 (neutral)
                    group_advantages = group_scores - 0.5
                else:
                    group_advantages = (group_scores - group_mean) / group_std
            elif self.config.advantage_normalization == "centered":
                if len(group_scores) == 1:
                    # For single scores, use the score relative to 0.5 (neutral)
                    group_advantages = group_scores - 0.5
                else:
                    group_advantages = group_scores - group_mean
            elif self.config.advantage_normalization == "none":
                group_advantages = group_scores
            else:
                raise ValueError(f"Invalid normalization: {self.config.advantage_normalization}")
            
            # Assign back to main array
            for idx, group_idx in enumerate(group_indices):
                advantages[group_idx] = group_advantages[idx]
        
        return advantages
    
    def training_step(self, step: int):
        """Perform one training step"""
        self.model.train()
        total_loss = 0.0
        step_metrics = {}
        
        # Process each training example
        for example_idx, example in enumerate(self.data.training_examples):
            prompt = example["prompt"]
            responses = example["responses"]
            
            if self.config.verbose:
                print(f"\nStep {step}, Example {example_idx}: {prompt}")
            
            # Process each response
            example_losses = []
            example_advantages = []
            
            for response_idx, response in enumerate(responses):
                # Create full text (prompt + response)
                full_text = prompt + " " + response
                
                # Tokenize
                inputs = self.tokenize_text(full_text)
                input_ids = inputs["input_ids"].squeeze(0)  # Remove batch dimension
                
                # Get model logits
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(0)  # Remove batch dimension
                
                # Compute rewards
                finegrained_scores, reward_logs = self.compute_rewards(example, response_idx, response)
                
                if not finegrained_scores:
                    continue
                
                # Convert character spans to token spans
                char_spans = [(start_char, end_char) for _, (start_char, end_char), _, _ in finegrained_scores]
                token_spans = self.convert_char_spans_to_token_spans(response, char_spans)
                
                # Compute advantages
                advantages = self.compute_advantages(finegrained_scores)
                
                if self.config.verbose:
                    print(f"  Response {response_idx}: {len(finegrained_scores)} reward spans")
                    for i, ((score, (start_char, end_char), group_id, _), (start_tok, end_tok)) in enumerate(zip(finegrained_scores, token_spans)):
                        span_text = response[start_char:end_char]
                        print(f"    Span {i}: score={score:.3f}, adv={advantages[i]:.3f}, group={group_id}, chars=[{start_char}:{end_char}], tokens=[{start_tok}:{end_tok}]")
                        print(f"      Text: '{span_text}'")
                
                # Compute loss for this response
                response_loss = 0.0
                num_spans = 0
                
                # Find where the response starts in the full tokenized text
                prompt_tokens = self.tokenize_text(prompt)["input_ids"].squeeze(0)
                response_start_token = len(prompt_tokens)
                
                for i, ((score, _, group_id, _), (start_tok, end_tok)) in enumerate(zip(finegrained_scores, token_spans)):
                    advantage = advantages[i]
                    
                    # Adjust token positions to account for prompt
                    abs_start_tok = response_start_token + start_tok
                    abs_end_tok = response_start_token + end_tok
                    
                    # Ensure we don't go beyond the sequence
                    abs_start_tok = max(0, min(abs_start_tok, len(input_ids) - 1))
                    abs_end_tok = max(abs_start_tok + 1, min(abs_end_tok, len(input_ids)))
                    
                    # Compute policy gradient loss for this span
                    for tok_idx in range(abs_start_tok, abs_end_tok):
                        if tok_idx >= len(logits) or tok_idx >= len(input_ids) - 1:
                            continue
                        
                        # Get log probability of the actual token
                        token_logits = logits[tok_idx]
                        actual_token = input_ids[tok_idx + 1]  # Next token prediction
                        log_prob = F.log_softmax(token_logits, dim=-1)[actual_token]
                        
                        # Policy gradient: -advantage * log_prob
                        token_loss = -advantage * log_prob
                        response_loss += token_loss
                        num_spans += 1
                
                if num_spans > 0:
                    response_loss = response_loss / num_spans
                    example_losses.append(response_loss)
                    example_advantages.extend(advantages.tolist())
            
            # Average loss across responses for this example
            if example_losses:
                example_loss = torch.stack(example_losses).mean()
                total_loss += example_loss
                
                if self.config.verbose:
                    print(f"  Example loss: {example_loss.item():.4f}")
        
        # Backpropagation
        if total_loss > 0:
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            step_metrics["loss"] = total_loss.item()
            step_metrics["avg_advantage"] = np.mean(example_advantages) if example_advantages else 0.0
            
            if self.config.verbose:
                print(f"Step {step} - Loss: {total_loss.item():.4f}, Avg Advantage: {step_metrics['avg_advantage']:.4f}")
        
        return step_metrics
    
    def train(self):
        """Run the full training loop"""
        print(f"Starting toy training for {self.config.num_training_steps} steps")
        print(f"Reward function: {self.config.reward_function}")
        print(f"Advantage normalization: {self.config.advantage_normalization}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("=" * 60)
        
        all_metrics = []
        
        for step in range(self.config.num_training_steps):
            metrics = self.training_step(step)
            all_metrics.append(metrics)
            
            if step % 5 == 0 or step == self.config.num_training_steps - 1:
                print(f"\nStep {step} Summary:")
                if metrics:
                    for key, value in metrics.items():
                        print(f"  {key}: {value:.4f}")
                else:
                    print("  No metrics (no valid losses computed)")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        
        # Print summary statistics
        valid_losses = [m["loss"] for m in all_metrics if "loss" in m]
        if valid_losses:
            print(f"Average loss: {np.mean(valid_losses):.4f}")
            print(f"Final loss: {valid_losses[-1]:.4f}")
        
        return all_metrics


def main():
    """Main function to run toy training"""
    
    # Configuration
    config = ToyTrainingConfig(
        model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",  # Use Qwen model
        learning_rate=1e-5,
        num_training_steps=10,
        beta=0.01,
        reward_function="custom",  # Try "custom", "combined", or "finegrained"
        advantage_normalization="standard",  # Try "standard", "centered", or "none"
        verbose=True,
    )
    
    # Create trainer and run training
    trainer = ToyTrainer(config)
    metrics = trainer.train()
    
    # Demonstrate different reward functions
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Different Reward Functions")
    print("=" * 60)
    
    example = trainer.data.training_examples[0]
    response = example["responses"][0]
    
    for reward_func in ["custom", "combined", "finegrained"]:
        print(f"\nReward function: {reward_func}")
        config.reward_function = reward_func
        trainer.config = config
        
        finegrained_scores, log_values = trainer.compute_rewards(example, 0, response)
        advantages = trainer.compute_advantages(finegrained_scores)
        
        print(f"Generated {len(finegrained_scores)} reward spans:")
        for i, (score, (start_char, end_char), group_id, response_idx) in enumerate(finegrained_scores):
            span_text = response[start_char:end_char]
            adv = advantages[i] if i < len(advantages) else 0.0
            print(f"  {i+1}. Score: {score:.3f}, Advantage: {adv:.3f}, Group: {group_id}")
            print(f"      Span: '{span_text}'")


if __name__ == "__main__":
    main() 