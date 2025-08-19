#!/usr/bin/env python3
"""
Offline Demo of Toy Training Script for Fine-Grained Token-Level Reward Control

This demo shows the key concepts without requiring model downloads.
It demonstrates:
- Fine-grained reward computation
- Token-level advantage calculation
- Character span to token span conversion
- Different reward normalization strategies
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

# Mock tokenizer for demonstration
class MockTokenizer:
    def __init__(self):
        # Simple word-based tokenization for demo
        self.vocab = {
            "<pad>": 0, "<eos>": 1, "the": 2, "capital": 3, "of": 4, "france": 5, "is": 6, "paris": 7,
            "a": 8, "beautiful": 9, "city": 10, "known": 11, "for": 12, "eiffel": 13, "tower": 14,
            "solve": 15, "2": 16, "+": 17, "3": 18, "*": 19, "4": 20, "=": 21, "12": 22, "14": 23,
            "following": 24, "order": 25, "operations": 26, "first": 27, "multiply": 28, "then": 29,
            "add": 30, "answer": 31, "photosynthesis": 32, "process": 33, "where": 34, "plants": 35,
            "convert": 36, "sunlight": 37, "into": 38, "energy": 39, "using": 40, "chlorophyll": 41,
            "make": 42, "food": 43, "from": 44, "water": 45, "and": 46, "carbon": 47, "dioxide": 48,
            "6co2": 49, "6h2o": 50, "light": 51, "c6h12o6": 52, "6o2": 53, "glucose": 54, "oxygen": 55,
            "it": 56, "'s": 57, ".": 58, ",": 59, ":": 60, "located": 61, "in": 62, "northern": 63,
            "part": 64, "country": 65, "also": 66, "its": 67, "largest": 68, "cultural": 69, "center": 70,
            "→": 71, "which": 72, "5": 73, "20": 74, "brief": 75, "briefly": 76
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
    
    def encode(self, text, add_special_tokens=False):
        words = text.lower().replace(".", " .").replace(",", " ,").replace(":", " :").split()
        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Use a simple hash for unknown words
                tokens.append(len(self.vocab) + hash(word) % 1000)
        return tokens
    
    def decode(self, tokens, skip_special_tokens=True):
        words = []
        for token in tokens:
            if token in self.reverse_vocab:
                word = self.reverse_vocab[token]
                if skip_special_tokens and word in ["<pad>", "<eos>"]:
                    continue
                words.append(word)
            else:
                words.append(f"<unk_{token}>")
        return " ".join(words)
    
    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, add_special_tokens=False, **kwargs):
        tokens = self.encode(text, add_special_tokens)
        result = {"input_ids": tokens}
        
        if return_offsets_mapping:
            # Create simple offset mapping
            words = text.lower().replace(".", " .").replace(",", " ,").replace(":", " :").split()
            offsets = []
            pos = 0
            for word in words:
                start = text.lower().find(word, pos)
                if start == -1:
                    start = pos
                end = start + len(word)
                offsets.append((start, end))
                pos = end
            result["offset_mapping"] = offsets
        
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor([result["input_ids"]])
            result["attention_mask"] = torch.ones_like(result["input_ids"])
        
        return result

# Mock model for demonstration
class MockModel:
    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size
        # Create random parameters for demo
        torch.manual_seed(42)  # For reproducible demo
        self.parameters_list = [torch.randn(100, 50, requires_grad=True)]
    
    def parameters(self):
        return self.parameters_list
    
    def train(self):
        pass
    
    def __call__(self, input_ids, attention_mask=None, **kwargs):
        # Generate random logits for demo
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size)  # Remove requires_grad=True
        
        # Make logits somewhat realistic by adding bias towards actual tokens
        for i in range(batch_size):
            for j in range(seq_len):
                if j < seq_len - 1:
                    next_token = input_ids[i, j + 1].item()
                    if next_token < self.vocab_size:
                        logits[i, j, next_token] += 2.0  # Bias towards actual next token
        
        # Now make it require gradients
        logits = logits.requires_grad_(True)
        
        class MockOutput:
            def __init__(self, logits):
                self.logits = logits
        
        return MockOutput(logits)

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
            }
        ]
    
    def get_custom_finegrained_scores(self, example_idx: int, response_idx: int, response_text: str) -> List[Tuple[float, Tuple[int, int], int, int]]:
        """Generate custom fine-grained scores for specific examples."""
        
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
        
        # Default fallback
        return [(0.5, (0, len(response_text)), 0, response_idx)]

def convert_char_spans_to_token_spans(tokenizer, text: str, char_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Convert character spans to token spans"""
    tokens = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
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

def compute_advantages(finegrained_scores: List[Tuple], normalization: str = "standard") -> np.ndarray:
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
        if normalization == "standard":
            group_advantages = (group_scores - group_mean) / group_std
        elif normalization == "centered":
            group_advantages = group_scores - group_mean
        elif normalization == "none":
            group_advantages = group_scores
        else:
            raise ValueError(f"Invalid normalization: {normalization}")
        
        # Assign back to main array
        for idx, group_idx in enumerate(group_indices):
            advantages[group_idx] = group_advantages[idx]
    
    return advantages

def demo_training_step(tokenizer, model, data, example_idx: int, normalization: str = "standard"):
    """Demonstrate one training step"""
    example = data.training_examples[example_idx]
    prompt = example["prompt"]
    responses = example["responses"]
    
    print(f"\n{'='*60}")
    print(f"EXAMPLE {example_idx}: {prompt}")
    print(f"{'='*60}")
    
    total_loss = 0.0
    all_advantages = []
    
    for response_idx, response in enumerate(responses):
        print(f"\nResponse {response_idx}: {response}")
        print("-" * 40)
        
        # Create full text (prompt + response)
        full_text = prompt + " " + response
        
        # Tokenize
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze(0)
        
        # Get model logits
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        logits = outputs.logits.squeeze(0)
        
        # Compute rewards
        finegrained_scores = data.get_custom_finegrained_scores(example_idx, response_idx, response)
        
        # Convert character spans to token spans
        char_spans = [(start_char, end_char) for _, (start_char, end_char), _, _ in finegrained_scores]
        token_spans = convert_char_spans_to_token_spans(tokenizer, response, char_spans)
        
        # Compute advantages
        advantages = compute_advantages(finegrained_scores, normalization)
        all_advantages.extend(advantages.tolist())
        
        print(f"Reward spans ({len(finegrained_scores)}):")
        for i, ((score, (start_char, end_char), group_id, _), (start_tok, end_tok)) in enumerate(zip(finegrained_scores, token_spans)):
            span_text = response[start_char:end_char]
            adv = advantages[i] if i < len(advantages) else 0.0
            group_names = {0: "Format", 1: "Accuracy", 2: "Reasoning", 3: "Completeness"}
            group_name = group_names.get(group_id, f"Group{group_id}")
            print(f"  {i+1}. {group_name}: score={score:.3f}, adv={adv:.3f}")
            print(f"      Chars [{start_char}:{end_char}]: '{span_text}'")
            print(f"      Tokens [{start_tok}:{end_tok}]")
        
        # Simulate loss computation
        response_loss = 0.0
        num_tokens = 0
        
        # Find where the response starts in the full tokenized text
        prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0)
        response_start_token = len(prompt_tokens)
        
        for i, ((score, _, group_id, _), (start_tok, end_tok)) in enumerate(zip(finegrained_scores, token_spans)):
            advantage = advantages[i]
            
            # Adjust token positions to account for prompt
            abs_start_tok = response_start_token + start_tok
            abs_end_tok = response_start_token + end_tok
            
            # Ensure we don't go beyond the sequence
            abs_start_tok = max(0, min(abs_start_tok, len(input_ids) - 1))
            abs_end_tok = max(abs_start_tok + 1, min(abs_end_tok, len(input_ids)))
            
            # Simulate policy gradient loss for this span
            for tok_idx in range(abs_start_tok, abs_end_tok):
                if tok_idx >= len(logits) or tok_idx >= len(input_ids) - 1:
                    continue
                
                # Get log probability of the actual token
                token_logits = logits[tok_idx]
                actual_token = input_ids[tok_idx + 1]  # Next token prediction
                log_prob = F.log_softmax(token_logits, dim=-1)[actual_token]
                
                # Policy gradient: -advantage * log_prob
                token_loss = -advantage * log_prob
                response_loss += token_loss.item()
                num_tokens += 1
        
        if num_tokens > 0:
            response_loss = response_loss / num_tokens
            total_loss += response_loss
            print(f"Response loss: {response_loss:.4f}")
    
    print(f"\nTotal loss: {total_loss:.4f}")
    print(f"Average advantage: {np.mean(all_advantages):.4f}")
    print(f"Advantage std: {np.std(all_advantages):.4f}")
    
    return total_loss

def main():
    """Main demo function"""
    print("🎯 TOY TRAINING DEMO: Fine-Grained Token-Level Reward Control")
    print("=" * 70)
    
    # Check GPU availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    # Initialize components
    print(f"\n🔧 Initializing mock components...")
    tokenizer = MockTokenizer()
    model = MockModel()
    data = ToyTrainingData()
    
    print(f"✓ Mock tokenizer with {len(tokenizer.vocab)} vocab items")
    print(f"✓ Mock model with {len(list(model.parameters()))} parameter tensors")
    print(f"✓ Training data with {len(data.training_examples)} examples")
    
    # Test tokenization
    print(f"\n🔤 Testing tokenization...")
    test_text = "The capital of France is Paris."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"Original: '{test_text}'")
    print(f"Tokens: {tokens}")
    print(f"Decoded: '{decoded}'")
    
    # Demonstrate different normalization strategies
    normalization_strategies = ["standard", "centered", "none"]
    
    for norm_strategy in normalization_strategies:
        print(f"\n🎯 DEMONSTRATION: {norm_strategy.upper()} Normalization")
        print("=" * 50)
        
        # Run demo training steps
        for example_idx in range(len(data.training_examples)):
            demo_training_step(tokenizer, model, data, example_idx, norm_strategy)
    
    # Show advantage computation details
    print(f"\n📊 ADVANTAGE COMPUTATION DETAILS")
    print("=" * 40)
    
    example = data.training_examples[0]
    response = example["responses"][0]
    finegrained_scores = data.get_custom_finegrained_scores(0, 0, response)
    
    print(f"Example response: '{response}'")
    print(f"Fine-grained scores: {len(finegrained_scores)} spans")
    
    for norm_strategy in normalization_strategies:
        advantages = compute_advantages(finegrained_scores, norm_strategy)
        print(f"\n{norm_strategy.capitalize()} normalization:")
        
        # Group by reward type
        groups = {}
        for i, (score, _, group_id, _) in enumerate(finegrained_scores):
            if group_id not in groups:
                groups[group_id] = {"scores": [], "advantages": []}
            groups[group_id]["scores"].append(score)
            groups[group_id]["advantages"].append(advantages[i])
        
        for group_id, group_data in groups.items():
            group_names = {0: "Format", 1: "Accuracy", 2: "Reasoning"}
            group_name = group_names.get(group_id, f"Group{group_id}")
            scores = group_data["scores"]
            advs = group_data["advantages"]
            print(f"  {group_name}: scores={scores}, advantages={[f'{a:.3f}' for a in advs]}")
    
    print(f"\n🎉 Demo completed! Key concepts demonstrated:")
    print("  ✓ Fine-grained reward assignment to text spans")
    print("  ✓ Character span to token span conversion")
    print("  ✓ Per-group advantage normalization")
    print("  ✓ Policy gradient loss computation")
    print("  ✓ Different normalization strategies")
    print("\nThis demonstrates the core mechanics of fine-grained RLHF training!")

if __name__ == "__main__":
    main() 