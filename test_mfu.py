#!/usr/bin/env python3
"""
Test script for MFU (Model FLOPs Utilization) functionality in grpo_fast.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the MFU functions from grpo_fast
from open_instruct.grpo_fast import calculate_model_flops_per_token, calculate_mfu

def test_mfu_functionality():
    """Test the MFU calculation functionality."""
    print("Testing MFU functionality...")
    
    # Test with a small model
    model_name = "microsoft/DialoGPT-small"  # Small model for testing
    print(f"Loading model: {model_name}")
    
    try:
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set model to evaluation mode
        model.eval()
        
        # Test FLOPs calculation
        print("Calculating FLOPs per token...")
        flops_per_token = calculate_model_flops_per_token(model, tokenizer)
        print(f"FLOPs per token: {flops_per_token}")
        
        # Test MFU calculation
        print("Testing MFU calculation...")
        tokens_per_second = 100.0  # Example tokens per second
        mfu = calculate_mfu(flops_per_token, tokens_per_second, model_name)
        print(f"MFU: {mfu:.2f}%")
        
        print("✅ MFU functionality test passed!")
        
    except Exception as e:
        print(f"❌ MFU functionality test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_mfu_functionality()