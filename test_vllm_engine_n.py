#!/usr/bin/env python3
"""Test script to debug vLLM LLMEngine behavior with n>1."""

import os
import sys
import torch
import vllm
from vllm import SamplingParams, LLMEngine, EngineArgs
from transformers import AutoTokenizer

def test_vllm_engine_n_parameter():
    """Test vLLM LLMEngine with n>1 to see how many completions it generates."""
    
    # Model configuration
    model_name = "EleutherAI/pythia-14m"  # Small model for testing
    
    print(f"Testing vLLM LLMEngine with model: {model_name}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create test prompts
    prompts = [
        "The capital of France is",
        "Machine learning is",
        "Python programming language was created by",
    ]
    
    # Tokenize prompts
    print(f"\nTokenizing {len(prompts)} prompts...")
    tokenized_prompts = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, return_tensors="pt").squeeze().tolist()
        tokenized_prompts.append(tokens)
        print(f"  Prompt: '{prompt[:30]}...' -> {len(tokens)} tokens")
    
    # Test different configurations
    test_configs = [
        {"n": 1, "best_of": 1, "temperature": 0.8},
        {"n": 4, "best_of": 4, "temperature": 0.8},
        {"n": 4, "best_of": 8, "temperature": 0.8},  # best_of > n
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing with n={config['n']}, best_of={config['best_of']}")
        print(f"{'='*60}")
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=config["temperature"],
            top_p=0.95,
            max_tokens=50,
            n=config["n"],
            best_of=config["best_of"],
        )
        
        print(f"\nSamplingParams: {sampling_params}")
        
        try:
            # Initialize LLMEngine
            print("\nInitializing LLMEngine...")
            engine_args = EngineArgs(
                model=model_name,
                gpu_memory_utilization=0.5,  # Use less GPU memory for testing
                enforce_eager=True,  # Disable CUDA graphs for testing
            )
            engine = LLMEngine.from_engine_args(engine_args)
            
            # Add requests
            print(f"\nAdding {len(tokenized_prompts)} requests to engine...")
            for i, prompt_tokens in enumerate(tokenized_prompts):
                request_id = f"test_request_{i}"
                tokens_prompt = vllm.TokensPrompt(prompt_token_ids=prompt_tokens)
                engine.add_request(request_id, tokens_prompt, sampling_params)
                print(f"  Added request {request_id} with {len(prompt_tokens)} tokens")
            
            # Run engine until all requests are finished
            print("\nRunning engine...")
            outputs = []
            step_count = 0
            while engine.has_unfinished_requests():
                step_outputs = engine.step()
                step_count += 1
                for output in step_outputs:
                    if output.finished:
                        outputs.append(output)
                        print(f"  Step {step_count}: Request {output.request_id} finished")
            
            # Sort outputs by request ID
            outputs.sort(key=lambda x: int(x.request_id.split("_")[-1]))
            
            # Analyze results
            print(f"\nResults:")
            print(f"  Total RequestOutputs: {len(outputs)}")
            print(f"  Expected RequestOutputs: {len(tokenized_prompts)}")
            
            total_completions = 0
            for i, output in enumerate(outputs):
                num_completions = len(output.outputs)
                total_completions += num_completions
                print(f"\n  RequestOutput {i} (request_id={output.request_id}):")
                print(f"    Number of completions: {num_completions}")
                print(f"    Expected completions: {config['n']}")
                
                for j, completion in enumerate(output.outputs[:2]):  # Show first 2
                    text = tokenizer.decode(completion.token_ids, skip_special_tokens=True)
                    print(f"    Completion {j}: '{text[:50]}...'")
                
                if num_completions > 2:
                    print(f"    ... and {num_completions - 2} more completions")
            
            print(f"\n  Total completions across all requests: {total_completions}")
            print(f"  Expected total: {len(tokenized_prompts) * config['n']}")
            
            # Check if we got the expected number
            if total_completions == len(tokenized_prompts) * config['n']:
                print(f"\n✅ SUCCESS: Got expected number of completions!")
            else:
                print(f"\n❌ MISMATCH: Expected {len(tokenized_prompts) * config['n']}, got {total_completions}")
            
            # Cleanup
            del engine
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n❌ ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


def test_high_level_api():
    """Test the high-level vLLM API for comparison."""
    print("\n" + "="*80)
    print("Testing HIGH-LEVEL vLLM API for comparison")
    print("="*80)
    
    model_name = "EleutherAI/pythia-14m"  # Change to match above
    
    try:
        from vllm import LLM
        
        # Initialize high-level LLM
        print(f"\nInitializing LLM with model: {model_name}")
        llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )
        
        prompts = [
            "The capital of France is",
            "Machine learning is",
            "Python programming language was created by",
        ]
        
        # Test with n=4
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50,
            n=4,
        )
        
        print(f"\nGenerating with high-level API (n=4)...")
        outputs = llm.generate(prompts, sampling_params)
        
        print(f"\nHigh-level API Results:")
        print(f"  Number of RequestOutputs: {len(outputs)}")
        
        total_completions = 0
        for i, output in enumerate(outputs):
            num_completions = len(output.outputs)
            total_completions += num_completions
            print(f"\n  Prompt {i}: '{prompts[i][:30]}...'")
            print(f"    Generated {num_completions} completions")
            for j, completion in enumerate(output.outputs[:2]):
                print(f"    Completion {j}: '{completion.text[:50]}...'")
            if num_completions > 2:
                print(f"    ... and {num_completions - 2} more")
        
        print(f"\n  Total completions: {total_completions}")
        print(f"  Expected: {len(prompts) * 4}")
        
        # Cleanup
        del llm
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\n❌ ERROR in high-level API test: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("vLLM LLMEngine n>1 Test Script")
    print("==============================")
    
    # Test LLMEngine behavior
    test_vllm_engine_n_parameter()
    
    # Test high-level API for comparison
    test_high_level_api()
    
    print("\n✅ All tests completed!")