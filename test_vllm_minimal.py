#!/usr/bin/env python3
"""Minimal test to reproduce vLLM LLMEngine n>1 issue."""

import vllm
from vllm import SamplingParams, LLMEngine, EngineArgs

def main():
    model_name = "EleutherAI/pythia-14m"
    
    # Test prompts (already tokenized for pythia)
    prompt_tokens = [
        [464, 3139, 286, 4881, 318],  # "The capital of France is"
        [37573, 4673, 318],            # "Machine learning is"
    ]
    
    # Create sampling params with n=4
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=20,
        n=4,  # Want 4 completions per prompt
        best_of=4,  # Try with best_of=n
    )
    
    print(f"Testing vLLM LLMEngine with n={sampling_params.n}, best_of={sampling_params.best_of}")
    
    # Initialize engine
    engine_args = EngineArgs(
        model=model_name,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    
    # Add requests
    for i, tokens in enumerate(prompt_tokens):
        request_id = f"req_{i}"
        tokens_prompt = vllm.TokensPrompt(prompt_token_ids=tokens)
        engine.add_request(request_id, tokens_prompt, sampling_params)
        print(f"Added request {request_id} with {len(tokens)} tokens")
    
    # Run engine
    outputs = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
    
    # Check results
    print(f"\nResults:")
    print(f"Number of RequestOutputs: {len(outputs)}")
    
    total_completions = 0
    for output in outputs:
        num_completions = len(output.outputs)
        total_completions += num_completions
        print(f"\nRequest {output.request_id}: {num_completions} completions (expected: {sampling_params.n})")
        
    print(f"\nTotal completions: {total_completions}")
    print(f"Expected: {len(prompt_tokens) * sampling_params.n} ({len(prompt_tokens)} prompts * {sampling_params.n} n)")
    
    if total_completions == len(prompt_tokens) * sampling_params.n:
        print("✅ SUCCESS: Got expected number of completions!")
    else:
        print("❌ FAILURE: Did not get expected number of completions!")
        print("\nThis confirms the issue: LLMEngine with n>1 only generates 1 completion per request")
        print("even when best_of is set correctly.")

if __name__ == "__main__":
    main()