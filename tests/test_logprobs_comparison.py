import unittest
import logging
import torch
import numpy as np
import transformers
import vllm
import parameterized


class TestLogprobsComparison(unittest.TestCase):
    """Test logprobs calculation and comparison between HuggingFace and vLLM."""
    
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.cuda_available = torch.cuda.is_available()
        
    @parameterized.parameterized.expand([
        ("gpt2", "The capital of France is", 0.0),
        ("gpt2", "The weather today is", 0.0),
        ("gpt2", "Machine learning is", 0.0),
    ])
    def test_vllm_hf_logprobs_match_small(self, model_name, prompt, temperature):
        """Test that vLLM and HuggingFace produce matching logprobs for small models."""
        max_tokens = 20
        seed = 42
        
        # Get HuggingFace logprobs
        hf_logprobs = self._get_hf_logprobs(model_name, prompt, max_tokens, temperature, seed)
        
        # Get vLLM logprobs
        vllm_logprobs = self._get_vllm_logprobs(model_name, prompt, max_tokens, temperature, seed)
        
        # Compare the logprobs
        self._compare_logprobs(hf_logprobs, vllm_logprobs, prompt)
    
    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([
        ("meta-llama/Llama-2-7b-hf", "The capital of France is", 0.0),
        ("meta-llama/Llama-2-7b-hf", "The weather today is", 0.0),
        ("meta-llama/Llama-2-7b-hf", "Machine learning is", 0.0),
    ])
    def test_vllm_hf_logprobs_match_large(self, model_name, prompt, temperature):
        """Test that vLLM and HuggingFace produce matching logprobs for large models (GPU only)."""
        max_tokens = 20
        seed = 42
        
        # Get HuggingFace logprobs
        hf_logprobs = self._get_hf_logprobs(model_name, prompt, max_tokens, temperature, seed)
        
        # Get vLLM logprobs
        vllm_logprobs = self._get_vllm_logprobs(model_name, prompt, max_tokens, temperature, seed)
        
        # Compare the logprobs
        self._compare_logprobs(hf_logprobs, vllm_logprobs, prompt)
        
    def _get_hf_logprobs(self, model_name, prompt, max_tokens, temperature, seed):
        """Get logprobs using HuggingFace transformers."""
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Determine dtype based on model and hardware
        if "llama" in model_name.lower() and self.cuda_available:
            dtype = torch.float16
        else:
            dtype = torch.float32
            
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.cuda_available else None
        )
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        if self.cuda_available:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        
        # Generate with logprobs
        torch.manual_seed(seed)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # Extract logprobs
        generated_ids = outputs.sequences[0, input_ids.shape[1]:]
        scores = outputs.scores  # tuple of tensors
        
        # Get the log probs for the generated tokens
        token_logprobs = []
        for i, token_id in enumerate(generated_ids):
            log_probs = torch.nn.functional.log_softmax(scores[i], dim=-1)
            token_logprobs.append(log_probs[0, token_id.item()].item())
            
        # Get tokens for debugging
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        
        return {
            "tokens": generated_tokens,
            "token_ids": generated_ids.tolist(),
            "logprobs": token_logprobs
        }
        
    def _get_vllm_logprobs(self, model_name, prompt, max_tokens, temperature, seed):
        """Get logprobs using vLLM."""
        # Determine dtype based on model
        if "llama" in model_name.lower():
            dtype = "float16"
        else:
            dtype = "float32"
            
        llm = vllm.LLM(
            model=model_name,
            seed=seed,
            enforce_eager=True,  # Disable CUDA graph for consistency
            dtype=dtype
        )
        
        sampling_params = vllm.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=1,  # Return top-1 logprob
            seed=seed
        )
        
        # Generate
        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        # Extract logprobs
        tokens = []
        token_ids = []
        logprobs = []
        
        for token_info in output.outputs[0].logprobs:
            # Get the token and its logprob
            token_id = list(token_info.keys())[0]
            logprob_info = token_info[token_id]
            
            tokens.append(logprob_info.decoded_token)
            token_ids.append(token_id)
            logprobs.append(logprob_info.logprob)
            
        return {
            "tokens": tokens,
            "token_ids": token_ids,
            "logprobs": logprobs
        }
        
    def _compare_logprobs(self, hf_result, vllm_result, prompt):
        """Compare logprobs between HuggingFace and vLLM."""
        self.logger.info("\n=== Comparison Results ===")
        self.logger.info(f"Prompt: '{prompt}'")
        self.logger.info(f"\nGenerated text (HF): {''.join(hf_result['tokens'])}")
        self.logger.info(f"Generated text (vLLM): {''.join(vllm_result['tokens'])}")
        
        # Check if the same tokens were generated
        hf_ids = hf_result['token_ids']
        vllm_ids = vllm_result['token_ids']
        
        if hf_ids != vllm_ids:
            self.logger.warning("Different tokens generated!")
            self.logger.warning(f"HF token IDs: {hf_ids}")
            self.logger.warning(f"vLLM token IDs: {vllm_ids}")
            
        # Compare logprobs
        self.logger.info("\nLogprob comparison:")
        self.logger.info(f"{'Token':<15} {'HF logprob':<15} {'vLLM logprob':<15} {'Difference':<15} {'Rel. Error':<15}")
        self.logger.info("-" * 75)
        
        max_abs_diff = 0
        max_rel_error = 0
        
        for i in range(min(len(hf_result['logprobs']), len(vllm_result['logprobs']))):
            hf_logprob = hf_result['logprobs'][i]
            vllm_logprob = vllm_result['logprobs'][i]
            
            abs_diff = abs(hf_logprob - vllm_logprob)
            rel_error = abs_diff / abs(hf_logprob) if hf_logprob != 0 else float('inf')
            
            max_abs_diff = max(max_abs_diff, abs_diff)
            max_rel_error = max(max_rel_error, rel_error)
            
            token = hf_result['tokens'][i] if i < len(hf_result['tokens']) else vllm_result['tokens'][i]
            self.logger.info(f"{token:<15} {hf_logprob:<15.6f} {vllm_logprob:<15.6f} {abs_diff:<15.6e} {rel_error:<15.6e}")
            
        self.logger.info(f"\nMax absolute difference: {max_abs_diff:.6e}")
        self.logger.info(f"Max relative error: {max_rel_error:.6e}")
        
        # Assert that differences are small
        tolerance = 1e-3  # Adjust based on expected precision
        self.assertLess(max_abs_diff, tolerance, 
                       f"Maximum absolute difference {max_abs_diff} exceeds tolerance {tolerance}")


if __name__ == "__main__":
    unittest.main()