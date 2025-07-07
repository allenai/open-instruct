import unittest
import logging
import torch
import numpy as np
import transformers
import parameterized

try:
    import vllm
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


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
    def test_hf_logprobs_calculation(self, model_name, prompt, temperature):
        """Test HuggingFace logprobs calculation and output format."""
        max_tokens = 10
        seed = 42
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32  # Use float32 for CPU
        )
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        
        self.logger.info(f"\nPrompt: '{prompt}'")
        self.logger.info(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
        
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
        # outputs.scores is a tuple of tensors, each of shape [batch_size, vocab_size]
        scores = outputs.scores  # tuple of tensors
        
        # Get the log probs for the generated tokens
        self.logger.info("\nGenerated tokens with logprobs:")
        self.logger.info(f"{'Token':<20} {'Token ID':<10} {'Logprob':<15} {'Prob':<15}")
        self.logger.info("-" * 60)
        
        for i, token_id in enumerate(generated_ids):
            token = tokenizer.decode([token_id.item()])
            # Get log probs for this step
            log_probs = torch.nn.functional.log_softmax(scores[i], dim=-1)
            logprob = log_probs[0, token_id.item()].item()  # batch_size=1, so index 0
            prob = np.exp(logprob)
            self.logger.info(f"{repr(token):<20} {token_id.item():<10} {logprob:<15.6f} {prob:<15.6f}")
            
        # Also show top-5 alternatives for first generated token
        self.logger.info("\nTop-5 alternatives for first generated token:")
        first_token_logprobs = torch.nn.functional.log_softmax(scores[0], dim=-1)[0]  # Get first token, first batch
        top5_values, top5_indices = torch.topk(first_token_logprobs, 5)
        
        for value, idx in zip(top5_values, top5_indices):
            token = tokenizer.decode([idx.item()])
            self.logger.info(f"  {repr(token):<20} logprob: {value.item():.6f}")
    
    @unittest.skipIf(not VLLM_AVAILABLE, "vLLM not available")
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
    
    @unittest.skipIf(not VLLM_AVAILABLE or not torch.cuda.is_available(), 
                     "vLLM not available or no GPU")
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
        
    def test_hf_logprobs_with_temperature(self):
        """Test how temperature affects logprobs."""
        model_name = "gpt2"
        prompt = "The capital of France is"
        max_tokens = 5
        seed = 42
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        
        for temp in [0.0, 0.5, 1.0]:
            self.logger.info(f"\n=== Temperature: {temp} ===")
            
            torch.manual_seed(seed)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temp if temp > 0 else 1.0,  # Avoid division by zero
                    do_sample=(temp > 0),
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            generated_ids = outputs.sequences[0, input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids)
            self.logger.info(f"Generated: {repr(generated_text)}")
    
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