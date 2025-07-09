import unittest
import torch
import numpy as np
import transformers
import vllm
import parameterized
from typing import Dict, List, Union, Any
from transformers import PreTrainedModel
from open_instruct import model_utils
from open_instruct import rl_utils2


MAX_TOKENS = 20
SEED = 42
PACK_LENGTH = 64


class TestLogprobsComparison(unittest.TestCase):
    """Test logprobs calculation and comparison between HuggingFace and vLLM."""
    
    @unittest.skipIf(not torch.cuda.is_available(), "No GPU available")
    @parameterized.parameterized.expand([
        ("hamishivi/qwen3_openthoughts2", "The capital of France is"),
        ("hamishivi/qwen3_openthoughts2", "The weather today is"),
        ("hamishivi/qwen3_openthoughts2", "Machine learning is"),
    ])
    def test_vllm_hf_logprobs_match_large(self, model_name, prompt):
        """Test that vLLM and HuggingFace produce matching logprobs for large models (GPU only)."""

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        query = tokenizer(prompt)['input_ids']
        
        # Get vLLM logprobs
        vllm_output = _get_vllm_logprobs(model_name, query)
        packed_sequences = rl_utils2.pack_sequences(
            queries=[query],
            responses=[vllm_output["response"]],
            # This mask is the tool use mask, which we mock to be all ones, as done in grpo_fast.py
            # when tooluse is disabled.
            masks=[[1] * len(vllm_output["response"])],
            pack_length=PACK_LENGTH,
            pad_token_id=tokenizer.pad_token_id,
        )

        hf_logprobs = _get_hf_logprobs(model_name, query,
                                       vllm_output["response"],
                                       packed_sequences.query_responses[0],
                                       packed_sequences.attention_masks[0],
                                       packed_sequences.position_ids[0],
                                       tokenizer.pad_token_id)
        vllm_logprobs = vllm_output["logprobs"]
        
        # Check that the tokens being scored match
        packed_response_tokens = packed_sequences.query_responses[0][len(query):len(query)+len(vllm_output['response'])].tolist()
        
        self.assertEqual(len(vllm_logprobs), len(vllm_output["response"]))
        self.assertEqual(len(vllm_logprobs), len(hf_logprobs), f'{vllm_logprobs=}\n{hf_logprobs=}')
        
        # Verify tokens match before comparing logprobs
        self.assertEqual(vllm_output['response'], packed_response_tokens, "Response tokens don't match between vLLM and packed sequences")
        
        np.testing.assert_array_almost_equal(vllm_logprobs, hf_logprobs)
        
         
def _get_hf_logprobs(model_name: str, query: List[int],
                     response: List[int],
                     query_response, attention_mask, position_ids, pad_token_id) -> List[float]:
    """Get logprobs using HuggingFace transformers."""
    padding_mask = query_response != pad_token_id
    input_ids = torch.masked_fill(query_response, ~padding_mask, 0)
    
    model: PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    
    with torch.no_grad():
        input_ids = input_ids[None, :].to('cuda')
        attention_mask = attention_mask[None, :].to('cuda')
        position_ids = position_ids[None, :].to('cuda')
        output = model(
            input_ids=input_ids[:, :-1],
            # @vwxyzjn: without clamp, we get index out of bounds errors; TODO: investigate
            attention_mask=attention_mask[:, :-1].clamp(0, 1),
            position_ids=position_ids[:, :-1],
            return_dict=True,
        )
        logits = output.logits
        logprobs = model_utils.log_softmax_and_gather(logits, input_ids[:, 1:])
        logprobs = logprobs[:, len(query) - 1:]
    return logprobs.flatten().tolist()


def _get_vllm_logprobs(model_name: str, prompt: str) -> Dict[str, Union[List[str], List[int], List[float]]]:
    """Get logprobs using vLLM."""
    # Determine dtype based on model
    llm = vllm.LLM(
        model=model_name,
        seed=SEED,
        enforce_eager=True,  # Disable CUDA graph for consistency
        dtype="bfloat16",
    )
    
    sampling_params = vllm.SamplingParams(
        max_tokens=MAX_TOKENS,
        logprobs=0,  # Return top-1 logprob
        seed=SEED
    )
    
    # Generate
    outputs = llm.generate(prompt_token_ids=[prompt], sampling_params=sampling_params)
    output = outputs[0]
    
    # Extract logprobs
    response = []
    logprobs = []
    
    for token_info in output.outputs[0].logprobs:
        # Get the token and its logprob
        token_id = list(token_info.keys())[0]
        logprob_info = token_info[token_id]
        
        response.append(token_id)
        logprobs.append(logprob_info.logprob)
        
    return {
        "response": response,
        "logprobs": logprobs
    }


if __name__ == "__main__":
    unittest.main()
