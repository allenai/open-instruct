import queue
import threading
import time
import unittest
import os

import numpy as np
import ray
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from open_instruct.grpo_fast import ShufflingIterator, vllm_generate_thread


# Define a simple Ray actor to wrap vLLM engine
@ray.remote(num_gpus=0)  # Use CPU for testing
class VLLMEngineActor:
    def __init__(self, model_name: str):
        self.engine = LLM(
            model=model_name,
            max_model_len=512,  # Small context for testing
            tensor_parallel_size=1,
            enforce_eager=True,  # Disable CUDA graph
            device="cpu",
            dtype="float32",  # Use float32 for CPU
        )
    
    def generate(self, prompt_token_ids, sampling_params, use_tqdm=False):
        return self.engine.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm
        )


class TestGrpoFast(unittest.TestCase):
    @unittest.skipIf(os.getenv("SKIP_VLLM_TESTS", "false").lower() == "true", 
                     "Skipping vLLM tests as requested")
    def test_queue_based_generator_communication(self):
        """Test vllm_generate_thread with a real vLLM engine using a tiny model.
        
        This test requires model download from HuggingFace. Make sure you have:
        1. Internet connection
        2. HF_TOKEN environment variable set if using gated models
        3. Or have the model cached locally
        
        NOTE: This test requires a GPU or a CPU-compatible model architecture.
        Many small models (pythia, gpt2) are not supported by vLLM on CPU.
        Set SKIP_VLLM_TESTS=true to skip this test if running on CPU.
        """
        # Initialize ray if not already initialized
        if not ray.is_initialized():
            ray.init(num_cpus=2)
        
        try:
            # Create queues
            inference_results_Q = queue.Queue(maxsize=2)
            param_prompt_Q = queue.Queue(maxsize=2)
            evaluation_inference_results_Q = queue.Queue(maxsize=2)
            
            # Create a small vLLM engine actor with TinyLlama (supported by vLLM)
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            vllm_engine = VLLMEngineActor.remote(model_name)
            
            # Get tokenizer to create real token IDs
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create test prompts
            test_texts = ["Hello world", "Testing vLLM"]
            test_prompts = [tokenizer.encode(text) for text in test_texts]
            
            # Start the vllm_generate_thread
            thread = threading.Thread(
                target=vllm_generate_thread,
                args=(
                    [vllm_engine],  # vllm_engines
                    SamplingParams(temperature=0.0, max_tokens=10),  # generation_config - deterministic, short
                    SamplingParams(temperature=0.0, max_tokens=10),  # eval_generation_config
                    inference_results_Q,
                    param_prompt_Q,
                    1,  # num_training_steps - just one step for testing
                    None,  # eval_prompt_token_ids
                    evaluation_inference_results_Q,
                    2,  # eval_freq
                    1,  # resume_training_step
                    False,  # tool_use
                )
            )
            thread.start()
            
            # Send prompts through the queue (expects tuple with (None, prompts))
            param_prompt_Q.put((None, test_prompts))
            
            # Get results
            try:
                result = inference_results_Q.get(timeout=30)  # Longer timeout for CPU generation
                
                # vllm_generate_thread returns a tuple: (response_ids, finish_reasons, masks, info)
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 4)
                
                response_ids, finish_reasons, masks, info = result
                
                # Verify the structure
                self.assertEqual(len(response_ids), len(test_prompts))
                self.assertEqual(len(finish_reasons), len(test_prompts))
                self.assertEqual(len(masks), len(test_prompts))
                
                # Each response should be a list of token IDs
                for resp in response_ids:
                    self.assertIsInstance(resp, list)
                    self.assertTrue(all(isinstance(token_id, int) for token_id in resp))
                    self.assertGreater(len(resp), 0)  # Should have generated something
                
                # Finish reasons should be strings
                for reason in finish_reasons:
                    self.assertIn(reason, ["stop", "length"])
                
                # Send None to stop the thread
                param_prompt_Q.put(None)
                
            finally:
                # Wait for thread to complete
                thread.join(timeout=5)
            
            # Clean up the actor
            ray.kill(vllm_engine)
            
        finally:
            # Cleanup ray
            ray.shutdown()
    
    def test_multiple_queue_pipeline(self):
        """Test a pipeline with multiple queues like in grpo_fast."""
        # Create queues similar to grpo_fast
        queries_queue = queue.Queue(maxsize=2)
        request_queue = queue.Queue(maxsize=2)
        response_queue = queue.Queue(maxsize=2)
        packed_queue = queue.Queue(maxsize=2)
        
        # Test data
        queries = ["query1", "query2", "query3"]
        
        # Use a stop event to cleanly shut down threads
        stop_event = threading.Event()
        
        def data_prep_thread():
            """Simulates data_preparation_thread from grpo_fast."""
            while not stop_event.is_set():
                try:
                    query = queries_queue.get(timeout=0.1)
                    if query is None:
                        break
                    
                    # Send to inference
                    request_queue.put({"query": query, "params": {}})
                    
                    # Get inference result
                    result = response_queue.get(timeout=1)
                    
                    # Pack and send to packed queue
                    packed = {"query": query, "result": result["response"]}
                    packed_queue.put(packed)
                    
                except queue.Empty:
                    continue
        
        def inference_thread():
            """Simulates vllm_generate_thread from grpo_fast."""
            while not stop_event.is_set():
                try:
                    item = request_queue.get(timeout=0.1)
                    if isinstance(item, dict) and "query" in item:
                        # Simulate generation
                        response = f"Response for {item['query']}"
                        response_queue.put({"response": response})
                except queue.Empty:
                    continue
        
        # Start threads
        threads = [
            threading.Thread(target=data_prep_thread),
            threading.Thread(target=inference_thread)
        ]
        
        for t in threads:
            t.start()
        
        # Send queries
        for query in queries:
            queries_queue.put(query)
        
        # Collect results
        results = []
        for _ in queries:
            try:
                result = packed_queue.get(timeout=2)
                results.append(result)
            except queue.Empty:
                break
        
        # Send sentinel and stop threads
        queries_queue.put(None)
        stop_event.set()
        
        for t in threads:
            t.join(timeout=2)
        
        # Verify results
        self.assertEqual(len(results), len(queries))
        for i, result in enumerate(results):
            self.assertEqual(result["query"], queries[i])
            self.assertEqual(result["result"], f"Response for {queries[i]}")


if __name__ == "__main__":
    unittest.main()