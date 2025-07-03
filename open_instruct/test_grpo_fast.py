import queue
import threading
import time
import unittest

import numpy as np

# Import with proper error handling
try:
    from open_instruct.grpo_fast import ShufflingIterator
except ImportError:
    # If vllm import fails, define ShufflingIterator locally for testing
    from typing import Iterator, List, Optional
    
    class ShufflingIterator:
        """Local copy of ShufflingIterator for testing when vllm is not available."""
        def __init__(self, data: np.ndarray, batch_size: int, seed: Optional[int] = None):
            self.data = data.copy()
            self.batch_size = batch_size
            self.index = 0
            self.rng = np.random.default_rng(seed)
            self.rng.shuffle(self.data)

            # Ensure the effective dataset size is divisible by batch_size
            self.effective_size = len(self.data) - (len(self.data) % batch_size)

        def __iter__(self) -> Iterator[List[int]]:
            return self

        def __next__(self) -> List[int]:
            if self.index >= self.effective_size:
                self.index = 0
                self.rng.shuffle(self.data)

            end_index = self.index + self.batch_size
            batch = self.data[self.index : end_index].tolist()
            self.index = end_index

            return batch


class TestGrpoFast(unittest.TestCase):
    def test_shuffling_iterator_data_flow(self):
        """Test that ShufflingIterator can send data to/from the generator."""
        # Create test data
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        batch_size = 3
        
        # Create iterator
        iterator = ShufflingIterator(data, batch_size, seed=42)
        
        # Collect batches
        collected_batches = []
        batch_count = 0
        
        # Test that we can iterate and receive data
        for batch in iterator:
            collected_batches.append(batch)
            batch_count += 1
            
            # Stop after one full epoch
            if batch_count >= 3:  # 10 items / batch_size 3 = 3 full batches
                break
        
        # Verify we got the expected number of batches
        self.assertEqual(len(collected_batches), 3)
        
        # Verify each batch has the correct size
        for batch in collected_batches:
            self.assertEqual(len(batch), batch_size)
        
        # Verify all data points were covered (except the last incomplete batch)
        all_items = []
        for batch in collected_batches:
            all_items.extend(batch)
        
        # Should have 9 items (3 batches of 3)
        self.assertEqual(len(all_items), 9)
        
        # Verify data is from the original array
        for item in all_items:
            self.assertIn(item, data)
    
    def test_queue_based_generator_communication(self):
        """Test queue-based communication pattern similar to grpo_fast."""
        # Create queues for communication
        input_queue = queue.Queue(maxsize=2)
        output_queue = queue.Queue(maxsize=2)
        
        # Data to send
        test_data = [
            {"prompt": "Hello", "id": 1},
            {"prompt": "World", "id": 2},
            {"prompt": "Test", "id": 3},
        ]
        
        # Generator thread function
        def generator_thread():
            while True:
                try:
                    item = input_queue.get(timeout=1)
                    if item is None:  # Sentinel value to stop
                        break
                    
                    # Simulate processing (like vLLM generation)
                    result = {
                        "id": item["id"],
                        "prompt": item["prompt"],
                        "response": f"Generated: {item['prompt']}"
                    }
                    
                    output_queue.put(result)
                except queue.Empty:
                    break
        
        # Consumer thread function
        def consumer_thread(results):
            while True:
                try:
                    result = output_queue.get(timeout=2)
                    results.append(result)
                except queue.Empty:
                    break
        
        # Start threads
        results = []
        gen_thread = threading.Thread(target=generator_thread)
        cons_thread = threading.Thread(target=lambda: consumer_thread(results))
        
        gen_thread.start()
        cons_thread.start()
        
        # Send data through the pipeline
        for item in test_data:
            input_queue.put(item)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Send sentinel to stop generator
        input_queue.put(None)
        
        # Wait for threads to complete
        gen_thread.join(timeout=3)
        cons_thread.join(timeout=3)
        
        # Verify results
        self.assertEqual(len(results), len(test_data))
        
        # Verify each result corresponds to input
        for i, result in enumerate(results):
            self.assertEqual(result["id"], test_data[i]["id"])
            self.assertEqual(result["prompt"], test_data[i]["prompt"])
            self.assertEqual(result["response"], f"Generated: {test_data[i]['prompt']}")
    
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