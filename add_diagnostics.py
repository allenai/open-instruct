#!/usr/bin/env python3
"""
Script to add diagnostic logging to grpo_fast.py and vllm_utils3.py
to identify queue flow mismatches.

Usage: python add_diagnostics.py
"""

import re

def add_queue_diagnostics_to_grpo_fast():
    """Add logging to track queue operations in grpo_fast.py"""
    
    with open("open_instruct/grpo_fast.py", "r") as f:
        content = f.read()
    
    # Add logging to split_and_insert_batch
    pattern = r'(param_prompt_Q\.put\([\s\S]*?\)\s*\))'
    replacement = r'logger.info(f"[QUEUE] param_prompt_Q.put: training_step={training_step}, batch_idx={batch_idx}, num_prompts={len(batch_queries)}")\n        \1'
    content = re.sub(pattern, replacement, content)
    
    # Add logging to accumulate_inference_batches
    pattern = r'(result = inference_results_Q\.get\(\))'
    replacement = r'logger.info(f"[QUEUE] inference_results_Q.get: waiting for result {batch_idx}/{args.vllm_num_engines}")\n        \1\n        logger.info(f"[QUEUE] inference_results_Q.get: got result for training_step={result.training_step if hasattr(result, \"training_step\") else \"unknown\"}")'
    content = re.sub(pattern, replacement, content)
    
    # Add logging to data_preparation_thread loop
    pattern = r'(for training_step in range\(1, num_training_steps \+ 1\):)'
    replacement = r'\1\n        logger.info(f"[DATA_PREP] Starting iteration for training_step={training_step}/{num_training_steps}")'
    content = re.sub(pattern, replacement, content)
    
    # Add logging to main training loop
    pattern = r'(for training_step in range\(resume_training_step, args\.num_training_steps \+ 1\):)'
    replacement = r'\1\n            logger.info(f"[MAIN_LOOP] Starting iteration for training_step={training_step}/{args.num_training_steps}")'
    content = re.sub(pattern, replacement, content)
    
    with open("open_instruct/grpo_fast_debug.py", "w") as f:
        f.write(content)
    
    print("Created grpo_fast_debug.py with diagnostic logging")

def add_queue_diagnostics_to_vllm_utils():
    """Add logging to track queue operations in vllm_utils3.py"""
    
    with open("open_instruct/vllm_utils3.py", "r") as f:
        content = f.read()
    
    # Add logging to process_from_queue
    pattern = r'(request = self\.prompt_queue\.get\(\))'
    replacement = r'logger.info(f"[vLLM-{self.rank}] Waiting for prompt_queue.get() for step {training_step}")\n            \1\n            logger.info(f"[vLLM-{self.rank}] Got request: training_step={getattr(request, \"training_step\", \"unknown\") if request else \"None\"}")'
    content = re.sub(pattern, replacement, content)
    
    pattern = r'(self\.results_queue\.put\(result\))'
    replacement = r'logger.info(f"[vLLM-{self.rank}] Putting result in results_queue for training_step={request.training_step}")\n            \1'
    content = re.sub(pattern, replacement, content)
    
    with open("open_instruct/vllm_utils3_debug.py", "w") as f:
        f.write(content)
    
    print("Created vllm_utils3_debug.py with diagnostic logging")

def create_queue_monitor():
    """Create a standalone queue monitoring script"""
    
    monitor_script = '''#!/usr/bin/env python3
"""
Monitor Ray queues to diagnose hanging issues.
Run this in parallel with your training script.
"""

import ray
import time
import sys

def monitor_queues():
    """Monitor Ray queue states."""
    # Connect to existing Ray cluster
    ray.init(address='auto')
    
    print("Monitoring Ray queues... Press Ctrl+C to stop")
    print("-" * 80)
    
    while True:
        try:
            # Get queue info from Ray (this is pseudocode - actual implementation depends on Ray version)
            # You may need to access queues through Ray's internal APIs
            
            print(f"\\n[{time.strftime('%H:%M:%S')}] Queue Status:")
            # Add actual queue monitoring logic here
            
            time.sleep(5)
        except KeyboardInterrupt:
            print("\\nStopping monitor...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_queues()
'''
    
    with open("monitor_queues.py", "w") as f:
        f.write(monitor_script)
    
    print("Created monitor_queues.py")

if __name__ == "__main__":
    print("Adding diagnostic logging to identify queue mismatches...")
    add_queue_diagnostics_to_grpo_fast()
    add_queue_diagnostics_to_vllm_utils()
    create_queue_monitor()
    print("\nDiagnostic files created. Use the _debug.py versions to run with extra logging.")
    print("\nLook for patterns like:")
    print("- param_prompt_Q.put called but vLLM never calls get")
    print("- Training step numbers that don't match between components")
    print("- Queues that fill up but never drain")