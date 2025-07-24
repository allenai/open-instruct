#!/usr/bin/env python3
"""
Diagnose hanging issues by monitoring thread states and queue operations.
"""

import threading
import time
import psutil
import os
import signal
import sys

class HangDiagnoser:
    def __init__(self):
        self.running = True
        self.start_time = time.time()
        
    def monitor_process(self, pid):
        """Monitor a specific process for hanging."""
        try:
            process = psutil.Process(pid)
            
            print(f"\nMonitoring process {pid} - {process.name()}")
            print("-" * 80)
            
            while self.running:
                # CPU usage
                cpu_percent = process.cpu_percent(interval=1)
                
                # Memory info
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024
                
                # Thread info
                threads = process.threads()
                num_threads = len(threads)
                
                # Check if process appears hung (low CPU for extended time)
                elapsed = time.time() - self.start_time
                
                print(f"\r[{elapsed:.1f}s] CPU: {cpu_percent:.1f}% | "
                      f"Memory: {mem_mb:.1f}MB | "
                      f"Threads: {num_threads} | "
                      f"Status: {process.status()}", end='')
                
                # If CPU is very low for extended period, might be hung
                if cpu_percent < 1.0 and elapsed > 30:
                    print("\n⚠️  WARNING: Process appears to be hung (low CPU usage)")
                    
                    # Try to get thread stack traces (Linux only)
                    if sys.platform.startswith('linux'):
                        try:
                            os.kill(pid, signal.SIGQUIT)  # Dumps thread stacks
                        except:
                            pass
                
                time.sleep(5)
                
        except psutil.NoSuchProcess:
            print(f"\nProcess {pid} no longer exists")
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    
    def stop(self):
        self.running = False

def analyze_logs_for_patterns():
    """Analyze logs to identify where hanging occurs."""
    print("\nCommon hanging patterns to look for:")
    print("-" * 80)
    print("""
1. Queue mismatches:
   - Look for "Waiting for result X/Y" without corresponding "Got result X"
   - Check if param_prompt_Q.put count matches vLLM engine count
   
2. Training step mismatches:
   - vLLM expects step X but receives step Y
   - Data prep thread processing different step than main loop
   
3. Missing initialization:
   - vLLM engines not starting process_from_queue
   - Initial batch not being sent
   
4. Deadlock indicators:
   - "Waiting for packed_sequences_Q.get()" with empty queue
   - "Accumulating results" stuck at 0/N engines
   
5. Check these log patterns:
   - [INITIAL_BATCH] - Was initial batch sent?
   - [VLLM_SETUP] - Did all engines start?
   - [SPLIT_BATCH] - How many batches were split?
   - [vLLM-X] - Are engines receiving requests?
   - [ACCUMULATE] - Is it waiting for all engines?
   - [DATA_PREP_THREAD] - Is it processing correct steps?
   - [MAIN_LOOP] - What step is the main loop on?
""")

def suggest_fixes():
    """Suggest potential fixes based on common issues."""
    print("\nPotential fixes:")
    print("-" * 80)
    print("""
1. If vLLM engines not receiving requests:
   - Check if resume_training_step matches between components
   - Verify initial batch is sent before main loop starts
   
2. If accumulate_inference_batches hangs:
   - Check if number of engines matches args.vllm_num_engines
   - Verify all engines are processing requests
   
3. If data prep thread hangs:
   - Check if it's waiting for more results than engines produce
   - Verify training step ranges align
   
4. Quick diagnostics to add:
   - Add timeout to queue.get() operations
   - Log queue sizes periodically
   - Add heartbeat logging in vLLM engines
""")

if __name__ == "__main__":
    print("Hanging Diagnosis Helper")
    print("=" * 80)
    
    if len(sys.argv) > 1:
        pid = int(sys.argv[1])
        diagnoser = HangDiagnoser()
        
        try:
            diagnoser.monitor_process(pid)
        except KeyboardInterrupt:
            diagnoser.stop()
    else:
        analyze_logs_for_patterns()
        suggest_fixes()
        print("\nUsage: python diagnose_hanging.py [PID] to monitor a specific process")