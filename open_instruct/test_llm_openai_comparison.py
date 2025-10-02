"""
Test comparing LLMRayActor with vLLM's OpenAI-compatible API.

This test loads datasets from the same sources as grpo_fast.py and runs N prompts
through both LLMRayActor and vLLM's OpenAI API server to compare their outputs.
Supports tool use (search and code tools) similar to tool_grpo_fast.sh.

Usage:
    # Run all tests
    uv run pytest open_instruct/test_llm_openai_comparison.py -v

    # Run specific comparison with 5 prompts
    uv run pytest open_instruct/test_llm_openai_comparison.py::TestLLMOpenAIComparison::test_completion_comparison_0 -v

    # Run tool use comparison
    uv run pytest open_instruct/test_llm_openai_comparison.py::TestLLMOpenAIComparison::test_tool_use_comparison -v

Note: This test uses two separate vLLM instances - one for LLMRayActor and one for
the OpenAI API server. While this uses more GPU memory, it allows testing the full
OpenAI API compatibility and matches typical production deployment patterns.
"""

import logging
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import unittest
from dataclasses import dataclass, field
from typing import List, Optional

import openai
import parameterized
import ray
import transformers
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group
from tqdm import tqdm

from open_instruct import vllm_utils3
from open_instruct.dataset_transformation import (
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    TokenizerConfig,
    get_cached_dataset_tulu,
)
from open_instruct.queue_types import PromptRequest
from open_instruct.search_utils.search_tool import SearchTool
from open_instruct.tool_utils.tools import MaxCallsExceededTool, PythonCodeTool
from open_instruct.utils import ray_get_with_progress

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonConfig:
    """Configuration for the comparison test."""

    # Model configuration
    model_name_or_path: str = "Qwen/Qwen3-1.7B"

    # Dataset configuration
    dataset_mixer_list: List[str] = field(
        default_factory=lambda: ["hamishivi/tulu_3_rewritten_100k_with_tool_prompt", "1.0"]
    )
    dataset_mixer_list_splits: List[str] = field(default_factory=lambda: ["train"])
    dataset_transform_fn: List[str] = field(default_factory=lambda: ["rlvr_tokenize_v1", "rlvr_filter_v1"])
    max_token_length: int = 512
    max_prompt_token_length: int = 512
    response_length: int = 512

    # Generation configuration
    temperature: float = 1.0
    top_p: float = 1.0
    stop_strings: Optional[List[str]] = field(default_factory=lambda: ["</answer>"])

    # Tool configuration
    tools: Optional[List[str]] = field(default_factory=lambda: ["code", "search"])
    max_tool_calls: int = 5
    search_api_endpoint: Optional[str] = "http://neptune-cs-aus-258.reviz.ai2.in:43189/search"
    code_tool_api_endpoint: Optional[str] = "https://open-instruct-tool-server-10554368204.us-central1.run.app/execute"
    number_documents_to_search: int = 3

    # Test configuration
    num_prompts_to_test: int = 5
    seed: int = 42

    # vLLM configuration
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.4
    vllm_port: int = 8000

    # Comparison configuration
    comparison_tolerance: float = 0.01  # For numerical comparisons


def find_free_port():
    """Find a free port to use for the vLLM server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class TestLLMOpenAIComparison(unittest.TestCase):
    """Test comparing LLMRayActor with vLLM's OpenAI-compatible API."""

    @classmethod
    def setUpClass(cls):
        """Setup that runs once for the entire test class."""
        logger.info("\n" + "=" * 80)
        logger.info("INITIALIZING TEST CLASS")
        logger.info("=" * 80)

        cls.config = ComparisonConfig()
        cls.tokenizer = None
        cls.dataset = None
        cls.vllm_process = None
        cls.llm_ray_actor = None
        cls.openai_client = None
        cls.tools = {}

        # Initialize Ray if not already initialized
        logger.info("Checking Ray initialization...")
        if not ray.is_initialized():
            logger.info("Initializing Ray with 1 GPU...")
            ray.init(ignore_reinit_error=True, num_gpus=1)
            logger.info("✓ Ray initialized")
        else:
            logger.info("✓ Ray already initialized")

    def setUp(self):
        """Setup for each test method."""
        logger.info("=" * 80)
        logger.info("STARTING TEST SETUP")
        logger.info("=" * 80)

        # Initialize tokenizer
        logger.info(f"Loading tokenizer from {self.config.model_name_or_path}...")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, padding_side="left", trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("✓ Tokenizer loaded successfully")

        # Setup tokenizer config
        self.tc = TokenizerConfig(tokenizer_name_or_path=self.config.model_name_or_path, trust_remote_code=True)

        # Load dataset
        logger.info("\n--- DATASET LOADING ---")
        self._load_dataset()

        # Setup tools
        logger.info("\n--- TOOL SETUP ---")
        self._setup_tools()

        # Start vLLM OpenAI server
        logger.info("\n--- STARTING VLLM OPENAI SERVER ---")
        self._start_vllm_server()

        # Initialize OpenAI client
        logger.info("\n--- INITIALIZING OPENAI CLIENT ---")
        self.openai_client = openai.Client(
            base_url=f"http://localhost:{self.config.vllm_port}/v1",
            api_key="dummy",  # vLLM doesn't require a real API key
        )
        logger.info("✓ OpenAI client initialized")

        logger.info("\n" + "=" * 80)
        logger.info("TEST SETUP COMPLETE")
        logger.info("=" * 80 + "\n")

    def _load_dataset(self):
        """Load the dataset using the same logic as grpo_fast.py."""
        logger.info(f"Loading dataset: {self.config.dataset_mixer_list[0]}")
        logger.info(f"Dataset splits: {self.config.dataset_mixer_list_splits}")
        logger.info(f"Transform functions: {self.config.dataset_transform_fn}")
        logger.info(f"Max token length: {self.config.max_token_length}")

        transform_fn_args = [
            {},
            {
                "max_token_length": self.config.max_token_length,
                "max_prompt_token_length": self.config.max_prompt_token_length,
            },
        ]

        logger.info("Calling get_cached_dataset_tulu (this may take some time)...")
        start_time = time.time()

        self.dataset = get_cached_dataset_tulu(
            dataset_mixer_list=self.config.dataset_mixer_list,
            dataset_mixer_list_splits=self.config.dataset_mixer_list_splits,
            tc=self.tc,
            dataset_transform_fn=self.config.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            dataset_cache_mode="local",
            dataset_config_hash=None,
            hf_entity=None,
            dataset_local_cache_dir="local_dataset_cache",
            dataset_skip_cache=True,  # Skip cache for testing
        )

        # Shuffle dataset with seed for reproducibility
        logger.info(f"Shuffling dataset with seed {self.config.seed}...")
        self.dataset = self.dataset.shuffle(seed=self.config.seed)

        elapsed = time.time() - start_time
        logger.info(f"✓ Dataset loaded successfully in {elapsed:.1f}s")
        logger.info(f"✓ Total examples: {len(self.dataset)}")

    def _setup_tools(self):
        """Setup tools for both LLMRayActor and comparison."""
        if "search" in (self.config.tools or []):
            self.tools["search"] = SearchTool(
                start_str="<query>",
                end_str="</query>",
                api_endpoint=self.config.search_api_endpoint,
                number_documents_to_search=self.config.number_documents_to_search,
            )

        if "code" in (self.config.tools or []):
            self.tools["code"] = PythonCodeTool(
                start_str="<code>", end_str="</code>", api_endpoint=self.config.code_tool_api_endpoint
            )

        # Add max calls exceeded tool
        self.tools["</tool>"] = MaxCallsExceededTool(start_str="<tool>", end_str="</tool>")

        logger.info(f"Initialized tools: {list(self.tools.keys())}")

    def _start_vllm_server(self):
        """Start the vLLM OpenAI-compatible server."""
        # Find a free port
        self.config.vllm_port = find_free_port()
        logger.info(f"Using port {self.config.vllm_port} for vLLM server")

        # Build vLLM server command
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.config.model_name_or_path,
            "--port",
            str(self.config.vllm_port),
            "--served-model-name",
            "test-model",
            "--tensor-parallel-size",
            str(self.config.vllm_tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.config.vllm_gpu_memory_utilization),
            "--trust-remote-code",
        ]

        logger.info("Server command:")
        logger.info(f"  {' '.join(cmd)}")

        # Start the server process
        logger.info("Starting vLLM server process...")
        self.vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,  # Create new process group for cleanup
        )

        # Collect stderr and stdout output for debugging
        self.vllm_stderr_lines = []
        self.vllm_stdout_lines = []

        # Start a thread to log server stderr output
        def log_stderr():
            for line in self.vllm_process.stderr:
                if line.strip():
                    self.vllm_stderr_lines.append(line.strip())
                    logger.info(f"[vLLM stderr] {line.strip()}")

        def log_stdout():
            for line in self.vllm_process.stdout:
                if line.strip():
                    self.vllm_stdout_lines.append(line.strip())
                    logger.info(f"[vLLM stdout] {line.strip()}")

        stderr_thread = threading.Thread(target=log_stderr, daemon=True)
        stderr_thread.start()
        stdout_thread = threading.Thread(target=log_stdout, daemon=True)
        stdout_thread.start()

        # Wait for server to be ready
        max_wait = 180  # seconds - allow time for model loading
        start_time = time.time()
        logger.info(f"Waiting for server to be ready (timeout: {max_wait}s)...")

        # First wait for specific initialization messages
        logger.info("Waiting for 'Available KV cache memory' message...")
        kv_cache_seen = False
        while time.time() - start_time < max_wait:
            # Check if process is still alive
            if self.vllm_process.poll() is not None:
                logger.error("vLLM server process terminated unexpectedly!")
                logger.error("Last 50 lines of stderr:")
                for line in self.vllm_stderr_lines[-50:]:
                    logger.error(f"  {line}")
                raise RuntimeError("vLLM server process terminated unexpectedly")

            # Check for KV cache message
            for line in self.vllm_stdout_lines:
                if "Available KV cache memory" in line:
                    kv_cache_seen = True
                    break

            if kv_cache_seen:
                logger.info("✓ KV cache initialized, waiting for server to be ready...")
                break

            time.sleep(2)

        if not kv_cache_seen:
            logger.warning("Did not see KV cache initialization message, continuing anyway...")

        # Now try to connect to the server
        attempt = 0
        while time.time() - start_time < max_wait:
            attempt += 1
            elapsed = int(time.time() - start_time)

            # Check if process is still alive
            if self.vllm_process.poll() is not None:
                logger.error("vLLM server process terminated unexpectedly!")
                logger.error("Last 50 lines of stderr:")
                for line in self.vllm_stderr_lines[-50:]:
                    logger.error(f"  {line}")
                raise RuntimeError("vLLM server process terminated unexpectedly")

            try:
                # Try to connect to the server
                _ = openai.Client(
                    base_url=f"http://localhost:{self.config.vllm_port}/v1", api_key="dummy"
                ).models.list()
                logger.info(f"✓ vLLM server is ready after {elapsed}s ({attempt} attempts)")
                break
            except Exception as e:
                if attempt % 10 == 0:  # Log every 10 attempts
                    logger.info(f"  Still waiting... ({elapsed}/{max_wait}s, attempt {attempt})")
                    logger.debug(f"  Last connection error: {e}")
                time.sleep(2)
        else:
            logger.error("vLLM server failed to start within timeout!")
            logger.error(
                f"Captured {len(self.vllm_stdout_lines)} lines of stdout, {len(self.vllm_stderr_lines)} lines of stderr"
            )
            logger.error("Last 50 lines of stdout:")
            for line in self.vllm_stdout_lines[-50:]:
                logger.error(f"  {line}")
            logger.error("Last 100 lines of stderr:")
            for line in self.vllm_stderr_lines[-100:]:
                logger.error(f"  {line}")
            raise RuntimeError("vLLM server failed to start within timeout")

    def _init_llm_ray_actor(self):
        """Initialize the LLMRayActor."""
        logger.info("Creating Ray placement group...")
        # Create placement group
        bundles = [{"GPU": 1, "CPU": 10}]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray_get_with_progress([pg.ready()], desc="Waiting for placement group")
        logger.info("✓ Placement group ready")

        # Get bundle indices
        bundle_indices = vllm_utils3.get_bundle_indices_list(pg)
        logger.info(f"Bundle indices: {bundle_indices}")

        # Create queues
        logger.info("Creating Ray queues...")
        self.prompt_queue = ray_queue.Queue()
        self.results_queue = ray_queue.Queue()
        self.eval_results_queue = ray_queue.Queue()

        # Create a mock actor manager
        @ray.remote
        class MockActorManager:
            def should_stop(self):
                return False

        logger.info("Creating mock actor manager...")
        actor_manager = MockActorManager.remote()

        # Initialize LLMRayActor
        logger.info("Creating LLMRayActor (this will load the model)...")
        logger.info(f"  Model: {self.config.model_name_or_path}")
        logger.info(f"  Tensor parallel size: {self.config.vllm_tensor_parallel_size}")
        logger.info(f"  GPU memory utilization: {self.config.vllm_gpu_memory_utilization}")

        self.llm_ray_actor = ray.remote(num_cpus=10, num_gpus=1)(vllm_utils3.LLMRayActor).remote(
            self.config.model_name_or_path,
            tools=self.tools,
            max_tool_calls={"</tool>": self.config.max_tool_calls},
            bundle_indices=bundle_indices,
            prompt_queue=self.prompt_queue,
            results_queue=self.results_queue,
            eval_results_queue=self.eval_results_queue,
            actor_manager=actor_manager,
            inference_batch_size=self.config.num_prompts_to_test,
            inflight_updates=False,
            verbose=True,
            # vLLM args
            noset_visible_devices=False,
            distributed_executor_backend="ray",
            num_gpus=1,
            tensor_parallel_size=self.config.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            max_model_len=self.config.max_token_length + self.config.response_length,
            trust_remote_code=True,
            disable_log_stats=True,
        )

        # Initialize the actor
        logger.info("Waiting for model to load...")
        start_time = time.time()
        model_dims = ray.get(self.llm_ray_actor.get_model_dims_dict.remote())
        elapsed = time.time() - start_time
        logger.info(f"✓ LLMRayActor initialized in {elapsed:.1f}s")
        logger.info(f"  Model dimensions: {model_dims}")

    def _generate_with_llm_ray_actor(self, prompts: List[str]) -> List[str]:
        """Generate completions using LLMRayActor.

        This mimics how grpo_fast.py uses the LLMRayActor for generation,
        including tool support if configured.
        """
        results = []

        # Submit all prompts to the queue
        logger.info(f"Submitting {len(prompts)} prompts to LLMRayActor...")
        for i, prompt in enumerate(prompts):
            dataset_idx = i
            prompt_request = PromptRequest(
                dataset_index=dataset_idx,
                prompt=prompt,
                is_eval=False,
                training_step=0,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.response_length,
                stop=self.config.stop_strings,
                n=1,  # Number of completions per prompt
            )
            self.prompt_queue.put(prompt_request)
            logger.debug(f"  Submitted prompt {i + 1}/{len(prompts)}")

        # Start generation
        logger.info("Starting generation on LLMRayActor...")
        generation_future = self.llm_ray_actor.generate.remote()

        # Collect results
        logger.info(f"Waiting for {len(prompts)} results from LLMRayActor...")
        for i in tqdm(range(len(prompts)), desc="LLMRayActor generation"):
            try:
                result = self.results_queue.get(timeout=30)
                results.append(result)
                logger.debug(f"  Received result {i + 1}/{len(prompts)}")
            except queue.Empty:
                logger.error(f"Timeout waiting for LLMRayActor result {i + 1}")
                results.append(None)

        # Stop generation
        ray.cancel(generation_future)
        logger.info(f"✓ Collected {len(results)} results from LLMRayActor")

        return results

    def _generate_with_openai_api(self, prompts: List[str]) -> List[str]:
        """Generate completions using OpenAI-compatible API."""
        results = []

        logger.info(f"Generating {len(prompts)} completions via OpenAI API...")
        for i, prompt in enumerate(tqdm(prompts, desc="OpenAI API generation")):
            try:
                logger.debug(f"  Sending prompt {i + 1}/{len(prompts)} to OpenAI API")
                response = self.openai_client.completions.create(
                    model="test-model",
                    prompt=prompt,
                    max_tokens=self.config.response_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    stop=self.config.stop_strings,
                    n=1,
                )
                results.append(response.choices[0].text)
                logger.debug(f"  Received response {i + 1}/{len(prompts)}")
            except Exception as e:
                logger.error(f"Error generating with OpenAI API for prompt {i + 1}: {e}")
                results.append(None)

        logger.info(f"✓ Collected {len(results)} results from OpenAI API")
        return results

    def _compare_completions(self, prompts: List[str], llm_ray_results: List, openai_results: List[str]):
        """Compare completions from both methods."""
        comparison_results = []

        for i, (prompt, llm_result, openai_result) in enumerate(zip(prompts, llm_ray_results, openai_results)):
            result = {
                "prompt_index": i,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "llm_ray_completion": None,
                "openai_completion": openai_result,
                "match": False,
                "differences": [],
            }

            # Extract completion from LLMRayActor result
            if llm_result and hasattr(llm_result, "responses"):
                if llm_result.responses:
                    # Decode token IDs to text
                    llm_text = self.tokenizer.decode(llm_result.responses[0], skip_special_tokens=True)
                    result["llm_ray_completion"] = llm_text

                    # Compare completions
                    if openai_result and llm_text:
                        # Simple equality check
                        if llm_text.strip() == openai_result.strip():
                            result["match"] = True
                        else:
                            # Find differences
                            result["differences"].append("Text mismatch")
                            result["differences"].append(f"LLM length: {len(llm_text)}")
                            result["differences"].append(f"OpenAI length: {len(openai_result)}")

            comparison_results.append(result)

        return comparison_results

    @parameterized.parameterized.expand(
        [
            (5,),  # Test with 5 prompts
            (10,),  # Test with 10 prompts
        ]
    )
    def test_completion_comparison(self, num_prompts):
        """Test completions from OpenAI API."""
        logger.info("\n" + "=" * 80)
        logger.info(f"RUNNING TEST: test_completion_comparison with {num_prompts} prompts")
        logger.info("=" * 80)

        # Select prompts from dataset
        logger.info(f"Selecting {num_prompts} prompts from dataset...")
        prompts = []
        for i in range(min(num_prompts, len(self.dataset))):
            example = self.dataset[i]
            # Get the raw prompt
            if RAW_PROMPT_KEY in example:
                prompts.append(example[RAW_PROMPT_KEY])
            else:
                # Decode from token IDs if raw prompt not available
                prompt_ids = example[INPUT_IDS_PROMPT_KEY]
                prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                prompts.append(prompt)

        logger.info(f"✓ Selected {len(prompts)} prompts")
        for i, prompt in enumerate(prompts[:3]):  # Log first 3 prompts as examples
            preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            logger.info(f"  Prompt {i + 1}: {preview}")

        # Generate with OpenAI API
        logger.info("\n--- GENERATING WITH OPENAI API ---")
        start_time = time.time()
        openai_results = self._generate_with_openai_api(prompts)
        openai_time = time.time() - start_time
        logger.info(f"OpenAI API generation took {openai_time:.1f}s")

        # Print report
        logger.info("\n" + "=" * 80)
        logger.info("GENERATION REPORT")
        logger.info("=" * 80)
        logger.info(f"Number of prompts: {len(prompts)}")
        logger.info(f"OpenAI API time: {openai_time:.2f}s")
        logger.info(f"Successful completions: {sum(1 for r in openai_results if r is not None)}/{len(openai_results)}")

        # Log sample results
        for i, (prompt, result) in enumerate(zip(prompts[:3], openai_results[:3])):
            logger.info(f"\nSample {i + 1}:")
            logger.info(f"  Prompt: {prompt[:100]}...")
            if result:
                logger.info(f"  Completion: {result[:100]}...")
            else:
                logger.info("  Completion: None (failed)")

        # Assert some basic checks
        self.assertGreater(len(openai_results), 0, "No results from OpenAI API")
        self.assertGreater(sum(1 for r in openai_results if r is not None), 0, "All completions failed")

    def test_tool_use_comparison(self):
        """Test tool use behavior with OpenAI API."""
        # Find prompts that likely trigger tool use
        tool_prompts = []
        for i in range(min(10, len(self.dataset))):
            example = self.dataset[i]
            prompt = example.get(RAW_PROMPT_KEY, "")

            # Look for prompts that might trigger search or code tools
            if "search" in prompt.lower() or "find" in prompt.lower() or "code" in prompt.lower():
                tool_prompts.append(prompt)

            if len(tool_prompts) >= 3:  # Test with 3 tool prompts
                break

        if not tool_prompts:
            # Use generic prompts if no tool-specific ones found
            tool_prompts = [
                "Search for information about Python data structures",
                "Write code to calculate the factorial of a number",
                "Find the latest news about artificial intelligence",
            ]

        logger.info(f"Testing tool use with {len(tool_prompts)} prompts")

        # Generate with OpenAI API
        openai_results = self._generate_with_openai_api(tool_prompts)

        # Analyze tool usage in results
        for i, (prompt, openai_result) in enumerate(zip(tool_prompts, openai_results)):
            logger.info(f"\nTool prompt {i}: {prompt[:50]}...")

            # Check for tool markers in OpenAI completion
            if openai_result:
                if "<query>" in openai_result or "<code>" in openai_result:
                    logger.info("  OpenAI completion contains tool markers")
                else:
                    logger.info("  No tool markers found in completion")

    def tearDown(self):
        """Cleanup after each test."""
        logger.info("\n" + "=" * 80)
        logger.info("CLEANING UP TEST")
        logger.info("=" * 80)

        # Shutdown vLLM server
        if self.vllm_process:
            logger.info("Shutting down vLLM server...")
            try:
                # Kill the process group to ensure all child processes are terminated
                os.killpg(os.getpgid(self.vllm_process.pid), signal.SIGTERM)
                self.vllm_process.wait(timeout=5)
                logger.info("✓ vLLM server shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down vLLM server: {e}")
                self.vllm_process.kill()

        # Clear references
        self.vllm_process = None
        self.openai_client = None

        logger.info("✓ Test cleanup complete\n")

    @classmethod
    def tearDownClass(cls):
        """Cleanup after all tests."""
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Run tests
    unittest.main(verbosity=2)
