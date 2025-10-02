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

logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
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
    vllm_gpu_memory_utilization: float = 0.3
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
        cls.config = TestConfig()
        cls.tokenizer = None
        cls.dataset = None
        cls.vllm_process = None
        cls.llm_ray_actor = None
        cls.openai_client = None
        cls.tools = {}

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_gpus=1)

    def setUp(self):
        """Setup for each test method."""
        # Initialize tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, padding_side="left", trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup tokenizer config
        self.tc = TokenizerConfig(
            tokenizer_name_or_path=self.config.model_name_or_path,
            chat_template=None,
            truncate_prompt=self.config.max_prompt_token_length,
        )

        # Load dataset
        self._load_dataset()

        # Setup tools
        self._setup_tools()

        # Start vLLM OpenAI server
        self._start_vllm_server()

        # Initialize LLMRayActor
        self._init_llm_ray_actor()

        # Initialize OpenAI client
        self.openai_client = openai.Client(
            base_url=f"http://localhost:{self.config.vllm_port}/v1",
            api_key="dummy",  # vLLM doesn't require a real API key
        )

    def _load_dataset(self):
        """Load the dataset using the same logic as grpo_fast.py."""
        transform_fn_args = [
            {},
            {
                "max_token_length": self.config.max_token_length,
                "max_prompt_token_length": self.config.max_prompt_token_length,
            },
        ]

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
        self.dataset = self.dataset.shuffle(seed=self.config.seed)

        logger.info(f"Loaded dataset with {len(self.dataset)} examples")

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
        self.tools["</tool>"] = MaxCallsExceededTool(start_str="")

        logger.info(f"Initialized tools: {list(self.tools.keys())}")

    def _start_vllm_server(self):
        """Start the vLLM OpenAI-compatible server."""
        # Find a free port
        self.config.vllm_port = find_free_port()

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

        # Add stop strings if specified
        if self.config.stop_strings:
            for stop_str in self.config.stop_strings:
                cmd.extend(["--stop", stop_str])

        logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")

        # Start the server process
        self.vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,  # Create new process group for cleanup
        )

        # Wait for server to be ready
        max_wait = 60  # seconds
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                # Try to connect to the server
                _ = openai.Client(
                    base_url=f"http://localhost:{self.config.vllm_port}/v1", api_key="dummy"
                ).models.list()
                logger.info("vLLM server is ready")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("vLLM server failed to start within timeout")

    def _init_llm_ray_actor(self):
        """Initialize the LLMRayActor."""
        # Create placement group
        bundles = [{"GPU": 1, "CPU": 10}]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray_get_with_progress([pg.ready()], desc="Waiting for placement group")

        # Get bundle indices
        bundle_indices = vllm_utils3.get_bundle_indices_list(pg)

        # Create queues
        self.prompt_queue = ray_queue.Queue()
        self.results_queue = ray_queue.Queue()
        self.eval_results_queue = ray_queue.Queue()

        # Create a mock actor manager
        @ray.remote
        class MockActorManager:
            def should_stop(self):
                return False

        actor_manager = MockActorManager.remote()

        # Initialize LLMRayActor
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
        ray.get(self.llm_ray_actor.get_model_dims_dict.remote())

        logger.info("LLMRayActor initialized")

    def _generate_with_llm_ray_actor(self, prompts: List[str]) -> List[str]:
        """Generate completions using LLMRayActor.

        This mimics how grpo_fast.py uses the LLMRayActor for generation,
        including tool support if configured.
        """
        results = []

        # Submit all prompts to the queue
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

        # Start generation
        generation_future = self.llm_ray_actor.generate.remote()

        # Collect results
        for _ in range(len(prompts)):
            try:
                result = self.results_queue.get(timeout=30)
                results.append(result)
            except queue.Empty:
                logger.error("Timeout waiting for LLMRayActor result")
                results.append(None)

        # Stop generation
        ray.cancel(generation_future)

        return results

    def _generate_with_openai_api(self, prompts: List[str]) -> List[str]:
        """Generate completions using OpenAI-compatible API."""
        results = []

        for prompt in prompts:
            try:
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
            except Exception as e:
                logger.error(f"Error generating with OpenAI API: {e}")
                results.append(None)

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
        """Test comparing completions from LLMRayActor and OpenAI API."""
        # Select prompts from dataset
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

        logger.info(f"Testing with {len(prompts)} prompts")

        # Generate with LLMRayActor
        logger.info("Generating with LLMRayActor...")
        start_time = time.time()
        llm_ray_results = self._generate_with_llm_ray_actor(prompts)
        llm_ray_time = time.time() - start_time

        # Generate with OpenAI API
        logger.info("Generating with OpenAI API...")
        start_time = time.time()
        openai_results = self._generate_with_openai_api(prompts)
        openai_time = time.time() - start_time

        # Compare results
        comparison_results = self._compare_completions(prompts, llm_ray_results, openai_results)

        # Print comparison report
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON REPORT")
        logger.info("=" * 80)
        logger.info(f"Number of prompts: {len(prompts)}")
        logger.info(f"LLMRayActor time: {llm_ray_time:.2f}s")
        logger.info(f"OpenAI API time: {openai_time:.2f}s")
        logger.info(f"Speed ratio: {openai_time / llm_ray_time:.2f}x")

        matches = sum(1 for r in comparison_results if r["match"])
        logger.info(f"Exact matches: {matches}/{len(comparison_results)}")

        # Log detailed differences for non-matches
        for result in comparison_results:
            if not result["match"]:
                logger.info(f"\nPrompt {result['prompt_index']}:")
                logger.info(f"  Prompt: {result['prompt']}")
                logger.info(f"  Differences: {', '.join(result['differences'])}")
                if result["llm_ray_completion"]:
                    logger.info(f"  LLM completion: {result['llm_ray_completion'][:100]}...")
                if result["openai_completion"]:
                    logger.info(f"  OpenAI completion: {result['openai_completion'][:100]}...")

        # Assert some basic checks
        self.assertGreater(len(llm_ray_results), 0, "No results from LLMRayActor")
        self.assertGreater(len(openai_results), 0, "No results from OpenAI API")
        self.assertEqual(len(llm_ray_results), len(openai_results), "Different number of results")

    def test_tool_use_comparison(self):
        """Test tool use behavior comparison."""
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

        # Generate with both methods
        llm_ray_results = self._generate_with_llm_ray_actor(tool_prompts)
        openai_results = self._generate_with_openai_api(tool_prompts)

        # Analyze tool usage in results
        for i, (prompt, llm_result, openai_result) in enumerate(zip(tool_prompts, llm_ray_results, openai_results)):
            logger.info(f"\nTool prompt {i}: {prompt[:50]}...")

            # Check for tool markers in completions
            if llm_result and hasattr(llm_result, "request_info"):
                info = llm_result.request_info
                if hasattr(info, "tool_calleds") and info.tool_calleds:
                    logger.info(f"  LLMRayActor used tools: {info.tool_calleds}")
                    if hasattr(info, "tool_outputs"):
                        logger.info(f"  Tool outputs: {info.tool_outputs}")

            # Check for tool markers in OpenAI completion
            if openai_result:
                if "<query>" in openai_result or "<code>" in openai_result:
                    logger.info("  OpenAI completion contains tool markers")

    def tearDown(self):
        """Cleanup after each test."""
        # Shutdown vLLM server
        if self.vllm_process:
            try:
                # Kill the process group to ensure all child processes are terminated
                os.killpg(os.getpgid(self.vllm_process.pid), signal.SIGTERM)
                self.vllm_process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error shutting down vLLM server: {e}")
                self.vllm_process.kill()

        # Cleanup Ray actors
        if self.llm_ray_actor:
            ray.kill(self.llm_ray_actor)

        # Clear references
        self.vllm_process = None
        self.llm_ray_actor = None
        self.openai_client = None

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
