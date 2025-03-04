import os
import re
import requests
import ray
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

from open_instruct.vllm_utils2 import WorkerWrap


def get_snippets_for_query(query: str) -> List[str]:
    """
    Retrieves the first snippet from a web search API for the given query.
    Raises a ValueError if the API key is missing.
    """
    api_key = os.environ.get("YOUCOM_API_KEY")
    if not api_key:
        raise ValueError("Missing YOUCOM_API_KEY environment variable.")
    
    headers = {"X-API-Key": api_key}
    params = {"query": query, "num_web_results": 1}
    try:
        response = requests.get(
            "https://api.ydc-index.io/search",
            params=params,
            headers=headers,
            timeout=10  # seconds
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        # Log the error as needed; returning empty list here
        return []
    
    snippets = []
    for hit in data.get("hits", []):
        for snippet in hit.get("snippets", []):
            snippets.append(snippet)
    # Return only the first snippet if available
    return snippets[:1]


def process_vllm_output_for_search(text: str) -> str:
    """
    Extracts a query from the given text and returns a snippet wrapped in a tag.
    If no query is found or no snippet is returned, an empty string is returned.
    """
    query_match = re.search(r"<query>(.*?)</query>", text)
    if not query_match:
        return ""
    
    query = query_match.group(1).strip()
    snippets = get_snippets_for_query(query)
    if not snippets:
        return ""
    
    return f"<snippet>{snippets[0]}</snippet>"


@dataclass
class GenerationOutput:
    token_ids: List[int]
    text: str

@dataclass
class CompletionList:
    outputs: List[GenerationOutput]

@ray.remote
class LLMSearchRayActor:
    def __init__(self, *args, **kwargs):
        import vllm

        self.__version__ = vllm.__version__
        assert self.__version__ >= "0.4.1", "OpenRLHF only supports vLLM >= 0.4.1"

        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:
            vllm.worker.worker.Worker = WorkerWrap
        else:
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            kwargs["worker_use_ray"] = True

            if vllm.__version__ > "0.4.1":
                RayWorkerWrapperPath = vllm.executor.ray_utils
            else:
                RayWorkerWrapperPath = vllm.engine.ray_utils

            # patch for newer vllm from openrlhf:
            # https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ray/vllm_engine.py#L40
            if vllm.__version__ > "0.6.4.post1":
                # https://github.com/vllm-project/vllm/pull/10555
                kwargs["worker_cls"] = "open_instruct.vllm_utils2.WorkerWrap"
            else:
                RayWorkerWrapperPath = vllm.executor.ray_utils

                class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                    def __init__(self, *args, **kwargs) -> None:
                        kwargs["worker_module_name"] = "open_instruct.vllm_utils2"
                        kwargs["worker_class_name"] = "WorkerWrap"
                        super().__init__(*args, **kwargs)

                RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

        self.llm = vllm.LLM(*args, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        sampling_params: Any,
        prompt_token_ids: List[List[int]],
        use_tqdm: bool,
    ) -> List[CompletionList]:
        max_searches = 5

        # Initialize queries as a list of tuples: (prompt_index, current_text)
        queries: List[Tuple[int, str]] = []
        original_queries: Dict[int, str] = {}
        for i, tokens in enumerate(prompt_token_ids):
            decoded = self.tokenizer.decode(tokens)
            queries.append((i, decoded))
            original_queries[i] = decoded

        # Store finished outputs for each prompt index.
        finished_queries: Dict[int, List[str]] = {}

        # Iteratively update queries with snippet responses.
        for _ in range(max_searches):
            if not queries:
                break

            query_texts = [q for _, q in queries]
            outputs = self.llm.generate(
                sampling_params=sampling_params, prompts=query_texts, use_tqdm=use_tqdm
            )
            new_queries: List[Tuple[int, str]] = []
            for (prompt_idx, current_text), comp_list in zip(queries, outputs):
                # Process each sample output for the prompt
                for gen_output in comp_list.outputs:
                    output_text = gen_output.text
                    snippet_response = process_vllm_output_for_search(output_text)
                    updated_text = current_text + output_text + snippet_response
                    if snippet_response:
                        new_queries.append((prompt_idx, updated_text))
                    else:
                        if prompt_idx not in finished_queries:
                            finished_queries[prompt_idx] = []
                        finished_queries[prompt_idx].append(updated_text)
            queries = new_queries

        # Postprocess: remove the original prompt from finished outputs.
        generation_outputs: List[CompletionList] = []
        for i in range(len(prompt_token_ids)):
            original = original_queries.get(i, "")
            finished_texts = finished_queries.get(i, [])
            outputs_list: List[GenerationOutput] = []
            for text in finished_texts:
                # Remove only the first occurrence of the original prompt.
                final_text = text.replace(original, "", 1)
                tokens = self.tokenizer.encode(final_text)
                outputs_list.append(GenerationOutput(token_ids=tokens, text=final_text))
            generation_outputs.append(CompletionList(outputs=outputs_list))
        
        return generation_outputs
    
    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name, backend
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
            )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.stop_remote_worker_execution_loop()

        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
        else:
            return self.llm.llm_engine.model_executor._run_workers("update_weight", name, dtype, shape, empty_cache)

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > "0.4.2":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()


# Debugging code
if __name__ == "__main__":
    from vllm import SamplingParams
    from transformers import AutoTokenizer
    ray.init()
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3-8B")
    
    # Instantiate the actor. The scheduling strategy has been removed for clarity.
    actor = LLMSearchRayActor.options(
        num_cpus=4,
        num_gpus=0.48
    ).remote(
        "allenai/Llama-3.1-Tulu-3-8B",
        revision="main",
        tokenizer_revision="main",
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="bfloat16",
        seed=42,
        enable_prefix_caching=True,
        max_model_len=8192,
        gpu_memory_utilization=0.95,
    )
    
    # Create a prompt using a chat template.
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "How much money, in euros, was the surgeon held responsible for Stella Obasanjo's death ordered to pay her son? "
                    "Search the web by wrapping a query in query tags like so: <query>{query}</query> "
                    "Then, based on the snippet, provide the answer, or another query if you need."
                ),
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    
    # Encode the prompt to token ids.
    prompt_token_ids = [tokenizer.encode(prompt, add_special_tokens=False)]

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=8192,
        include_stop_str_in_output=True,
        n=1,
        stop=["</query>"],
    )
    
    # Generate output using the actor.
    result = ray.get(
        actor.generate.remote(
            sampling_params=sampling_params,  # Uses default dummy sampling params.
            prompt_token_ids=prompt_token_ids,
            use_tqdm=False,
        )
    )
    print(prompt)
    print(result)
    ray.shutdown()