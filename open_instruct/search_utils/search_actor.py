# Taken and modified from https://github.com/huggingface/trl
# Copyright 2024 The AllenAI Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import requests
import ray
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

from open_instruct.vllm_utils2 import WorkerWrap
from open_instruct.search_utils.s2 import get_snippets_for_query


def process_vllm_output_for_search(text: str) -> str:
    """
    Extracts a query from the given text and returns a snippet wrapped in a tag.
    If no query is found or no snippet is returned, an empty string is returned.
    """
    query_match = re.search(r"<query>(.*?)</query>", text)
    if not query_match:
        return ""
    
    query = query_match.group(1).strip()
    print(f"Searching: {query}")
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
        
        # Set default max context length if not provided
        self.max_context_length = kwargs.pop("max_context_length", None)
        # If not explicitly set, use max_model_len as the default
        if self.max_context_length is None and "max_model_len" in kwargs:
            self.max_context_length = kwargs["max_model_len"]
        # Final fallback to a reasonable default
        if self.max_context_length is None:
            self.max_context_length = 8192

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
    ) -> List[List[GenerationOutput]]:
        max_searches = 5

        # if num samples > 1, remove from sampling params and instead duplicate prompts.
        original_n = sampling_params.n
        if sampling_params.n > 1:
            new_prompt_token_ids = []
            for tokens in prompt_token_ids:
                new_prompt_token_ids.extend([tokens] * sampling_params.n)
            prompt_token_ids = new_prompt_token_ids
            sampling_params.n = 1

        # Initialize queries as a list of tuples: (index, decoded query text)
        queries: List[Tuple[int, str]] = [
            (i, self.tokenizer.decode(tokens))
            for i, tokens in enumerate(prompt_token_ids)
        ]
        original_queries: Dict[int, str] = {i: q for i, q in queries}
        finished_queries: Dict[int, str] = {}
        
        # Track token counts for each query
        query_token_counts: Dict[int, int] = {i: len(tokens) for i, tokens in enumerate(prompt_token_ids)}

        # Iteratively update queries with snippet responses.
        for _ in range(max_searches):
            if not queries:
                break

            query_texts = [q for _, q in queries]
            outputs = self.llm.generate(
                sampling_params=sampling_params, prompts=query_texts, use_tqdm=use_tqdm
            )
            updated_queries: List[Tuple[int, str]] = []
            for (idx, current_text), output in zip(queries, outputs):
                # Assume each output has at least one result.
                output_text = output.outputs[0].text
                
                # Count tokens in the output
                output_tokens = self.tokenizer.encode(output_text)
                query_token_counts[idx] += len(output_tokens)
                
                # Check if we've exceeded the max context length
                if query_token_counts[idx] >= self.max_context_length:
                    # We've exceeded the limit, mark as finished
                    finished_queries[idx] = current_text + output_text
                    continue
                
                # Process potential snippet
                snippet_response = process_vllm_output_for_search(output_text)
                
                # If there's a snippet, check if we need to truncate it
                if snippet_response:
                    snippet_tokens = self.tokenizer.encode(snippet_response)
                    remaining_tokens = self.max_context_length - query_token_counts[idx]
                    
                    if len(snippet_tokens) > remaining_tokens:
                        # Need to truncate the snippet to fit
                        if remaining_tokens > 0:
                            # Truncate to the remaining token count
                            truncated_snippet_tokens = snippet_tokens[:remaining_tokens]
                            truncated_snippet = self.tokenizer.decode(truncated_snippet_tokens)
                            # We leave the truncated snippet as is, without closing the tag
                            
                            # Update token count with truncated snippet
                            query_token_counts[idx] += len(truncated_snippet_tokens)
                            
                            # Mark as finished since we've hit the limit
                            finished_queries[idx] = current_text + output_text + truncated_snippet
                        else:
                            # No room for snippet at all
                            finished_queries[idx] = current_text + output_text
                    else:
                        # Snippet fits, add it to the count
                        query_token_counts[idx] += len(snippet_tokens)
                        # Continue with search if we're still under the limit
                        updated_queries.append((idx, current_text + output_text + snippet_response))
                else:
                    # No snippet, mark as finished
                    finished_queries[idx] = current_text + output_text
            
            queries = updated_queries

        # Postprocess: remove the original prompt from finished outputs.
        final_texts: List[str] = []
        for i in range(len(prompt_token_ids)):
            full_text = finished_queries.get(i, "")
            original = original_queries.get(i, "")
            # Remove only the first occurrence of the original prompt.
            final_texts.append(full_text.replace(original, "", 1))
        
        # Encode final outputs and wrap in GenerationOutput objects.
        encoded_outputs = [self.tokenizer.encode(text) for text in final_texts]
        # recreate the outputs based on the original `n` value
        generation_outputs = []
        for i in range(0, len(encoded_outputs), original_n):
            start, stop = i, i + original_n
            generation_outputs.append(
                CompletionList([
                    GenerationOutput(token_ids=tokens, text=text)
                    for tokens, text in zip(encoded_outputs[start:stop], final_texts[start:stop])
                ])
            )

        print("Final outputs:", generation_outputs)
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
        max_model_len=8192,  # This will be used as default max_context_length if not explicitly set
        max_context_length=4096,  # Explicitly set a custom max context length
        gpu_memory_utilization=0.95,
    )
    
    # Create a prompt using a chat template.
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "How much money, in euros, was the surgeon held responsible for Stella Obasanjo\'s death ordered to pay her son? "
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
