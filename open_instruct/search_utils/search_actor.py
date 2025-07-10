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
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import ray

from open_instruct.search_utils.massive_ds import get_snippets_for_query


def process_vllm_output_for_search(text: str, number_documents_to_search: int = 10) -> str:
    """
    Extracts a query from the given text and returns a snippet wrapped in a tag.
    If no query is found or no snippet is returned, an empty string is returned.
    """
    query_match = re.search(r"<query>(.*?)</query>", text)
    if not query_match:
        return ""

    query = query_match.group(1).strip()
    print(f"Searching: {query}")
    snippets = get_snippets_for_query(query, number_of_results=number_documents_to_search)
    if not snippets:
        return "<document>Query failed.</document>"

    return f"<document>{snippets[0]}</document>"


@dataclass
class GenerationOutput:
    token_ids: List[int]
    text: str
    finish_reason: str = "stop"


@dataclass
class CompletionList:
    outputs: List[GenerationOutput]


@ray.remote
class LLMSearchRayActor:
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        noset_visible_devices = kwargs.pop("noset_visible_devices")
        self.max_output_length = kwargs.pop("max_output_len", 8192)
        self.max_searches = kwargs.pop("max_searches", 5)
        self.number_documents_to_search = kwargs.pop("number_documents_to_search", 10)
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        from vllm import LLM

        self.llm = LLM(*args, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self, sampling_params: Any, prompt_token_ids: List[List[int]], use_tqdm: bool
    ) -> List[List[GenerationOutput]]:
        max_searches = self.max_searches
        number_documents_to_search = self.number_documents_to_search

        # If num samples > 1, duplicate prompts and set sampling_params.n to 1.
        original_n = sampling_params.n
        if sampling_params.n > 1:
            new_prompt_token_ids = []
            for tokens in prompt_token_ids:
                new_prompt_token_ids.extend([tokens] * sampling_params.n)
            prompt_token_ids = new_prompt_token_ids
            sampling_params.n = 1

        # Initialize queries as a list of tuples: (index, decoded query text)
        queries: List[Tuple[int, str]] = [
            (i, self.tokenizer.decode(tokens)) for i, tokens in enumerate(prompt_token_ids)
        ]
        original_queries: Dict[int, str] = {i: q for i, q in queries}
        finished_queries: Dict[int, str] = {}

        # Track token counts for each query.
        # init with 0, we just want max output length
        query_token_counts: Dict[int, int] = {i: 0 for i, _ in enumerate(prompt_token_ids)}

        # Iteratively update queries with snippet responses.
        for _ in range(max_searches):
            if not queries:
                break

            query_texts = [q for _, q in queries]
            outputs = self.llm.generate(sampling_params=sampling_params, prompts=query_texts, use_tqdm=use_tqdm)
            updated_queries: List[Tuple[int, str]] = []
            # Process each query and update its text.
            for (idx, current_text), output in zip(queries, outputs):
                output_text = output.outputs[0].text
                output_tokens = self.tokenizer.encode(output_text)
                remaining_tokens = self.max_output_length - query_token_counts[idx]

                # Truncate output_text if it exceeds the remaining token budget.
                if len(output_tokens) > remaining_tokens:
                    truncated_output_tokens = output_tokens[:remaining_tokens]
                    truncated_output = self.tokenizer.decode(truncated_output_tokens)
                    finished_queries[idx] = current_text + truncated_output
                    query_token_counts[idx] = self.max_output_length
                    continue
                else:
                    query_token_counts[idx] += len(output_tokens)

                # Process potential snippet.
                snippet_response = process_vllm_output_for_search(
                    output_text, number_documents_to_search=number_documents_to_search
                )

                if snippet_response:
                    snippet_tokens = self.tokenizer.encode(snippet_response)
                    remaining_tokens = self.max_output_length - query_token_counts[idx]
                    if len(snippet_tokens) > remaining_tokens:
                        if remaining_tokens > 0:
                            truncated_snippet_tokens = snippet_tokens[:remaining_tokens]
                            truncated_snippet = self.tokenizer.decode(truncated_snippet_tokens)
                            query_token_counts[idx] += len(truncated_snippet_tokens)
                            finished_queries[idx] = current_text + output_text + truncated_snippet
                        else:
                            finished_queries[idx] = current_text + output_text
                    else:
                        query_token_counts[idx] += len(snippet_tokens)
                        updated_queries.append((idx, current_text + output_text + snippet_response))
                else:
                    finished_queries[idx] = current_text + output_text

            queries = updated_queries

        # Finalize any queries that haven't been marked finished ---
        for idx, current_text in queries:
            if idx not in finished_queries:
                finished_queries[idx] = current_text

        # Postprocess: remove the original prompt from finished outputs.
        final_texts: List[str] = []
        for i in range(len(prompt_token_ids)):
            full_text = finished_queries.get(i, "")
            original = original_queries.get(i, "")
            # Remove only the first occurrence of the original prompt.
            final_texts.append(full_text.replace(original, "", 1))

        # Encode final outputs and wrap in GenerationOutput objects.
        # Truncate to max_context_length.
        encoded_outputs = [
            self.tokenizer.encode(text, max_length=self.max_output_length, truncation=True) for text in final_texts
        ]
        # Re-decode with max length.
        final_texts = [self.tokenizer.decode(tokens) for tokens in encoded_outputs]
        # Recreate the outputs based on the original `n` value.
        generation_outputs = []

        # just hardcoding things as stop finish for now... TODO: also add length finish reason.
        for i in range(0, len(encoded_outputs), original_n):
            start, stop = i, i + original_n
            generation_outputs.append(
                CompletionList(
                    [
                        GenerationOutput(token_ids=tokens, text=text, finish_reason="stop")
                        for tokens, text in zip(encoded_outputs[start:stop], final_texts[start:stop])
                    ]
                )
            )
        # set the sampling params back to original
        sampling_params.n = original_n

        return generation_outputs

    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray=False
    ):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()


# Debugging code
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    ray.init()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3-8B")

    # Instantiate the actor. The scheduling strategy has been removed for clarity.
    actor = LLMSearchRayActor.options(num_cpus=4, num_gpus=0.48).remote(
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
                    "How much money, in euros, was the surgeon held responsible for Stella Obasanjo's death ordered to pay her son? "
                    "Search the web by wrapping a query in query tags like so: <query>{query}</query> "
                    "Then, based on the document, provide the answer, or another query if you need."
                ),
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    # Encode the prompt to token ids.
    prompt_token_ids = [tokenizer.encode(prompt, add_special_tokens=False)]

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=8192, include_stop_str_in_output=True, n=1, stop=["</query>"]
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
