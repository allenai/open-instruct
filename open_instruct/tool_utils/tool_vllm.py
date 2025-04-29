"""
python open_instruct/tool_utils/tool_vllm.py
"""
from collections import defaultdict
from collections.abc import Sequence
import copy
from dataclasses import dataclass
import re
import warnings
import requests
import traceback
from vllm import LLM, SamplingParams, RequestOutput, PoolingRequestOutput, TokensPrompt, PromptType, PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.model_executor.guided_decoding.guided_fields import (GuidedDecodingRequest)
from vllm.sampling_params import RequestOutputKind
from vllm.lora.request import LoRARequest
from tqdm import tqdm
from typing import Union, Optional
from rich.console import Console
from concurrent.futures import ThreadPoolExecutor  # add import for async execution

@dataclass
class ToolOutput:
    output: str
    called: bool
    success: bool

class Tool:
    def __init__(self, start_str: str, end_str: str):
        self.start_str = start_str
        self.end_str = end_str

    def __call__(self, prompt: str) -> ToolOutput:
        raise NotImplementedError("Subclasses must implement this method")

class PythonCodeTool(Tool):
    """@vwxyzjn: I recommend using something like a FastAPI for this kind of stuff; 1) you 
    won't accidentally block the main vLLM process and 2) way easier to parallelize via load balancing."""
    
    def __init__(self, api_endpoint: str, *args, **kwargs):
        self.api_endpoint = api_endpoint
        super().__init__(*args, **kwargs)

    def __call__(self, prompt: str) -> ToolOutput:
        """
        NOTE: We avoid using `r'<tool>\s*(.*?)\s*</tool>'` because it will fail in this case
        Let's implement this in Python using the `<code>` tag to execute the code and get the result.
        </think>

        <code>
        def find_sum_of_a():
            total_sum = 0
            for n in range(100):  # Arbitrary large range for n
                for m in range(100):  # Arbitrary large range for m
                    a = 2**n * 3**m
                    if 6*n > a or 6*m > a:
                        continue
                    total_sum += a
            return total_sum

        result = find_sum_of_a()
        print(result)
        </code>

        Instead, Use negative look-behind approach to find the code block.
        """
        re_str = r'(?s)(?<!`)<tool>\s*(.*?)\s*</tool>'
        re_str = re_str.replace('<tool>', "<code>").replace('</tool>', "</code>")

        code_blocks = re.findall(re_str, prompt, re.DOTALL)
        all_outputs = []
        
        if len(code_blocks) == 0:
            return ToolOutput(output="", called=False, success=False)

        # Only execute the last code block
        code = code_blocks[-1]
        # Define timeout in seconds
        timeout_seconds = 3
        try:
            # Call the FastAPI endpoint to execute the code with client-side timeout
            response = requests.post(
                self.api_endpoint,
                json={"code": code, "timeout": timeout_seconds},  # Server-side timeout (keeping this)
                timeout=timeout_seconds  # Client-side timeout
            )
            
            # Parse the response
            result = response.json()
            
            # Handle the response based on success/failure
            if result["success"]:
                output = result["output"]
                error = result.get("error", "")
                
                # Combine output and error if both exist
                if error:
                    if output:
                        all_outputs.append(output + "\n" + error)
                    else:
                        all_outputs.append(error)
                else:
                    all_outputs.append(output)
            else:
                # Handle execution failure
                error_message = result.get("error", "Unknown error occurred")
                all_outputs.append(error_message)
                
        except requests.Timeout:
            # Handle client-side timeout specifically
            all_outputs.append(f"Timeout after {timeout_seconds} seconds")
            
        except Exception as e:
            # Capture any other exceptions that occur during the API call
            error_message = f"Error calling API: {str(e)}\n"
            error_traceback = traceback.format_exc()
            all_outputs.append(error_message + error_traceback)
        
        # Return all captured outputs as a single string
        return ToolOutput(output='\n'.join(all_outputs), called=True, success=True)



class ToolUseLLM(LLM):
    def __init__(
            self,
            tools: dict[str, Tool] = None,
            max_tool_calls: int = 4,
            *args,
            **kwargs
        ):
        self.tools = tools
        self.max_tool_calls = max_tool_calls
        # Initialize executor and store for pending tool calls
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.pending_tool_futures = {}
        super().__init__(*args, **kwargs)


    def _validate_and_add_requests(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        guided_options: Optional[GuidedDecodingRequest] = None,
        priority: Optional[list[int]] = None,
    ) -> None:
        """@vwxyzjn: we keep everything the same except override the sampling params to have n=1 for `ToolUseLLM`"""
        if guided_options is not None:
            warnings.warn(
                "guided_options_request is deprecated, use "
                "SamplingParams.guided_decoding instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(prompts, (str, dict)):
            # Convert a single prompt to a list.
            prompts = [prompts]

        num_requests = len(prompts)
        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        for sp in params if isinstance(params, list) else (params, ):
            if isinstance(sp, SamplingParams):
                self._add_guided_params(sp, guided_options)

                # We only care about the final output
                sp.output_kind = RequestOutputKind.FINAL_ONLY

        # @vwxyzjn: ToolUseLLM change 1: override the sampling params to have n=1
        assert not isinstance(params, list)
        self.original_n = params.n
        params.n = 1
        self.sampling_params = copy.deepcopy(params)
        # Add requests to the engine.
        for i, prompt in enumerate(prompts):
            for j in range(self.original_n):
                request_id = f"{i}-{j}"
                self.llm_engine.add_request(
                    request_id,
                    prompt,
                    params,
                    lora_request=lora_request[i] if isinstance(
                        lora_request, Sequence) else lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                    priority=priority[i] if priority else 0,
                )



    def _run_engine(
            self, *, use_tqdm: bool
    ) -> list[Union[RequestOutput, PoolingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )

        # Run the engine.
        outputs: list[Union[RequestOutput, PoolingRequestOutput]] = []
        total_in_toks = 0
        total_out_toks = 0
        tokenizer = self.get_tokenizer()
        num_calls = defaultdict(int)
        concat_outputs = {}  # concat multi-turn response token ids, tool token ids
        masks = defaultdict(list)
        while True:
            # @vwxyzjn: ToolUseLLM change 1: append tool output to the prompt and
            # submit another request if possible.
            # Poll pending tool futures without blocking
            dict_keys_to_delete = []
            for req_id, (future, last_o, last_output) in self.pending_tool_futures.items():
                if future.done():
                    tool_result = future.result()
                    last_prompt_token_ids = last_output.prompt_token_ids
                    last_token_ids = last_o.token_ids
                    tool_output_token_ids = tokenizer.encode("<output>\n" + tool_result.output + "</output>\n", add_special_tokens=False)
                    # Edge case 1: clip against model context length
                    prompt_and_tool_output_token = last_prompt_token_ids + last_token_ids + tool_output_token_ids
                    num_calls[req_id] += 1
                    excess = len(prompt_and_tool_output_token) - self.llm_engine.model_config.max_model_len
                    if excess > 0:
                        tool_output_token_ids = tool_output_token_ids[:-excess]
                        can_make_new_request = False
                    else:
                        can_make_new_request = True

                    # Edge case 2: clip against per-request max_tokens
                    remaining = self.sampling_params.max_tokens - len(masks[req_id])
                    if remaining <= 0:
                        tool_output_token_ids = []
                    elif len(tool_output_token_ids) > remaining:
                        tool_output_token_ids = tool_output_token_ids[:remaining]
                    concat_outputs[req_id].outputs[0].token_ids.extend(tool_output_token_ids)
                    masks[req_id].extend([0] * len(tool_output_token_ids))
                    new_sample_tokens = self.sampling_params.max_tokens - len(masks[req_id])
                    can_make_new_request = can_make_new_request and new_sample_tokens > 0
                    if can_make_new_request:
                        try:
                            new_sampling_params = copy.deepcopy(self.sampling_params)
                            new_sampling_params.max_tokens = new_sample_tokens
                            self.llm_engine.add_request(
                                req_id,
                                TokensPrompt(prompt_token_ids=prompt_and_tool_output_token),
                                new_sampling_params,
                            )
                        except Exception as e:
                            print("Error:", e)
                            print("prompt_and_tool_output_token:", prompt_and_tool_output_token)
                            print("last_prompt_token_ids:", last_prompt_token_ids)
                            print("last_token_ids:", last_token_ids)
                            print("tool_output_token_ids:", tool_output_token_ids)
                            breakpoint()
                            print("end")
                    dict_keys_to_delete.append(req_id)
            for req_id in dict_keys_to_delete:
                del self.pending_tool_futures[req_id]

            if self.llm_engine.has_unfinished_requests():
                step_outputs = self.llm_engine.step()
                for output in step_outputs:
                    if output.finished:
                        # @vwxyzjn: ToolUseLLM change 2: if the output is a tool call,
                        # we submit the tool to a thread pool and wait for the result.
                        assert len(output.outputs) <= 1 # because with tool calls, the (overriden) sampling_params.n == 1
                        o = output.outputs[0]
                        output_processed = False
                        if output.request_id not in concat_outputs:
                            concat_outputs[output.request_id] = output
                        else:
                            concat_outputs[output.request_id].outputs[0].token_ids.extend(o.token_ids)
                        masks[output.request_id].extend([1] * len(o.token_ids))
                        for stop_str in self.sampling_params.stop:
                            if o.text.endswith(stop_str) and stop_str in self.tools and num_calls[output.request_id] <= self.max_tool_calls:
                                # Schedule tool call asynchronously
                                tool = self.tools[stop_str]
                                future = self.executor.submit(tool, o.text)
                                self.pending_tool_futures[output.request_id] = (future, o, output)
                                output_processed = True
                                break
                        if not output_processed:
                            outputs.append(output)
                            if use_tqdm:
                                if isinstance(output, RequestOutput):
                                    # Calculate tokens only for RequestOutput
                                    n = len(output.outputs)
                                    assert output.prompt_token_ids is not None
                                    total_in_toks += len(output.prompt_token_ids) * n
                                    in_spd = total_in_toks / pbar.format_dict["elapsed"]
                                    total_out_toks += sum(
                                        len(stp.token_ids) for stp in output.outputs)
                                    out_spd = (total_out_toks /
                                            pbar.format_dict["elapsed"])
                                    pbar.postfix = (
                                        f"est. speed input: {in_spd:.2f} toks/s, "
                                        f"output: {out_spd:.2f} toks/s")
                                    pbar.update(n)
                                else:
                                    pbar.update(1)
            if not self.llm_engine.has_unfinished_requests() and len(self.pending_tool_futures) == 0:
                break

        print(f"number of calls per request ids", num_calls)
        # print(tokenizer.decode(concat_outputs["0"].prompt_token_ids))
        # print(tokenizer.decode(concat_outputs["0"].outputs[0].token_ids))
        if use_tqdm:
            pbar.close()
        # Add the masks to the outputs.
        for req_id in masks:
            assert req_id in concat_outputs
            setattr(concat_outputs[req_id].outputs[0], "mask", masks[req_id])
            setattr(concat_outputs[req_id].outputs[0], "num_calls", num_calls[req_id])
            if len(masks[req_id]) != len(concat_outputs[req_id].outputs[0].token_ids):
                visualize_token_role(concat_outputs[req_id].outputs[0].token_ids, masks[req_id], tokenizer)
                breakpoint()
                raise ValueError(
                    f"Mask length {len(masks[req_id])} does not match "
                    f"token IDs length {len(concat_outputs[req_id].outputs[0].token_ids)}"
                )

        # Merge n completions into the same outputs of the same prompt
        merged_outputs = {}
        for req_id in concat_outputs:
            assert (
                len(concat_outputs[req_id].outputs[0].token_ids)
                <= self.sampling_params.max_tokens
            ), "ToolUseLLM generated more response tokens than max_tokens!"
            real_req_id, _ = req_id.split("-")
            if real_req_id not in merged_outputs:
                merged_outputs[real_req_id] = concat_outputs[req_id]
            else:
                merged_outputs[real_req_id].outputs.append(concat_outputs[req_id].outputs[0])
        final_outputs = sorted(merged_outputs.values(), key=lambda x: (int(x.request_id.split("-")[0]), int(x.request_id.split("-")[1])))
        return final_outputs


if __name__ == "__main__":
    console = Console()
    from transformers import AutoTokenizer


    # Sample prompts.
    system_prompt = (
"""Below is a conversation between an user and an assitant. The assistant helps with the user's tasks. When the task is completed, the assistant ends the conversation with <endoftext>. The assistant can also use a tool for multiple times. The assitant has the following tools:

1. `<code>`: Python execution service:
You could run python code by putting your code between <code> and </code> tags. For example, it could be 
<code>
print("Hello, world!")
</code>
and you will get the output between the <output> and </output> tags.
"""
    )

    console.print(f"system_prompt: {system_prompt}")
    prompts = [
        "User: Write a python program which calculates the sum of 1 3 4. Then write another separate program to calculate the product of 1 3 4.\nAssistant:",
        "User: Write a python program which prints 'Hello, Costa!'.\nAssistant:",
    ]
    prompts = [system_prompt + "\n\n" + p for p in prompts]

    # Create a tool.
    python_code_tool = PythonCodeTool(api_endpoint="https://open-instruct-tool-server-10554368204.us-central1.run.app/execute", start_str="<code>", end_str="</code>")
    tools = {
        python_code_tool.end_str: python_code_tool,
    }
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        stop=[item.end_str for item in tools.values()] + ["<endoftext>"],
        n=3,
        max_tokens=1000,
        include_stop_str_in_output=True,
    )
    # Create an LLM.
    model_name = "Qwen/Qwen2.5-7B"
    llm = ToolUseLLM(
        tools=tools,
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=10000,
    )

    # Tokenization generation
    from open_instruct.dataset_transformation import visualize_token, visualize_token_role
    tok = AutoTokenizer.from_pretrained(model_name)
    prompt_token_ids = [tok.encode(p) for p in prompts]
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
    for i, output in enumerate(outputs):
        prompt = tok.decode(output.prompt_token_ids)
        console.rule(f"Conversation {i}")
        console.rule("Prompt")
        console.print(prompt)
        for j, o in enumerate(output.outputs):
            generated_text = tok.decode(o.token_ids)
            assert len(o.mask) == len(o.token_ids)
            console.rule(f"Generated text {j}")
            console.rule("Generated text w/ masks")
            visualize_token_role(o.token_ids, o.mask, tok)
            console.rule("Generated text")
            visualize_token(o.token_ids, tok)
    
    print("debugging tests 2 all done")
    # breakpoint()
    # More serious benchmarks

    # from datasets import load_dataset
    # tok = AutoTokenizer.from_pretrained(model_name)
    # ds = load_dataset("ai2-adapt-dev/rlvr_open_reasoner_math", split="train")
    # ds = ds.select(range(8192))
    # def process(example):
    #     messages = [{"role": "system", "content": system_prompt}] + example["messages"]
    #     example["input_ids_prompt"] = tok.apply_chat_template(messages, add_generation_prompt=True)
    #     return example
    # ds = ds.map(process, remove_columns=["messages"])

    # print("ds:", ds)
    # outputs = llm.generate(prompt_token_ids=ds["input_ids_prompt"], sampling_params=sampling_params)
    # print(f"len(outputs): {len(outputs)}")
    # print("debugging tests all done")
    # # need to handle the case the response length actually goes down overtime
    from open_instruct.dataset_transformation import get_cached_dataset_tulu
    from open_instruct.dataset_transformation import TokenizerConfig
    
    tc = TokenizerConfig(tokenizer_name_or_path=model_name, chat_template_name="r1_simple_chat_postpend_think_tools2")
    transform_fn_args = [
        {},
        {
            "max_token_length": 8192,
            "max_prompt_token_length": 2048,
        },
    ]
    train_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=["ai2-adapt-dev/rlvr_open_reasoner_math", "1.0"],
        dataset_mixer_list_splits=["train"],
        tc=tc,
        dataset_transform_fn=["rlvr_tokenize_v1", "rlvr_filter_v1"],
        transform_fn_args=transform_fn_args,
        dataset_cache_mode="local",
        hf_entity="allenai",
        dataset_local_cache_dir="/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache",
    )
    outputs = llm.generate(prompt_token_ids=train_dataset["input_ids_prompt"], sampling_params=sampling_params)
    # breakpoint()
    print("debugging tests all done")