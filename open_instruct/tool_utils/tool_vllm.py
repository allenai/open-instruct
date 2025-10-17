"""
vLLM wrapper for tool use. You can instantiate this just like vLLM, but also with a tool dictionary.
This makes debugging and eval fun. See the bottom of the file for examples.
"""

import copy
import os
import signal
import subprocess
import sys
import time
import types
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor  # add import for async execution
from typing import Callable, Union

import requests
from rich.console import Console
from tqdm import tqdm
from vllm import LLM, PoolingParams, PoolingRequestOutput, PromptType, RequestOutput, SamplingParams, TokensPrompt
from vllm.inputs import DataPrompt
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind

# Import base classes from tools.py to avoid duplication
from open_instruct.tool_utils.tools import MaxCallsExceededTool, PythonCodeTool, Tool


class ToolUseLLM(LLM):
    def __init__(self, tools: dict[str, Tool] = None, max_tool_calls: Union[int, dict[str, int]] = 4, *args, **kwargs):
        self.tools = tools
        # Convert max_tool_calls to a dict if it's an int
        if isinstance(max_tool_calls, int):
            self.max_tool_calls = {k: max_tool_calls for k in tools.keys()} if tools else {}
        else:
            self.max_tool_calls = max_tool_calls
        # Initialize executor and store for pending tool calls
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.pending_tool_futures = {}
        super().__init__(*args, **kwargs)

    def _validate_and_add_requests(
        self,
        prompts: PromptType | Sequence[PromptType] | DataPrompt,
        params: SamplingParams | Sequence[SamplingParams] | PoolingParams | Sequence[PoolingParams],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None,
        priority: list[int] | None = None,
    ) -> None:
        """@vwxyzjn: we keep everything the same except override the sampling params to have n=1 for `ToolUseLLM`"""
        if isinstance(prompts, (str, dict)):
            # Convert a single prompt to a list.
            prompts = [prompts]  # type: ignore[list-item]

        num_requests = len(prompts)
        if isinstance(params, Sequence) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params must be the same.")
        if isinstance(lora_request, Sequence) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request must be the same.")

        for sp in params if isinstance(params, Sequence) else (params,):
            if isinstance(sp, SamplingParams):
                # We only care about the final output
                sp.output_kind = RequestOutputKind.FINAL_ONLY

        # TOOL VLLM CHANGE: override the sampling params to have n=1
        assert not isinstance(params, Sequence), (
            "ToolUseLLM only supports one sampling param setting for all requests."
        )
        self.single_n_sampling_params = copy.deepcopy(params)
        self.single_n_sampling_params.n = 1

        # Add requests to the engine.
        for i, prompt in enumerate(prompts):
            for j in range(params.n):
                if isinstance(prompt, dict):
                    self._validate_mm_data_and_uuids(prompt.get("multi_modal_data"), prompt.get("multi_modal_uuids"))
                request_id = f"{i}-{j}"
                lora_request = lora_request[i] if isinstance(lora_request, Sequence) else lora_request
                priority = priority[i] if priority else 0
                self.llm_engine.add_request(
                    request_id, prompt, self.single_n_sampling_params, lora_request=lora_request, priority=priority
                )

    def _run_engine(
        self, *, use_tqdm: bool | Callable[..., tqdm] = True
    ) -> list[Union[RequestOutput, PoolingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"),
            )

        # Run the engine.
        outputs: list[Union[RequestOutput, PoolingRequestOutput]] = []
        total_in_toks = 0
        total_out_toks = 0
        tokenizer = self.get_tokenizer()
        num_calls = defaultdict(int)
        timeout = defaultdict(bool)
        tool_error = defaultdict(str)
        tool_output = defaultdict(str)
        tool_runtime = defaultdict(float)
        tool_called = defaultdict(bool)
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
                    tool_output_token_ids = tokenizer.encode(
                        tool_result.start_str + tool_result.output + tool_result.end_str, add_special_tokens=False
                    )
                    timeout[req_id] = tool_result.timeout
                    tool_error[req_id] += "" if tool_result.error is None else tool_result.error
                    tool_output[req_id] += tool_result.output
                    tool_runtime[req_id] += tool_result.runtime
                    tool_called[req_id] = True
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
                    remaining = self.single_n_sampling_params.max_tokens - len(masks[req_id])
                    if remaining <= 0:
                        tool_output_token_ids = []
                    elif len(tool_output_token_ids) > remaining:
                        tool_output_token_ids = tool_output_token_ids[:remaining]
                    concat_outputs[req_id].outputs[0].token_ids.extend(tool_output_token_ids)
                    concat_outputs[req_id].outputs[0].logprobs.extend(
                        [{tid: types.SimpleNamespace(logprob=0.0)} for tid in tool_output_token_ids]
                    )
                    if len(concat_outputs[req_id].outputs[0].token_ids) > self.single_n_sampling_params.max_tokens:
                        breakpoint()
                        raise ValueError(
                            f"ToolUseLLM generated more response tokens than max_tokens! "
                            f"len(concat_outputs[req_id].outputs[0].token_ids): {len(concat_outputs[req_id].outputs[0].token_ids)}"
                        )
                    masks[req_id].extend([0] * len(tool_output_token_ids))
                    new_sample_tokens = self.single_n_sampling_params.max_tokens - len(masks[req_id])
                    can_make_new_request = can_make_new_request and new_sample_tokens > 0
                    if can_make_new_request:
                        try:
                            new_sampling_params = copy.deepcopy(self.single_n_sampling_params)
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
                        assert (
                            len(output.outputs) <= 1
                        )  # because with tool calls, the (overriden) sampling_params.n == 1
                        o = output.outputs[0]
                        output_processed = False
                        if output.request_id not in concat_outputs:
                            # Ensure initial logprobs is present
                            if getattr(o, "logprobs", None) is None:
                                o.logprobs = [{tid: types.SimpleNamespace(logprob=0.0)} for tid in o.token_ids]
                            concat_outputs[output.request_id] = output
                        else:
                            # Extend token ids and corresponding logprobs for continued model output
                            concat_outputs[output.request_id].outputs[0].token_ids.extend(o.token_ids)
                            if getattr(o, "logprobs", None) is None:
                                o.logprobs = [{tid: types.SimpleNamespace(logprob=0.0)} for tid in o.token_ids]
                            concat_outputs[output.request_id].outputs[0].logprobs.extend(o.logprobs)
                            if (
                                len(concat_outputs[output.request_id].outputs[0].token_ids)
                                > self.single_n_sampling_params.max_tokens
                            ):
                                breakpoint()
                                raise ValueError(
                                    f"ToolUseLLM generated more response tokens than max_tokens! "
                                    f"len(concat_outputs[output.request_id].outputs[0].token_ids): {len(concat_outputs[output.request_id].outputs[0].token_ids)}"
                                )
                        masks[output.request_id].extend([1] * len(o.token_ids))
                        for stop_str in self.single_n_sampling_params.stop:
                            if (
                                o.text.endswith(stop_str)
                                and stop_str in self.tools
                                and num_calls[output.request_id] < self.max_tool_calls[stop_str]
                            ):
                                # Schedule tool call asynchronously
                                tool = self.tools[stop_str]
                                future = self.executor.submit(tool, o.text)
                                self.pending_tool_futures[output.request_id] = (future, o, output)
                                output_processed = True
                                break
                            elif (
                                o.text.endswith(stop_str)
                                and stop_str in self.tools
                                and num_calls[output.request_id] >= self.max_tool_calls[stop_str]
                            ):
                                # If the tool has been called too many times, we tell the model it has exceeded the limit.
                                # use a dummy tool object to keep things simple.
                                tool = MaxCallsExceededTool(start_str="<tool>", end_str="</tool>")
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
                                    total_out_toks += sum(len(stp.token_ids) for stp in output.outputs)
                                    out_spd = total_out_toks / pbar.format_dict["elapsed"]
                                    pbar.postfix = (
                                        f"est. speed input: {in_spd:.2f} toks/s, output: {out_spd:.2f} toks/s"
                                    )
                                    pbar.update(n)
                                else:
                                    pbar.update(1)
            if not self.llm_engine.has_unfinished_requests() and len(self.pending_tool_futures) == 0:
                break

        if use_tqdm:
            pbar.close()
        # Add the masks to the outputs.
        for req_id in masks:
            assert req_id in concat_outputs
            setattr(concat_outputs[req_id].outputs[0], "mask", masks[req_id])
            setattr(concat_outputs[req_id].outputs[0], "num_calls", num_calls[req_id])
            setattr(concat_outputs[req_id].outputs[0], "timeout", timeout[req_id])
            setattr(concat_outputs[req_id].outputs[0], "tool_error", tool_error[req_id])
            setattr(concat_outputs[req_id].outputs[0], "tool_output", tool_output[req_id])
            setattr(concat_outputs[req_id].outputs[0], "tool_runtime", tool_runtime[req_id])
            setattr(concat_outputs[req_id].outputs[0], "tool_called", tool_called[req_id])
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
            if len(concat_outputs[req_id].outputs[0].token_ids) > self.single_n_sampling_params.max_tokens:
                breakpoint()
                raise ValueError(
                    f"ToolUseLLM generated more response tokens than max_tokens! "
                    f"len(concat_outputs[req_id].outputs[0].token_ids): {len(concat_outputs[req_id].outputs[0].token_ids)}"
                )
            real_req_id, _ = req_id.split("-")
            if real_req_id not in merged_outputs:
                merged_outputs[real_req_id] = concat_outputs[req_id]
            else:
                merged_outputs[real_req_id].outputs.append(concat_outputs[req_id].outputs[0])
        final_outputs = sorted(
            merged_outputs.values(), key=lambda x: (int(x.request_id.split("-")[0]), int(x.request_id.split("-")[1]))
        )
        return final_outputs


if __name__ == "__main__":
    ## basic example of how to use tool_vllm.
    console = Console()
    from transformers import AutoTokenizer

    # Sample prompts.
    system_prompt = """Below is a conversation between an user and an assitant. The assistant helps with the user's tasks. When the task is completed, the assistant ends the conversation with <endoftext>. The assistant can also use a tool for multiple times. The assitant has the following tools:

1. `<code>`: Python execution service:
You could run python code by putting your code between <code> and </code> tags. For example, it could be
<code>
print("Hello, world!")
</code>
and you will get the output between the <output> and </output> tags.
"""

    console.print(f"system_prompt: {system_prompt}")
    prompts = [
        "User: Write a python program which calculates the sum of 1 3 4. Then write another separate program to calculate the product of 1 3 4.\nAssistant:",
        "User: Write a python program which prints 'Hello, Hamish!'.\nAssistant:",
    ]
    prompts = [system_prompt + "\n\n" + p for p in prompts]

    # launch the tool server (portable: uses current python, no dependency on `uv` CLI)
    tool_utils_dir = os.path.dirname(__file__)
    server_cmd = [sys.executable, "-m", "uvicorn", "tool_server:app", "--host", "127.0.0.1", "--port", "1212"]
    server_process = subprocess.Popen(
        server_cmd,
        cwd=tool_utils_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,  # Create new process group
    )

    try:
        # Wait for server readiness
        base_url = "http://127.0.0.1:1212"
        health_url = f"{base_url}/"
        start_wait = time.time()
        while True:
            if server_process.poll() is not None:
                raise RuntimeError("Tool server exited unexpectedly. Check stderr for details.")
            try:
                r = requests.get(health_url, timeout=0.5)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            if time.time() - start_wait > 30:
                raise TimeoutError("Timed out waiting for tool server to start")
            time.sleep(0.2)

        # Create a tool.
        python_code_tool = PythonCodeTool(api_endpoint=f"{base_url}/execute", start_str="<code>", end_str="</code>")
        tools = {python_code_tool.end_str: python_code_tool}
        # Create a sampling params object.
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            stop=[item.end_str for item in tools.values()] + ["<endoftext>"],
            n=3,
            max_tokens=1000,
            include_stop_str_in_output=True,
        )
        print(f"{sampling_params.n=}")
        # Create an LLM.
        model_name = "Qwen/Qwen2.5-7B"
        llm = ToolUseLLM(
            tools=tools, model=model_name, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=10000
        )

        # Tokenization generation
        from open_instruct.dataset_transformation import visualize_token_role

        tok = AutoTokenizer.from_pretrained(model_name)
        prompt_token_ids = [tok.encode(p) for p in prompts]
        prompt = [TokensPrompt(prompt_token_ids=p) for p in prompt_token_ids]
        outputs = llm.generate(prompt, sampling_params=sampling_params)
        for i, output in enumerate(outputs):
            prompt = tok.decode(output.prompt_token_ids)
            console.rule(f"Conversation {i}")
            console.rule("Prompt")
            console.print(prompt)
            for j, o in enumerate(output.outputs):
                generated_text = tok.decode(o.token_ids)
                assert len(o.mask) == len(o.token_ids)
                console.rule(f"Generated text {j}")
                console.print(generated_text)
                console.rule("Generated text w/ masks")
                visualize_token_role(o.token_ids, o.mask, tok)
                # Print tool usage details if available
                tool_called = getattr(o, "tool_called", False)
                if tool_called:
                    num_calls = getattr(o, "num_calls", 0)
                    timeout_flag = getattr(o, "timeout", False)
                    tool_err = getattr(o, "tool_error", "")
                    tool_out = getattr(o, "tool_output", "")
                    console.rule("Tool execution summary")
                    console.print(f"tool_called={tool_called}, num_calls={num_calls}, timeout={timeout_flag}")
                    if tool_err:
                        console.print(f"tool_error=\n{tool_err}")
                    if tool_out:
                        console.print(f"tool_output=\n{tool_out}")
        print(f"{sampling_params.n=}")
        print("Self-contained tool_vllm example complete.")
        # Explicitly shutdown the vLLM engine and thread pool to avoid
        # the background monitor logging an unexpected death on program exit.
        try:
            # Cleanly stop engine core processes first
            llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        try:
            # Then stop the local thread pool used for tool calls
            llm.executor.shutdown(wait=True)
        except Exception:
            pass
    finally:
        # Gracefully terminate the tool server
        try:
            os.killpg(server_process.pid, signal.SIGTERM)
            try:
                server_process.wait(timeout=5)
            except Exception:
                os.killpg(server_process.pid, signal.SIGKILL)
        except Exception:
            pass
