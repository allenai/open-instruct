"""
python open_instruct/tool_utils/tool_vllm.py
"""

import copy
import json
import re
import time
import traceback
import warnings
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor  # add import for async execution
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import requests
from rich.console import Console
from tqdm import tqdm
from vllm import LLM, PoolingParams, PoolingRequestOutput, PromptType, RequestOutput, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding.guided_fields import GuidedDecodingRequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import RequestOutputKind

from open_instruct.tool_utils.swe_tool_parser import Argument as SWEArgument
from open_instruct.tool_utils.swe_tool_parser import Command as SWECommand
from open_instruct.tool_utils.swe_tool_parser import FunctionCallingParser


@dataclass
class ToolOutput:
    output: str
    called: bool
    error: str
    timeout: bool
    runtime: float
    terminate: bool = False  # If True, terminate generation immediately
    start_str: str = "<|im_start|>user\n<tool_response>\n"
    end_str: str = "</tool_response>\n<|im_end|>\n<|im_start|>\n"


class Tool:
    def __init__(self, start_str: str, end_str: str):
        self.start_str = start_str
        self.end_str = end_str

    def __call__(self, prompt: str) -> ToolOutput:
        raise NotImplementedError("Subclasses must implement this method")


class MaxCallsExceededTool(Tool):
    def __call__(self, prompt: str) -> ToolOutput:
        return ToolOutput(
            output="Max tool calls exceeded. Terminating generation.",
            called=True,
            error="",
            timeout=False,
            runtime=0,
            terminate=True
        )


class PythonCodeTool(Tool):
    """@vwxyzjn: I recommend using something like a FastAPI for this kind of stuff; 1) you
    won't accidentally block the main vLLM process and 2) way easier to parallelize via load balancing."""

    def __init__(self, api_endpoint: str, *args, **kwargs):
        self.api_endpoint = api_endpoint
        super().__init__(*args, **kwargs)

    def __call__(self, prompt: str) -> ToolOutput:
        r"""
        NOTE: We avoid using `r'<tool>\s*(.*?)\s*</tool>'` because it will fail in this case  # noqa: W605
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
        re_str = r"(?s)(?<!`)<tool>\s*(.*?)\s*</tool>"
        re_str = re_str.replace("<tool>", "<code>").replace("</tool>", "</code>")

        code_blocks = re.findall(re_str, prompt, re.DOTALL)
        all_outputs = []
        timeout = False
        error = ""
        if len(code_blocks) == 0:
            return ToolOutput(output="", called=False, error="", timeout=False, runtime=0)

        # Only execute the last code block
        code = code_blocks[-1]

        # Define timeout in seconds
        timeout_seconds = 3
        start_time = time.time()
        try:
            # Call the FastAPI endpoint to execute the code with client-side timeout
            response = requests.post(
                self.api_endpoint,
                json={"code": code, "timeout": timeout_seconds},  # Server-side timeout (keeping this)
                timeout=timeout_seconds,  # Client-side timeout
            )

            # Parse the response
            result = response.json()

            # Process the API response
            output = result["output"]
            error = result.get("error") or ""

            all_outputs.append(output)
            if len(error) > 0:
                all_outputs.append("\n" + error)

        except requests.Timeout:
            # Handle client-side timeout specifically
            all_outputs.append(f"Timeout after {timeout_seconds} seconds")
            timeout = True

        except Exception as e:
            # Capture any other exceptions that occur during the API call
            error_message = f"Error calling API: {str(e)}\n"
            error_traceback = traceback.format_exc()
            all_outputs.append(error_message + error_traceback)

        # Return all captured outputs as a single string
        return ToolOutput(
            output="\n".join(all_outputs), called=True, error=error, timeout=timeout, runtime=time.time() - start_time
        )

class CodeAgentTool(Tool):
    def __init__(self, api_endpoint: str, repo_name: str = None, *args, **kwargs):
        # api_endpoint is the full endpoint for view_file. We'll derive base URL for /run_bash
        self.api_endpoint = api_endpoint
        # Compute base URL (strip trailing path like /view_file)
        try:
            if api_endpoint.endswith("/view_file"):
                self.base_api = api_endpoint[: -len("/view_file")]
            else:
                self.base_api = api_endpoint.rsplit("/", 1)[0]
        except Exception:
            self.base_api = api_endpoint
        self.repo_name = repo_name
        super().__init__(*args, **kwargs)

    def __call__(self, prompt: str) -> ToolOutput:
        """
        Use the code agent tool to view files of a repo.
        The prompt will contain tool calls with file paths to view.
        Expected formats:
        <tool_call>
        {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/starlette/config.py", "view_range": [121, 138], "repo_name": "encode/starlette"}}
        </tool_call>
        or
        <tool_call>
        {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/starlette/config.py", "repo_name": "encode/starlette"}}
        </tool_call>
        The repo_name should be provided in the tool call. If not provided, will fallback to "testbed".
        """

        # Extract optional hidden per-request tool context appended by the engine
        # Pattern: <!--tool_context:{...}-->
        ctx_match = re.search(r"<!--tool_context:(.*?)-->", prompt, re.DOTALL)
        tool_ctx = {}
        if ctx_match:
            try:
                tool_ctx = json.loads(ctx_match.group(1))
            except Exception:
                tool_ctx = {}

        # Find tool calls in the prompt
        tool_call_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        tool_calls = re.findall(tool_call_pattern, prompt, re.DOTALL)
        # Fallback: support XML-style self-closing tags like
        # <str_replace_editor command="view" path="/testbed/file.py" />
        if not tool_calls:
            xml_calls = re.findall(r"<([a-zA-Z_][\w]*)\s+([^/>]*?)\s*/>", prompt)
            # Convert to pseudo JSON strings for unified processing below
            for name, attrs in xml_calls:
                kvs = dict(re.findall(r"([a-zA-Z_][\w]*)=\"([^\"]*)\"", attrs))
                tool_calls.append(json.dumps({"name": name, "arguments": kvs}))

        if not tool_calls:
            return ToolOutput(output="", called=False, error="", timeout=False, runtime=0)

        all_outputs = []
        error = ""
        start_time = time.time()
        timeout = False

        # Prepare parser and command schemas (bash + str_replace_editor)
        parser = FunctionCallingParser()
        bash_cmd_schema = SWECommand(
            name="bash",
            docstring="runs the given command directly in bash",
            arguments=[
                SWEArgument(name="command", type="string", description="The bash command to execute.", required=True),
                SWEArgument(name="path", type="string", description="Optional working dir", required=False),
                SWEArgument(name="cwd", type="string", description="Optional working dir", required=False),
                SWEArgument(name="repo_name", type="string", description="Optional repo name", required=False),
                SWEArgument(name="base_commit", type="string", description="Optional base commit", required=False),
                SWEArgument(
                    name="patches",
                    type="array",
                    description="Optional patches",
                    required=False,
                    items={"type": "string"},
                ),
            ],
        )
        str_replace_editor_schema = SWECommand(
            name="str_replace_editor",
            docstring=(
                "Custom tool for viewing/editing files. If path is a file, view displays cat -n. "
                "If path is a directory, view lists non-hidden files/dirs up to 2 levels."
            ),
            arguments=[
                SWEArgument(
                    name="command",
                    type="string",
                    description="The command to run",
                    required=True,
                    enum=["view", "create", "str_replace", "insert", "undo_edit"],
                ),
                SWEArgument(
                    name="path", type="string", description="Absolute path to file or directory", required=True
                ),
                SWEArgument(
                    name="view_range",
                    type="array",
                    description="[start,end] when viewing a file",
                    required=False,
                    items={"type": "integer"},
                ),
                SWEArgument(name="file_text", type="string", description="File contents for create", required=False),
                SWEArgument(name="old_str", type="string", description="Old string for replace", required=False),
                SWEArgument(
                    name="new_str", type="string", description="New string for replace/insert", required=False
                ),
                SWEArgument(
                    name="insert_line", type="integer", description="Insertion line (0-indexed)", required=False
                ),
                SWEArgument(name="repo_name", type="string", description="Optional repo name", required=False),
                SWEArgument(name="base_commit", type="string", description="Optional base commit", required=False),
                SWEArgument(
                    name="patches",
                    type="array",
                    description="Optional patches",
                    required=False,
                    items={"type": "string"},
                ),
            ],
        )
        submit_schema = SWECommand(
            name="submit", docstring="Signal that the task is complete and submit the solution.", arguments=[]
        )

        for tool_call_str in tool_calls:
            try:
                # Parse the JSON tool call
                tool_call = json.loads(tool_call_str)

                # Build a LiteLLM-style wrapper and validate with FunctionCallingParser
                model_response = {
                    "message": prompt,
                    "tool_calls": [
                        {"function": {"name": tool_call.get("name"), "arguments": tool_call.get("arguments", {})}}
                    ],
                }
                name = tool_call.get("name", "")
                if name == "bash":
                    _thought, _action = parser(model_response, commands=[bash_cmd_schema], strict=True)
                elif name == "submit":
                    _thought, _action = parser(model_response, commands=[submit_schema], strict=True)
                else:
                    # default to str_replace_editor schema
                    _thought, _action = parser(model_response, commands=[str_replace_editor_schema], strict=True)

                args = tool_call.get("arguments", {})
                # Normalize key variants for bash
                if name == "bash" and "cmd" in args and "command" not in args:
                    args["command"] = args["cmd"]
                command = args.get("command")

                # Extract common arguments
                path = args.get("path", "")
                view_range = args.get("view_range", None)
                patches = args.get("patches", None)
                base_commit = args.get("base_commit", None)
                bash_cmd = args.get("command") or args.get("cmd")
                cwd = args.get("cwd")

                # Keep the path as-is - the API will handle normalization
                # We don't strip /testbed anymore since API expects testbed/... paths

                # Extract repo_name from the tool call arguments or fallback to context
                repo_name = args.get("repo_name") or tool_ctx.get("repo_name")

                # Fallback to instance variable if not in tool call
                if not repo_name:
                    repo_name = self.repo_name

                # For instances from sample data, extract repo name from extra_fields if available
                if not repo_name and "extra_fields" in tool_call.get("arguments", {}):
                    repo_name = tool_call["arguments"]["extra_fields"].get("repo_name")

                # Do not force a misleading default. If still missing,
                # let the server attempt inference based on paths/cwd.

                # If not provided in the call, fallback to context for base_commit and patches
                if base_commit is None:
                    base_commit = tool_ctx.get("base_commit")
                if patches is None:
                    patches = tool_ctx.get("patches")

                timeout_seconds = 60
                if name == "bash":
                    if not bash_cmd:
                        raise ValueError("Missing 'command' for bash function call")

                    # Retry logic for API gateway errors
                    max_retries = 2
                    response = None
                    for attempt in range(max_retries):
                        try:
                            response = requests.post(
                                f"{self.base_api}/run_bash",
                                json={
                                    "repo_name": repo_name,
                                    "cmd": bash_cmd,
                                    "cwd": path or cwd,
                                    "base_commit": base_commit,
                                    "patches": patches,
                                    "timeout_seconds": timeout_seconds,
                                },
                                timeout=timeout_seconds,
                            )
                            if response.status_code in [502, 503, 504]:
                                if attempt < max_retries - 1:
                                    time.sleep(1)  # Brief pause before retry
                                    continue
                                else:
                                    # Return a cleaner error message instead of HTML
                                    content = f"OBSERVATION:\nAPI error (status {response.status_code}): Gateway timeout or service unavailable. Please try again.\n"
                                    all_outputs.append(content)
                                    response = None  # Mark as failed
                            break
                        except requests.exceptions.RequestException as e:
                            if attempt < max_retries - 1:
                                time.sleep(1)
                                continue
                            else:
                                content = f"OBSERVATION:\nAPI connection error: {str(e)}\n"
                                all_outputs.append(content)
                                response = None  # Mark as failed
                elif name == "submit":
                    # Submit should terminate generation immediately
                    content = "OBSERVATION:\nSubmission received. Task marked as complete.\n"
                    all_outputs.append(content)
                    # Set terminate flag to stop generation
                    return ToolOutput(
                        output="\n\n".join(all_outputs) if all_outputs else content,
                        called=True,
                        error="",
                        timeout=False,
                        runtime=time.time() - start_time,
                        terminate=True,  # Signal to terminate generation
                    )
                else:
                    # str_replace_editor commands routed to /edit_file (including view)
                    payload = {
                        "repo_name": repo_name,
                        "command": command,
                        "path": path,
                        "view_range": view_range,
                        "file_text": args.get("file_text"),
                        "old_str": args.get("old_str"),
                        "new_str": args.get("new_str"),
                        "insert_line": args.get("insert_line"),
                        "base_commit": base_commit,
                        "patches": patches,
                    }
                    response = requests.post(f"{self.base_api}/edit_file", json=payload, timeout=timeout_seconds)

                # Only process response if we have one
                if response is not None:
                    if response.status_code == 200:
                        result = response.json()
                        content = result.get("content", "")
                        all_outputs.append(content)
                    else:
                        error_msg = f"API error (status {response.status_code}): {response.text}"
                        all_outputs.append(error_msg)
                        error = error_msg

            except requests.Timeout:
                all_outputs.append("Timeout viewing file")
                timeout = True
                break
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON in tool call: {e}"
                all_outputs.append(error_msg)
                error = error_msg
            except Exception as e:
                error_msg = f"Error processing tool call: {str(e)}"
                all_outputs.append(error_msg)
                error = error_msg

        runtime = time.time() - start_time
        called = len(tool_calls) > 0

        return ToolOutput(
            output="\n\n".join(all_outputs) if all_outputs else "",
            called=called,
            error=error,
            timeout=timeout,
            runtime=runtime,
        )

class ToolUseLLM(LLM):
    def __init__(self, tools: dict[str, Tool] = None, max_tool_calls: Union[int, dict[str, int]] = 4, *args, **kwargs):
        self.tools = tools
        # Convert max_tool_calls to a dict if it's an int
        if isinstance(max_tool_calls, int):
            self.max_tool_calls = {tool.end_str: max_tool_calls for tool in tools.values()} if tools else {}
        else:
            self.max_tool_calls = max_tool_calls
        # Initialize executor and store for pending tool calls
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.pending_tool_futures = {}
        self._tool_contexts: Optional[list[Optional[str]]] = None
        super().__init__(*args, **kwargs)

    def set_tool_contexts(self, tool_contexts: Optional[list[Optional[str]]]):
        self._tool_contexts = tool_contexts

    def _validate_and_add_requests(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams, Sequence[PoolingParams]],
        *,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        guided_options: Optional[GuidedDecodingRequest] = None,
        priority: Optional[list[int]] = None,
    ) -> None:
        """@vwxyzjn: we keep everything the same except override the sampling params to have n=1 for `ToolUseLLM`"""
        if guided_options is not None:
            warnings.warn(
                "guided_options_request is deprecated, use SamplingParams.guided_decoding instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(prompts, (str, dict)):
            # Convert a single prompt to a list.
            prompts = [prompts]

        num_requests = len(prompts)
        if isinstance(params, Sequence) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params must be the same.")
        if isinstance(lora_request, Sequence) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request must be the same.")

        for sp in params if isinstance(params, Sequence) else (params,):
            if isinstance(sp, SamplingParams):
                self._add_guided_params(sp, guided_options)

                # We only care about the final output
                sp.output_kind = RequestOutputKind.FINAL_ONLY

        # Add requests to the engine.
        it = prompts
        if use_tqdm:
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            it = tqdm_func(it, desc="Adding requests")

        # @vwxyzjn: ToolUseLLM change 1: override the sampling params to have n=1
        # for now, don't allow list of params
        assert not isinstance(params, list)
        self.single_n_sampling_params = copy.deepcopy(params)
        self.single_n_sampling_params.n = 1

        for i, prompt in enumerate(it):
            for j in range(params.n):
                request_id = f"{i}-{j}"
                self.llm_engine.add_request(
                    request_id,
                    prompt,
                    self.single_n_sampling_params,
                    tokenization_kwargs=tokenization_kwargs,
                    lora_request=lora_request[i] if isinstance(lora_request, Sequence) else lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                    priority=priority[i] if priority else 0,
                )

    def _run_engine(self, *, use_tqdm: bool) -> list[Union[RequestOutput, PoolingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
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
                       "<|im_start|>user\n<tool_response>\n" + tool_result.output + "</tool_response>\n<|im_end|>\n<|im_start|>\n", add_special_tokens=False
                    )
                    # If the tool requests termination, append a stop string and prevent further requests
                    terminate_generation = getattr(tool_result, "terminate", False)
                    if terminate_generation:
                        try:
                            stop_tokens = tokenizer.encode("<endoftext>", add_special_tokens=False)
                        except Exception:
                            stop_tokens = []
                        tool_output_token_ids = tool_output_token_ids + stop_tokens
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
                    if len(concat_outputs[req_id].outputs[0].token_ids) > self.single_n_sampling_params.max_tokens:
                        breakpoint()
                        raise ValueError(
                            f"ToolUseLLM generated more response tokens than max_tokens! "
                            f"len(concat_outputs[req_id].outputs[0].token_ids): {len(concat_outputs[req_id].outputs[0].token_ids)}"
                        )
                    masks[req_id].extend([0] * len(tool_output_token_ids))
                    new_sample_tokens = self.single_n_sampling_params.max_tokens - len(masks[req_id])
                    can_make_new_request = can_make_new_request and new_sample_tokens > 0
                    if terminate_generation:
                        can_make_new_request = False
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
                            concat_outputs[output.request_id] = output
                        else:
                            concat_outputs[output.request_id].outputs[0].token_ids.extend(o.token_ids)
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
                                and num_calls[output.request_id] <= self.max_tool_calls[stop_str]
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
                                and num_calls[output.request_id] > self.max_tool_calls[stop_str]
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
        "User: Write a python program which prints 'Hello, Costa!'.\nAssistant:",
    ]
    prompts = [system_prompt + "\n\n" + p for p in prompts]

    # Create a tool.
    python_code_tool = PythonCodeTool(api_endpoint="http://localhost:1212", start_str="<code>", end_str="</code>")
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
            # console.rule("Generated text")
            # visualize_token(o.token_ids, tok)
    print(f"{sampling_params.n=}")
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
    from open_instruct.dataset_transformation import TokenizerConfig, get_cached_dataset_tulu

    tc = TokenizerConfig(tokenizer_name_or_path=model_name, chat_template_name="r1_simple_chat_postpend_think_tools7")
    transform_fn_args = [{}, {"max_token_length": 8192, "max_prompt_token_length": 2048}]
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
    outputs = llm.generate(prompt_token_ids=train_dataset["input_ids_prompt"][:30], sampling_params=sampling_params)
    # calculate the percentage of timeout
    timeouts = [o for output in outputs for o in output.outputs if o.timeout]
    print(f"Timeout percentage: {len(timeouts) / (len(outputs) * sampling_params.n)}")
    empty_outputs = [o for output in outputs for o in output.outputs if len(o.tool_output) == 0 and o.tool_called]
    print(f"Empty output percentage: {len(empty_outputs) / (len(outputs) * sampling_params.n)}")
    errors = [o for output in outputs for o in output.outputs if len(o.tool_error) > 0]
    print(f"Error percentage: {len(errors) / (len(outputs) * sampling_params.n)}")
    tool_called = [o for output in outputs for o in output.outputs if o.tool_called]
    print(f"Tool called percentage: {len(tool_called) / (len(outputs) * sampling_params.n)}")
    tool_runtime = [o for output in outputs for o in output.outputs if o.tool_runtime > 0]
    print(f"Tool runtime > 0 percentage: {len(tool_runtime) / (len(outputs) * sampling_params.n)}")
    # print(tok.decode(empty_outputs[0].token_ids))

    print_samples = True
    if print_samples:
        for i, output in enumerate(outputs):
            prompt = tok.decode(output.prompt_token_ids)
            console.rule(f"Conversation {i}")
            console.rule("Prompt")
            console.print(prompt)
            console.rule("Ground truth")
            console.print(train_dataset[i]["ground_truth"])
            for j, o in enumerate(output.outputs):
                generated_text = tok.decode(o.token_ids)
                assert len(o.mask) == len(o.token_ids)
                console.rule(f"Generated text {j}")
                console.rule("Generated text w/ masks")
                visualize_token_role(o.token_ids, o.mask, tok)
                # console.rule("Generated text")
                # visualize_token(o.token_ids, tok)
            breakpoint()

    # breakpoint()
    print("debugging tests all done")
