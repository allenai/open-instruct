"""
python open_instruct/tool_utils/tool_vllm1.py
"""

import re
import traceback
from concurrent.futures import ThreadPoolExecutor  # add import for async execution
from dataclasses import dataclass
from typing import Union

import requests
from rich.console import Console
from tqdm import tqdm
from vllm import LLM, PoolingRequestOutput, RequestOutput, SamplingParams


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

    def __call__(self, prompt: str) -> ToolOutput:
        # Find Python code blocks using regex
        re_str = r"<tool>\s*(.*?)\s*</tool>"
        re_str = re_str.replace("<tool>", self.start_str).replace("</tool>", self.end_str)

        code_blocks = re.findall(re_str, prompt, re.DOTALL)
        all_outputs = []

        if len(code_blocks) == 0:
            return ToolOutput(output="", called=False, success=False)

        # Only execute the last code block
        code = code_blocks[-1]
        try:
            # Call the FastAPI endpoint to execute the code
            response = requests.post("http://phobos-cs-aus-453:1212/execute", json={"code": code, "timeout": 10})

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

        except Exception as e:
            # Capture any exceptions that occur during the API call
            error_message = f"Error calling API: {str(e)}\n"
            error_traceback = traceback.format_exc()
            all_outputs.append(error_message + error_traceback)

        # Return all captured outputs as a single string
        return ToolOutput(output="\n".join(all_outputs), called=True, success=True)


class ToolUseLLM(LLM):
    def __init__(self, tools: dict[str, Tool] = None, sampling_params: SamplingParams = None, *args, **kwargs):
        self.tools = tools
        self.sampling_params = sampling_params
        # Initialize executor and store for pending tool calls
        self.executor = ThreadPoolExecutor(max_workers=max(1, len(tools)))
        self.pending_tool_futures = {}
        super().__init__(*args, **kwargs)

    def _run_engine(self, *, use_tqdm: bool) -> list[Union[RequestOutput, PoolingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, " f"output: {0:.2f} toks/s"),
            )

        # Run the engine.
        outputs: list[Union[RequestOutput, PoolingRequestOutput]] = []
        total_in_toks = 0
        total_out_toks = 0
        already_executed_tool_request_ids = set()
        while self.llm_engine.has_unfinished_requests() or len(self.pending_tool_futures) != 0:
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    # @vwxyzjn: this is the only change: if the output is a tool call,
                    # we need to add a new request
                    # one limitation though: we cannot generate n completions per prompt;
                    # otherwise we can only deal with one tool call per prompt.
                    remaining_outputs = []
                    for o in output.outputs:
                        output_processed = False
                        for stop_str in self.sampling_params.stop:
                            if (
                                o.text.endswith(stop_str)
                                and output.request_id not in already_executed_tool_request_ids
                            ):
                                # Schedule tool call asynchronously
                                tool = self.tools[stop_str]
                                future = self.executor.submit(tool, o.text)
                                # Save context to poll later
                                self.pending_tool_futures[output.request_id] = (future, output.prompt, o.text)
                                output_processed = True
                                break
                        if not output_processed:
                            remaining_outputs.append(o)
                    output.outputs = remaining_outputs
                    if len(remaining_outputs) > 0:
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
                                    f"est. speed input: {in_spd:.2f} toks/s, " f"output: {out_spd:.2f} toks/s"
                                )
                                pbar.update(n)
                            else:
                                pbar.update(1)

            # Poll pending tool futures without blocking
            for req_id, (future, prompt_text, tool_text) in list(self.pending_tool_futures.items()):
                if future.done():
                    tool_result = future.result()
                    # Inject tool output when ready
                    self.llm_engine.add_request(
                        req_id,
                        prompt_text + tool_text + "\n<output>\n" + tool_result.output + "</output>\n",
                        self.sampling_params,
                    )
                    already_executed_tool_request_ids.add(req_id)
                    del self.pending_tool_futures[req_id]

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))


if __name__ == "__main__":
    console = Console()

    # Create a tool.
    python_code_tool = PythonCodeTool("<code>", "</code>")
    tools = {
        python_code_tool.end_str: python_code_tool,
    }

    # Sample prompts.
    system_prompt = """Below is a conversation between an user and an assitant. The assistant helps with the user's tasks.
When the task is completed, the assistant ends the conversation with <endoftext>.

The assitant has the following tools:

1. `<code>`: Python execution service.

You could run python code by putting your code between <code> and </code> tags. For example, it could be 
<code>
print("Hello, world!")
</code>
and you will get the output between the <output> and </output> tags.
"""
    console.print(f"system_prompt: {system_prompt}")
    prompts = [
        "User: Write a python program which calculates the sum of 1 3 4.\nAssistant:",
        "User: Write a python program which prints 'Hello, Costa!'.\nAssistant:",
    ]
    prompts = [system_prompt + "\n\n" + p for p in prompts]

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        stop=[item.end_str for item in tools.values()] + ["<endoftext>"],
        max_tokens=1000,
        include_stop_str_in_output=True,
    )

    # Create an LLM.
    llm = ToolUseLLM(
        tools=tools,
        sampling_params=sampling_params,
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        console.rule(f"Conversation {i}")
        console.rule("Prompt")
        console.print(prompt)
        console.rule("Generated text")
        console.print(generated_text)

    breakpoint()
    print("all done")
