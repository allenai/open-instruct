"""
python open_instruct/tool_utils/tool_vllm.py
"""

import re
import traceback
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
        while self.llm_engine.has_unfinished_requests():
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
                                and not output.request_id in already_executed_tool_request_ids
                            ):
                                tool = self.tools[stop_str]
                                tool_result = tool(o.text)
                                self.llm_engine.add_request(
                                    output.request_id,  # @vwxyzjn: if you want to use n completions per prompt, at least change here.
                                    output.prompt
                                    + o.text
                                    + "\nHere is the execution result of the tool call:\n<output>\n"
                                    + tool_result.output
                                    + "\n</output>\n",
                                    self.sampling_params,
                                )
                                output_processed = True
                                already_executed_tool_request_ids.add(output.request_id)
                                break
                        if not output_processed:
                            remaining_outputs.append(o)
                    output.outputs = remaining_outputs
                    if len(remaining_outputs) > 0:
                        # @vwxyzjn: rest of the code is the same as the original code
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

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))


if __name__ == "__main__":
    console = Console()

    # Create a tool.
    python_code_tool = PythonCodeTool("```python\n", "```\n")
    tools = {
        python_code_tool.end_str: python_code_tool,
    }

    # Sample prompts.
    system_prompt = (
        # f"Please put your code between {python_code_tool.start_str} and {python_code_tool.end_str} tags."
        ""
    )
    console.print(f"system_prompt: {system_prompt}")
    prompts = [
        "Write a python program which calculates the sum of 1 3 4.",
        "Write a python program which prints 'Hello, world!'.",
    ]
    prompts = [system_prompt + "\n\n" + p for p in prompts]

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        stop=[item.end_str for item in tools.values()],
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
