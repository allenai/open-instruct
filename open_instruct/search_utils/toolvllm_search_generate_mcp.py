"""
Generate from json file.
Used for astabench

python open_instruct/search_utils/toolvllm_search_generate_mcp.py \
    --json_path rubrics_v2_recomputed.json \
    --model_path 0409_rl_rag_sft_mcp__1__1757043824_checkpoints/step_200 \
    --output_dir 0409_rl_rag_sft_mcp__1__1757043824_checkpoints/step_200/astabench \
    --max_eval_samples 1000 \
    --offset 0 \
    --num_docs 3 \
    --search_api_endpoint https://api.semanticscholar.org/graph/v1/snippet/search \
    --use_astabench_format \
    --mcp_tool_names 'snippet_search,google_search,browse_webpage' \
    --mcp_server_command 'python -m rl-rag-mcp.mcp_agents.mcp_backend.main --transport http --port 8000 --host 0.0.0.0 --path /mcp'
"""

import argparse
import json
import os
import re
import time
import signal

import ray
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct.tool_utils.tool_vllm import ToolUseLLM
from open_instruct.search_utils.mcp_tools import MCPTool
from open_instruct.grpo_fast import launch_mcp_subprocess

# load system prompt from a dataset to make my life easier.
ds = load_dataset("rl-rag/rl_rag_sqa_searcharena_rubrics_web_augmented_rubrics_only_call_tool", split="train")
assert ds[0]["messages"][0]["role"] == "system"
SYSTEM_PROMPT = ds[0]["messages"][0]["content"]

def format_citation_data_into_sqa_format(response: str) -> dict:
    """
    Format a generation following the proper citation format into something that can be used for astabench cached solver.
    format is: { "section": [{"title": ..., "text": ..., "citations": [{"id": ..., "title": ...., "snippets": ["...", ...]}]}], ...}
    """
    # for now, split sections as paragraphs and then just name section 1/2/3/4/5...?
    sections = []
    # thinking section will have the snippets
    answer_section = response.split("<answer>")[-1].replace("</answer>", "")
    thinking_section = response.split("<answer>")[0]
    answer_sections = answer_section.split("\n")
    seen_citations = set()
    text_seen_so_far = ""
    # map each answer line index to its corresponding section index (or None if empty)
    answer_idx_to_section_idx = {}
    for i, section in enumerate(answer_sections):
        if section.strip() == "":
            answer_idx_to_section_idx[i] = None
            continue
        sections.append({
            "title": f"Section {i+1}",
            "text": section.strip(),
            "citations": [],
        })
        answer_idx_to_section_idx[i] = len(sections) - 1
        # if there are citations inside the text, extract them
        citations = re.findall(r"<cite id=\"(\w+)\">((\n|.)*?)</cite>", section)
        citations += re.findall(r"<cite id=(\w+)>((\n|.)*?)</cite>", section)
        for j, citation in enumerate(citations):
            citation_id = citation[0]
            seen_citations.add(citation_id)
            # find corresponding snippet (support both <snippet> and <snippets>)
            snippet = re.findall(r"<snippet? id=" + re.escape(citation_id) + r">((\n|.)*?)</snippet?>", thinking_section)
            if not snippet:
                print(f"Snippet {citation_id} not found in thinking section, but it was cited in the answer section. Hallucination?")
                snippet_text = ""
            else:
                snippet_text = snippet[0][0]
            citation_title = citation[1]  # use the query as the title
            sections[-1]["citations"].append({
                "id": citation[0],
                "title": citation_title,
                "snippets": [snippet_text],
            })
    # now, try to find citations that span multiple sections. This shouldn't really happen, but ive seen the model do it,
    # so I'm going to support it anyway. Hopefully not too costly.
    # note that since we slowly grow the sections, we should add the citation to the minimal section it spans.
    for i in range(len(answer_sections)):
        if answer_sections[i].strip() == "":
            continue
        # j is exclusive; iterate up to and including len(answer_sections)
        for j in range(i + 1, len(answer_sections) + 1):
            # j is exclusive, so check the last included section (j - 1)
            if answer_sections[j - 1].strip() == "":
                continue
            citations = re.findall(r"<cite id=\"(\w+)\">((\n|.)*?)</cite>", "\n".join(answer_sections[i:j]))
            citations += re.findall(r"<cite id=(\w+)>((\n|.)*?)</cite>", "\n".join(answer_sections[i:j]))
            
            for citation in citations:
                citation_id = citation[0]
                if citation_id in seen_citations:
                    continue
                seen_citations.add(citation_id)
                # find corresponding snippet
                snippet = re.findall(r"<snippet? id=" + re.escape(citation_id) + r">((\n|.)*?)</snippet?>", thinking_section)
                if not snippet:
                    print(f"Snippet {citation_id} not found in thinking section, but it was cited in the answer section. Hallucination?")
                    snippet_text = ""
                else:
                    snippet_text = snippet[0][0]
                citation_title = citation[1]  # use the query as the title
                # add to all sections it spans
                for k in range(i, j):
                    section_idx = answer_idx_to_section_idx.get(k)
                    if section_idx is None:
                        continue
                    sections[section_idx]["citations"].append({
                        "id": citation_id,
                        "title": citation_title,
                        "snippets": [snippet_text],
                    })

    return {"response": {"section": sections}}
    

def main():
    try:
        parser = argparse.ArgumentParser(description="Eval SimpleQA using the search actor.")
        parser.add_argument(
            "--json_path", type=str, help="Path to the json file."
        )
        parser.add_argument("--model_path", type=str, help="Path to the model.")
        parser.add_argument("--model_revision", type=str, default="main", help="Model revision.")
        parser.add_argument("--tokenizer_revision", type=str, default="main", help="Tokenizer revision.")
        parser.add_argument("--model_len", type=int, default=16384, help="Max model length.")
        parser.add_argument("--output_dir", type=str, default="tmp", help="Output directory.")
        parser.add_argument("--max_eval_samples", type=int, default=2000, help="Max eval samples.")
        parser.add_argument("--offset", type=int, default=0, help="Offset for the eval samples.")
        parser.add_argument("--num_docs", type=int, default=3, help="Number of documents to retrieve.")
        parser.add_argument("--search_api_endpoint", type=str, default="http://localhost:8000", help="Search API endpoint.")
        parser.add_argument("--use_astabench_format", action="store_true", help="Format citations into a format that can be used for astabench cached solver.")
        parser.add_argument("--dont_use_system_prompt", action="store_true", help="Don't use the system prompt.")
        parser.add_argument("--mcp_tool_names", type=str, default="semantic_scholar,serper", help="MCP tool names.")
        parser.add_argument("--mcp_server_command", type=str, default="fastmcp run rl-rag-mcp/rag_mcp/main.py:mcp --transport streamable-http --port 8008", help="MCP server command.")
        parser.add_argument("--mcp_parser_name", type=str, default="v20250824", help="MCP parser name.")
        args = parser.parse_args()

        if 'port' in args.mcp_server_command:
            os.environ["MCP_TRANSPORT_PORT"] = args.mcp_server_command.split("--port ")[1].split(" ")[0]

        # make output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.tokenizer_revision)

        # launch mcp server
        mcp_process = launch_mcp_subprocess(args.mcp_server_command, "./mcp_logs")

        # load the json data
        with open(args.json_path, "r") as f:
            dataset = json.load(f)

        original_dataset = dataset
        
        # surveyqa has "query", asta-bench has "question" -- check
        if "query" in dataset[0]:
            question_key = "query"
        elif "question" in dataset[0]:
            question_key = "question"
        elif "prompt" in dataset[0]:
            question_key = "prompt"
        else:
            raise ValueError(f"Question key not found in dataset: {dataset[0]}")

        # load mcp tools
        tool_objects = {}
        tool = MCPTool(
            mcp_tool_names=args.mcp_tool_names.split(","),
            parser_name=args.mcp_parser_name,
            number_documents_to_search=args.num_docs,
            base_url=args.search_api_endpoint,
        )
        # mcp tools can have multiple end strings.
        for end_str in tool.get_stop_strings():
            tool_objects[end_str] = tool

        if not args.dont_use_system_prompt:
            initial_message = [{"role": "system", "content": SYSTEM_PROMPT}]
        else:
            initial_message = []

        if question_key == "prompt":
            # we have a list of messages instead of just the single string.
            new_data = []
            for i, data in enumerate(dataset):
                messages = initial_message + data["prompt"]
                new_data.append({"messages": messages})
        else:
            ds = [{"messages": initial_message + [{"role": "user", "content": data[question_key]}]} for data in dataset]
        ds = Dataset.from_list(ds)

        if args.max_eval_samples > -1 and args.max_eval_samples < len(ds):
            ds = ds.select(range(args.offset, max(args.offset + args.max_eval_samples, len(ds))))
            original_dataset = original_dataset[args.offset:max(args.offset + args.max_eval_samples, len(original_dataset))]

        prompt_token_ids = [tokenizer.apply_chat_template(data["messages"], add_generation_prompt=True) for data in ds]

        # make actor.
        actor = ToolUseLLM(
            model=args.model_path,
            revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            tools=tool_objects,
            max_tool_calls=10,
            max_model_len=args.model_len,
        )
        # use greedy decoding
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=1.0,
            max_tokens=args.model_len,
            include_stop_str_in_output=True,
            n=1,
            stop=list(tool_objects.keys()),
        )
        # Generate output using the actor.
        result = actor.generate(
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
        )
        # grab text answers - for tool vllm, we have to decode the output ids.
        generations = [x.outputs[0].token_ids for x in result]
        generations = [tokenizer.decode(x, skip_special_tokens=True) for x in generations]
        # parse out answer
        predictions = []
        extracted_count = 0
        for generation in generations:
            final_answer = ""
            # first, try to find stuff between <answer> and </answer>
            answer = re.search(r"<answer>(.*?)</answer>", generation)
            if answer:
                final_answer = answer.group(1).strip()
                extracted_count += 1
            # second, anything after <answer>, and strip the tags
            answer = generation.split("<answer>")[-1].replace("</answer>", "")
            final_answer = answer.strip()
            if answer != generation.strip():                
                final_answer = answer.strip()
                extracted_count += 1
            predictions.append(final_answer)
        # stat: print extracted predictions
        print(f"Extracted {extracted_count} predictions out of {len(generations)}, thats {extracted_count / len(generations) * 100:.2f}%")
        # save predictions with sample data.
        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/predictions.jsonl", "w") as f:
            for sample, prediction, generation, original_sample in zip(ds, predictions, generations, original_dataset):
                f.write(json.dumps({**sample, **original_sample, "answer": prediction, "generation": generation}) + "\n")

        if args.use_astabench_format:
            predictions = [format_citation_data_into_sqa_format(x) for x in generations]
            samples = [{**sample, **original_sample} for sample, original_sample in zip(ds, original_dataset)]
            with open(f"{args.output_dir}/astabench_formatted_predictions.json", "w") as f:
                json.dump(samples, f)
    except Exception as e:
        print(f"Error: {e}")
        try:
            print("🧹 Cleaning up MCP server subprocess...")
            if mcp_process.poll() is None:
                os.killpg(os.getpgid(mcp_process.pid), signal.SIGTERM)
                time.sleep(2)
                if mcp_process.poll() is None:
                    os.killpg(os.getpgid(mcp_process.pid), signal.SIGKILL)
            print("✅ MCP server subprocess cleaned up")
        except (OSError, ProcessLookupError) as cleanup_error:
            print(f"Warning: Error during MCP cleanup: {cleanup_error}")
        ray.shutdown()
        os._exit(1)
        raise  # Re-raise the exception after shutdown

if __name__ == "__main__":
    main()
