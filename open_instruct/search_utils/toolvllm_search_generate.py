"""
Generate from json file.
Used for astabench

export MCP_TRANSPORT=StreamableHttpTransport
export S2_API_KEY=xxxx
python open_instruct/search_utils/toolvllm_search_generate.py \
    --json_path rubrics_v2_recomputed.json \
    --model_path /weka/oe-adapt-default/hamishi/model_checkpoints/rl_rag/rl_rag_surveyqa_samples_search_mcp_reward__1__1753332293_checkpoints/step_200 \
    --output_dir /weka/oe-adapt-default/hamishi/model_checkpoints/rl_rag/rl_rag_surveyqa_samples_search_mcp_reward__1__1753332293_checkpoints/step_200/test \
    --max_eval_samples 1000 \
    --num_docs 3 \
    --search_api_endpoint https://api.semanticscholar.org/graph/v1/snippet/search \
    --use_mcp_tool
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

from open_instruct.ground_truth_utils import f1_score
from open_instruct.search_utils.search_tool import SearchTool
from open_instruct.tool_utils.tool_vllm import ToolUseLLM
from open_instruct.tool_utils.tool_mcp import SemanticScholarSnippetSearchTool
from open_instruct.grpo_fast import launch_mcp_subprocess


SYSTEM_PROMPT = "You are a research assistant that answers questions through iterative reasoning and research.\n\nPROCESS:\n- Use <think></think> tags to show your reasoning at any point\n- Use <search>query</search> tags when you need information\n- You can alternate between thinking and searching multiple times\n- Only provide <answer></answer> tags when you have enough information for a complete response\n\nSEARCH RESULTS:\n- Results appear as: <snippet id=UNIQUE_ID>content</snippet>\n- Use exact snippet IDs for citations\n\nCITATION FORMAT:\n- In your final answer, wrap cited claims as: <cite id=SNIPPET_ID>your claim</cite>\n- Example: <cite id=a1b2c3d4>Studies show 85% effectiveness rates</cite>\n\nWORKFLOW EXAMPLE:\n<think>I need to understand the current market trends first</think>\n<search>2024 renewable energy market trends</search>\n[results provided]\n<think>Now I need specific data on solar panel efficiency</think>\n<search>latest solar panel efficiency 2024</search>\n[results provided]\n<answer>Based on my research... <cite id=abc123>claim from source</cite></answer>\n\nREQUIREMENTS:\n- Think and search iteratively until you have sufficient information\n- Only provide final answer when ready\n- Cite all claims from search results using exact snippet IDs"

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
    for i, section in enumerate(answer_sections):
        sections.append({
            "title": f"Section {i+1}",
            "text": section,
            "citations": [],
        })
        # if there are citations inside the text, extract them
        citations = re.findall(r"<cite id=\"(\w+)\">((\n|.)*?)</cite>", section)
        citations += re.findall(r"<cite id=(\w+)>((\n|.)*?)</cite>", section)
        print(citations)
        for j, citation in enumerate(citations):
            citation_id = citation[0]
            seen_citations.add(citation_id)
            # find corresponding snippet
            snippet = re.findall(r"<snippets id=" + citation_id + r">((\n|.)*?)</snippets>", thinking_section)
            if not snippet:
                print(f"Snippet {citation_id} not found in thinking section, but it was cited in the answer section. Hallucination?")
                snippet_text = ""
            else:
                snippet_text = snippet[0]
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
        for j in range(i+1, len(answer_sections)+1):
            citations = re.findall(r"<cite id=\"(\w+)\">((\n|.)*?)</cite>", "\n".join(answer_sections[i:j]))
            citations += re.findall(r"<cite id=(\w+)>((\n|.)*?)</cite>", "\n".join(answer_sections[i:j]))
            
            for citation in citations:
                citation_id = citation[0]
                if citation_id in seen_citations:
                    continue
                seen_citations.add(citation_id)
                # find corresponding snippet
                snippet = re.findall(r"<snippets id=" + citation_id + r">((\n|.)*?)</snippets>", thinking_section)
                if not snippet:
                    print(f"Snippet {citation_id} not found in thinking section, but it was cited in the answer section. Hallucination?")
                    snippet_text = ""
                else:
                    snippet_text = snippet[0]
                citation_title = citation[1]  # use the query as the title
                # add to all sections it spans
                for k in range(i, j):
                    sections[k]["citations"].append({
                        "id": citation_id,
                        "title": citation_title,
                        "snippets": [snippet_text],
                    })

    return {"section": sections}
    

def main():
    try:
        parser = argparse.ArgumentParser(description="Eval SimpleQA using the search actor.")
        parser.add_argument(
            "--json_path", type=str, help="Path to the json file."
        )
        parser.add_argument("--model_path", type=str, help="Path to the model.")
        parser.add_argument("--model_revision", type=str, default="main", help="Model revision.")
        parser.add_argument("--tokenizer_revision", type=str, default="main", help="Tokenizer revision.")
        parser.add_argument("--model_len", type=int, default=8192, help="Max model length.")
        parser.add_argument("--output_dir", type=str, default="tmp", help="Output directory.")
        parser.add_argument("--max_eval_samples", type=int, default=2000, help="Max eval samples.")
        parser.add_argument("--num_docs", type=int, default=3, help="Number of documents to retrieve.")
        parser.add_argument("--search_api_endpoint", type=str, default="http://localhost:8000", help="Search API endpoint.")
        parser.add_argument("--use_mcp_tool", action="store_true", help="Use the MCP search tool.")
        parser.add_argument("--use_astabench_format", action="store_true", help="Format citations into a format that can be used for astabench cached solver.")
        args = parser.parse_args()

        # make output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.tokenizer_revision)

        # load the json data
        with open(args.json_path, "r") as f:
            dataset = json.load(f)
        
        # surveyqa has "query", asta-bench has "question" -- check
        if "query" in dataset[0]:
            question_key = "query"
        elif "question" in dataset[0]:
            question_key = "question"
        else:
            raise ValueError(f"Question key not found in dataset: {dataset[0]}")


        ds = [{"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": data[question_key]}]} for data in dataset]
        ds = Dataset.from_list(ds)

        if args.max_eval_samples > -1 and args.max_eval_samples < len(ds):
            ds = ds.shuffle(42).select(range(args.max_eval_samples))

        prompt_token_ids = [tokenizer.apply_chat_template(data["messages"], add_generation_prompt=True) for data in ds]

        if args.use_mcp_tool:
            # rn just hardcode the mcp server command
            mcp_process = launch_mcp_subprocess(True, "fastmcp run rl-rag-mcp/rag_mcp/main.py:mcp --transport streamable-http --port 8000")
            if mcp_process is None:
                raise RuntimeError("Failed to launch MCP server subprocess")
            tool = SemanticScholarSnippetSearchTool(
                start_str="<search>",
                end_str="</search>",
            )
        else:
            tool = SearchTool(
                start_str="<search>",
                end_str="</search>",
                api_endpoint=args.search_api_endpoint,
                number_documents_to_search=args.num_docs,
            )

        # make actor.
        actor = ToolUseLLM(
            model=args.model_path,
            revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            tools={tool.end_str: tool},
            max_tool_calls=3,
            max_model_len=args.model_len,
        )
        # use greedy decoding
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.model_len,
            include_stop_str_in_output=True,
            n=1,
            stop=[tool.end_str, "</answer>"],
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
        predictions = [x.split("<answer>")[-1].replace("</answer>", "") for x in generations]

        # save predictions with sample data.
        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/predictions.jsonl", "w") as f:
            for sample, prediction, generation in zip(dataset, predictions, generations):
                f.write(json.dumps({**sample, "answer": prediction, "generation": generation}) + "\n")

        if args.use_astabench_format:
            predictions = [format_citation_data_into_sqa_format(x) for x in generations]
            with open(f"{args.output_dir}/astabench_formatted_predictions.json", "w") as f:
                json.dump(predictions, f)
    except Exception as e:
        print(f"Error: {e}")
        if mcp_process is not None:
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

def test_format_citation_data_into_sqa_format():
    from open_instruct.search_rewards.tests.formatted_test_answer import example_answer
    import pprint
    pprint.pprint(format_citation_data_into_sqa_format(example_answer))


if __name__ == "__main__":
    main()
