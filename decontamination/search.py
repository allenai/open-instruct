
import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from elasticsearch import Elasticsearch

parser = argparse.ArgumentParser()
parser.add_argument("--es_url", type=str, default="http://localhost:9200")
parser.add_argument("--dataset", type=str)
parser.add_argument("--subset", type=str)
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--field", type=str, nargs="+")
parser.add_argument("--limit", type=int, help="Limit the number of eval instances")
parser.add_argument("--index_name", type=str, required=True)
parser.add_argument("--index_type", type=str, choices=["text", "vector"], default="text")
parser.add_argument("--model", type=str, default="nvidia/NV-Embed-v2")
parser.add_argument("--max_batch_tokens", type=int, default=10000, help="Maximum number of tokens per batch if the `index_type` is `vector`.")
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

eval_sets = [
    # (dataset, subset, split, fields, limit)
    # Dev evals
    ("cais/mmlu", "all", "test", ["question"], None),
    ("openai/openai_humaneval", None, "test", ["prompt"], None),
    ("openai/gsm8k", "main", "test", ["question"], None),
    ("ucinlp/drop", None, "validation", ["passage", "question"], None),
    ("lighteval/MATH", "all", "test", ["problem"], None),
    ("google/IFEval", None, "train", ["prompt"], None),
    ("akariasai/PopQA", None, "test", ["subj", "prop", "obj"], None),
    ("tatsu-lab/alpaca_eval", None, "eval", ["instruction"], None),
    ("lukaemon/bbh", None, "test", ["input"], None),
    ("truthfulqa/truthful_qa", "generation", "validation", ["question"], None),
    # Test evals
    ("TIGER-Lab/MMLU-Pro", None, "test", ["question"], None),
    ("Idavidrein/gpqa", "gpqa_extended", "train", ["Question"], None),
    ("lighteval/agi_eval_en", None, "train", ["passage", "question"], None),
    ("bigcode/bigcodebench", None, "v0.1.2", ["instruct_prompt"], None),
    ("deepmind/math_dataset", None, "test", ["question"], 50),
] if args.dataset is None else [
    (args.dataset, args.subset, args.split, args.field, args.limit)
]

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

es = Elasticsearch(
    args.es_url,
    basic_auth=("elastic", os.environ["ELASTIC_PASSWORD"]),
)
if args.index_type == "vector":
    import torch
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    model.cuda()
    print(f"Loaded {args.model} on device:{model.device}")
    if torch.cuda.device_count() > 1:
        print("Found multiple gpus. Will use data parallel.")
        for module_key, module in model._modules.items():
            model._modules[module_key] = torch.nn.DataParallel(module)

mean_match_scores = {}
for dataset, subset, split, fields, limit in eval_sets:
    print(f"Querying {args.index_name} for {dataset}.")
    try:
        query_dataset = list(load_dataset(dataset, subset, split=split))[:limit]
    except ValueError:
        query_dataset = []
        if args.subset is None:
            # Dataset has multiple subsets. We want to concatenate all of them.
            from datasets import get_dataset_config_names
            for subset in get_dataset_config_names(dataset):
                query_dataset.extend(list(load_dataset(dataset, subset, split=split))[:limit])
        else:
            raise

    match_scores = []
    output_data = []
    if args.index_type == "text":
        for i, datum in tqdm(enumerate(query_dataset)):
            query_strings = [datum[field] for field in fields]
            if any([s is None for s in query_strings]):
                continue
            search_output = es.search(
                index=args.index_name,
                search_type="query_then_fetch",
                rest_total_hits_as_int=True,
                query={
                    "bool": {
                        "filter": [
                            {
                                "match_phrase": {
                                    "text": query_str
                                }
                            }
                            for query_str in query_strings
                        ] 
                    }
                }
            )
            num_hits = search_output["hits"]["total"]
            match_scores.append(1 if num_hits > 0 else 0)
            output_data.append(
                {
                    "query": query_strings,
                    "num_hits": num_hits,
                }
            )

    else:
        batch_inputs = []
        batch_size = 0
        max_seq_tokens = 0
        for i, datum in tqdm(enumerate(query_dataset)):
            batch_inputs.append(" ".join([datum[field] for field in fields]))
            max_seq_tokens = max(max_seq_tokens, len(tokenizer.tokenize(batch_inputs[-1])))
            batch_size += 1
            if (max_seq_tokens * batch_size >= args.max_batch_tokens) or (i == len(query_dataset) - 1):
                question_embeddings = model.encode(batch_inputs)
                question_embeddings = torch.nn.functional.normalize(question_embeddings, p=2, dim=1)
                for query, embedding in zip(batch_inputs, question_embeddings):
                    sem_search = es.search(
                        index=args.index_name,
                        knn={"field": "vector", "query_vector": embedding.cpu().numpy(), "k": 10, "num_candidates": 100},
                    )
                    results = sem_search["hits"]["hits"][:5]
                    match_scores.append(results[0]["_score"])
                    output_results = [
                        {
                            "index": r["_index"],
                            "score": r["_score"],
                            "id": r["_id"],
                            "text": r["_source"]["text"],
                            "original_id": r["_source"]["original_id"],
                        }
                        for r in results
                    ]
                    output_data.append(
                        {
                            "query": query,
                            "results": output_results,
                        }
                    )
                batch_inputs = []
                max_seq_tokens = 0
                batch_size = 0

    mean_match_score = sum(match_scores) / len(match_scores)
    print(f"\tMean match score: {mean_match_score}")
    mean_match_scores[dataset] = mean_match_score
    output_filename = os.path.join(args.output_dir, f"{args.index_name}_{dataset.split('/')[-1]}.jsonl")
    with open(output_filename, "w") as outfile:
        for datum in output_data:
            print(json.dumps(datum), file=outfile)

output_file = os.path.join(args.output_dir, "contamination_results.tsv")
print(f"TSV file with all results: {output_file}")
with open(output_file, "w") as outfile:
    print("\t" + "\t".join(ev[0] for ev in eval_sets), file=outfile)
    print(args.index_name + "\t" + "\t".join([f"{mean_match_scores[ev[0]]:.4f}" for ev in eval_sets]), file=outfile)