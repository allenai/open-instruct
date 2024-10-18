
import os
import json
import yaml
import argparse
from collections import defaultdict
from tqdm import tqdm
import spacy
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from elasticsearch import Elasticsearch

SPACY_MODEL = spacy.load("en_core_web_lg")


def prepare_embedding_model(model_name):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.cuda()
    print(f"Loaded {model_name} on device:{model.device}")
    if torch.cuda.device_count() > 1:
        print("Found multiple gpus. Will use data parallel.")
        for module_key, module in model._modules.items():
            model._modules[module_key] = torch.nn.DataParallel(module)
    
    return model, tokenizer


def get_ngram_mapping(string: str, n: int):
    doc = SPACY_MODEL(string)
    ngram_docs = [doc[i:i+n] for i in range(len(doc) - n + 1)]
    # Mapping from the ngram to the indices of tokens in the original string.
    mapping = {ngram_doc.text: [token.i for token in ngram_doc] for ngram_doc in ngram_docs}
    return mapping


def exact_match(es, index_name, query_dataset, fields):
    match_scores = []
    output_data = []
    for datum in tqdm(query_dataset):
        query_strings = [datum[field] for field in fields]
        if any([s is None for s in query_strings]):
            continue
        search_output = es.search(
            index=index_name,
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
        if num_hits > 0:
            match_scores.append(1)
            output_data.append(
                {
                    "query": query_strings,
                    "num_hits": num_hits
                    }
            )
        else:
            match_scores.append(0)
    return match_scores, output_data


def ngram_match(es, index_name, query_dataset, fields, ngram_size):
    match_scores = []
    output_data = []
    for datum in tqdm(query_dataset):
        query_strings = [datum[field] for field in fields]
        if any([s is None for s in query_strings]):
            continue
        query_string_match_scores = []
        query_string_match_tokens = defaultdict(list)
        matching_doc_ids = set()
        doc_id_text_mapping = {}
        match_info = None
        for query_string in query_strings:
            # We compute the match score for each query string for ngram matches as follows:
            # For each token in the query string, we retrieve the training documents that contain ngrams from the query string
            # the token belongs to. Then we compute the match score as the ratio of the tokens in the query string that match that training document.
            query_string_tokens = [d.text for d in SPACY_MODEL(query_string)]
            query_string_length = len(query_string_tokens)
            ngram_mapping = get_ngram_mapping(query_string, ngram_size)
            train_doc_matches = defaultdict(set)
            for ngram, tokens in ngram_mapping.items():
                search_output = es.search(
                    index=index_name,
                    search_type="query_then_fetch",
                    rest_total_hits_as_int=True,
                    query={
                        "bool": {
                            "filter": [
                                {
                                    "match_phrase": {
                                        "text": ngram
                                    }
                                }
                            ] 
                        }
                    }
                )
                train_doc_ids = [h["_id"] for h in search_output["hits"]["hits"]]
                train_texts = [h["_source"]["text"] for h in search_output["hits"]["hits"]]
                for doc_id, text in zip(train_doc_ids, train_texts):
                    train_doc_matches[doc_id].update(tokens)
                    matching_doc_ids.add(doc_id)
                    doc_id_text_mapping[doc_id] = text

            query_string_match_scores.append({doc_id: len(matching_tokens) / query_string_length for doc_id, matching_tokens in train_doc_matches.items()})
            for doc_id, matching_tokens in train_doc_matches.items():
                query_string_match_tokens[doc_id].append([query_string_tokens[t] for t in matching_tokens])
        
        if matching_doc_ids:
            # Averaging the match scores of training documents over all query strings.
            aggregated_match_scores = {doc_id: sum([x.get(doc_id, 0.0) for x in query_string_match_scores]) / len(query_string_match_scores) for doc_id in matching_doc_ids}
            largest_match_doc_id, match_score = sorted(aggregated_match_scores.items(), key=lambda x: x[1], reverse=True)[0]
            match_scores.append(match_score)
            match_info = {
                "query": query_strings,
                "largest_match": {
                    "doc_id": largest_match_doc_id,
                    "text": doc_id_text_mapping[largest_match_doc_id],
                    "matching_tokens": query_string_match_tokens[largest_match_doc_id]
                },
                "score": match_score,
            }
            output_data.append(match_info)
        else:
            match_scores.append(0)
    return match_scores, output_data     


def vector_match(es, index_name, query_dataset, fields, model, tokenizer, max_batch_tokens):
    match_scores = []
    output_data = []
    batch_inputs = []
    batch_size = 0
    max_seq_tokens = 0
    for i, datum in tqdm(enumerate(query_dataset)):
        batch_inputs.append(" ".join([datum[field] for field in fields]))
        max_seq_tokens = max(max_seq_tokens, len(tokenizer.tokenize(batch_inputs[-1])))
        batch_size += 1
        if (max_seq_tokens * batch_size >= max_batch_tokens) or (i == len(query_dataset) - 1):
            question_embeddings = model.encode(batch_inputs)
            question_embeddings = torch.nn.functional.normalize(question_embeddings, p=2, dim=1)
            for query, embedding in zip(batch_inputs, question_embeddings):
                sem_search = es.search(
                    index=index_name,
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
    return match_scores, output_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--es_url", type=str, default="http://localhost:9200")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--field", type=str, nargs="+")
    parser.add_argument("--limit", type=int, help="Limit the number of eval instances")
    parser.add_argument("--index_names", type=str, nargs="+")
    parser.add_argument("--dataset_mixer_config", type=str, help="Path to a train config file in yml format with a `dataset_mixer` field.")
    parser.add_argument("--index_type", type=str, choices=["text", "vector"], default="text")
    parser.add_argument("--ngram_size", type=int, help="If `index_type` is `text`, will use n-gram matches of this size if this field is set. Default is full match.")
    parser.add_argument("--match_threshold", type=float, help="For ngram and vector matching, transform match scores to 0/1 based on this threshold.")
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

    if args.dataset_mixer_config is not None:
        print(f"Reading from dataset mixer info from train config: {args.dataset_mixer_config}")
        train_config = yaml.safe_load(open(args.dataset_mixer_config))
        dataset_names = list(train_config["dataset_mixer"].keys())
        index_names = [d.replace("/", "_") + f"_{args.index_type}" for d in dataset_names]
        print(f"Config has {len(dataset_names)} datasets. Looking for corresponding indexes: {index_names}")
    elif args.index_names is not None:
        index_names = args.index_names
    else:
        raise RuntimeError("Specify index_names or provide a train config file with dataset mixer info.")

    all_index_match_scores = []
    for index_name in index_names:
        mean_match_scores = {}
        for dataset, subset, split, fields, limit in eval_sets:
            print(f"Querying {index_name} for {dataset}.")
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

            if args.index_type == "text": 
                if args.ngram_size is None:
                    match_scores, output_data = exact_match(es, index_name, query_dataset, fields)
                else:
                    match_scores, output_data = ngram_match(es, index_name, query_dataset, fields, args.ngram_size)
            else:
                model, tokenizer = prepare_embedding_model(args.model)
                match_scores, output_data = vector_match(es, index_name, query_dataset, fields, model, tokenizer, args.max_batch_tokens)

            
            if args.match_threshold is not None:
                match_scores = [1 if score > args.match_threshold else 0 for score in match_scores]
            mean_match_score = sum(match_scores) / len(match_scores)
            print(f"\tMean match score: {mean_match_score}")
            mean_match_scores[dataset] = mean_match_score
            output_filename = os.path.join(args.output_dir, f"{index_name}_{dataset.split('/')[-1]}.jsonl")
            with open(output_filename, "w") as outfile:
                for datum in output_data:
                    print(json.dumps(datum), file=outfile)
        all_index_match_scores.append(mean_match_scores)

    output_file = os.path.join(args.output_dir, "contamination_results.tsv")
    print(f"TSV file with all results: {output_file}")
    with open(output_file, "w") as outfile:
        print("\t" + "\t".join(ev[0] for ev in eval_sets), file=outfile)
        for index_name, mean_match_scores in zip(index_names, all_index_match_scores):
            print(index_name + "\t" + "\t".join([f"{mean_match_scores[ev[0]]:.4f}" for ev in eval_sets]), file=outfile)


if __name__ == "__main__":
    main()
