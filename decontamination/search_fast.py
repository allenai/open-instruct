import argparse
import itertools
import json
import os
from collections import defaultdict
from functools import partial
from multiprocessing import get_context

import spacy
import torch
import yaml
from datasets import Dataset, load_dataset
from elasticsearch import Elasticsearch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ---------------------------
# Lazy globals for workers
# ---------------------------
_ES = None
_SPACY = None
_MODEL = None
_TOKENIZER = None


def get_es(es_url, password):
    global _ES
    if _ES is None:
        _ES = Elasticsearch(es_url, basic_auth=("elastic", password))
    return _ES


def get_spacy():
    # IMPORTANT: same model as your original code (no behavior change)
    global _SPACY
    if _SPACY is None:
        _SPACY = spacy.load("en_core_web_lg")
    return _SPACY


def prepare_embedding_model(model_name):
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        if torch.cuda.device_count() > 1:
            for module_key, module in model._modules.items():
                model._modules[module_key] = torch.nn.DataParallel(module)
        _MODEL, _TOKENIZER = model, tokenizer
        print(f"Loaded {model_name} on device:{next(model.parameters()).device}")
    return _MODEL, _TOKENIZER


def get_ngram_mapping(string: str, n: int):
    sp = get_spacy()
    doc = sp(string or "")
    if n <= 0 or len(doc) == 0:
        return {}
    ngram_docs = [doc[i:i + n] for i in range(len(doc) - n + 1)]
    mapping = {ngram_doc.text: [token.i for token in ngram_doc] for ngram_doc in ngram_docs}
    return mapping


# ---------------------------
# Iter utils (no full materialization)
# ---------------------------
def iter_split(dataset, subset, split, limit, need_trust=False):
    ds = load_dataset(dataset, subset, split=split, trust_remote_code=need_trust)
    return itertools.islice(ds, 0, limit) if limit else ds


# ---------------------------
# Worker tasks (TEXT index)
# ---------------------------
def _exact_task(datum, *, es_url, es_password, index_name, fields, search_size):
    es = get_es(es_url, es_password)
    query_strings = [datum.get(f) for f in fields]
    if any(s is None for s in query_strings):
        return {"query": query_strings, "num_hits": 0}, [], 0
    so = es.search(
        index=index_name,
        search_type="query_then_fetch",
        rest_total_hits_as_int=True,
        size=search_size,
        query={
            "bool": {
                "filter": [{"match_phrase": {"text": qs}} for qs in query_strings]
            }
        }
    )
    num_hits = so["hits"]["total"]
    if num_hits > 0:
        # Preserve original schema: include full _source docs
        train_docs = [d["_source"] for d in so["hits"]["hits"]]
        ids = list({td["original_id"] for td in train_docs})
        rec = {
            "query": query_strings,
            "num_hits": num_hits,
            "train_docs": train_docs,
        }
        return rec, ids, 1
    else:
        return {"query": query_strings, "num_hits": 0}, [], 0


def _ngram_task(datum, *, es_url, es_password, index_name, fields, ngram_size, search_size):
    es = get_es(es_url, es_password)
    sp = get_spacy()

    query_strings = [datum.get(f) for f in fields]
    if any(s is None for s in query_strings):
        return {"query": query_strings, "matches": [], "score": 0.0}, {}, 0.0

    query_string_match_scores = []
    query_string_match_tokens = defaultdict(list)
    matching_doc_ids = set()
    doc_id_source_mapping = {}

    for query_string in query_strings:
        query_string_tokens = [d.text for d in sp(query_string or "")]
        query_string_length = max(1, len(query_string_tokens))
        ngram_mapping = get_ngram_mapping(query_string or "", ngram_size)
        train_doc_matches = defaultdict(set)

        for ngram, tokens in ngram_mapping.items():
            so = es.search(
                index=index_name,
                search_type="query_then_fetch",
                rest_total_hits_as_int=True,
                size=search_size,
                query={
                    "bool": {
                        "filter": [{"match_phrase": {"text": ngram}}]
                    }
                }
            )
            for hit_info in so["hits"]["hits"]:
                doc_id = hit_info["_id"]
                doc = hit_info["_source"]
                train_doc_matches[doc_id].update(tokens)
                matching_doc_ids.add(doc_id)
                doc_id_source_mapping[doc_id] = doc

        # per-query normalized scores
        qs_scores = {doc_id: len(matching_tokens) / query_string_length
                     for doc_id, matching_tokens in train_doc_matches.items()}
        query_string_match_scores.append(qs_scores)

        for doc_id, matching_tokens in train_doc_matches.items():
            query_string_match_tokens[doc_id].append([query_string_tokens[t] for t in matching_tokens])

    if matching_doc_ids:
        aggregated_match_scores = {
            doc_id: sum([x.get(doc_id, 0.0) for x in query_string_match_scores]) / len(query_strings)
            for doc_id in matching_doc_ids
        }
        sorted_matches = sorted(aggregated_match_scores.items(), key=lambda x: x[1], reverse=True)
        match_info = []
        per_doc_scores = {}
        for doc_id, score in sorted_matches:
            match_info.append(
                {
                    "doc_id": doc_id,
                    "source": doc_id_source_mapping[doc_id],
                    "matching_tokens": query_string_match_tokens[doc_id],
                    "score": score,
                }
            )
            per_doc_scores[doc_id_source_mapping[doc_id]["original_id"]] = max(
                per_doc_scores.get(doc_id_source_mapping[doc_id]["original_id"], 0.0),
                score
            )
        match_score = sorted_matches[0][1]
        rec = {
            "query": query_strings,
            "matches": match_info,
            "score": match_score,
        }
        return rec, per_doc_scores, match_score
    else:
        return {"query": query_strings, "matches": [], "score": 0.0}, {}, 0.0


# ---------------------------
# Vector path (single-process, streaming)
# ---------------------------
def vector_match_streaming(es, index_name, query_dataset, fields, model, tokenizer, max_batch_tokens, search_size, outfile, match_threshold, contaminated_ids, mean_acc):
    batch_inputs = []
    batch_size = 0
    max_seq_tokens = 0

    def flush():
        nonlocal batch_inputs, batch_size, max_seq_tokens
        if not batch_inputs:
            return
        question_embeddings = model.encode(batch_inputs)
        question_embeddings = torch.nn.functional.normalize(question_embeddings, p=2, dim=1)
        for query, embedding in zip(batch_inputs, question_embeddings):
            sem_search = es.search(
                index=index_name,
                knn={"field": "vector", "query_vector": embedding.detach().cpu().numpy(), "k": search_size, "num_candidates": 10 * search_size},
            )
            results = sem_search["hits"]["hits"][:5]
            if results:
                top_score = results[0]["_score"]
                if match_threshold is not None:
                    mean_acc["sum"] += 1 if top_score > match_threshold else 0
                else:
                    mean_acc["sum"] += top_score
                mean_acc["n"] += 1
                out_results = []
                per_doc_scores = {}
                for result in results:
                    out_results.append(
                        {
                            "index": result["_index"],
                            "score": result["_score"],
                            "id": result["_id"],
                            "text": result["_source"]["text"],
                            "original_id": result["_source"]["original_id"],
                        }
                    )
                    _id = result["_source"]["original_id"]
                    s = result["_score"]
                    if match_threshold is not None and s > match_threshold:
                        contaminated_ids.add(_id)
                    else:
                        # still track max if needed later; but original code only updates with threshold
                        pass
                json.dump({"query": query, "results": out_results}, outfile)
                outfile.write("\n")
            else:
                if match_threshold is not None:
                    mean_acc["sum"] += 0
                else:
                    mean_acc["sum"] += 0.0
                mean_acc["n"] += 1
                json.dump({"query": query, "results": []}, outfile)
                outfile.write("\n")
        batch_inputs = []
        batch_size = 0
        max_seq_tokens = 0

    for i, datum in tqdm(enumerate(query_dataset)):
        query_strings = [datum[field] for field in fields]
        query_text = " ".join(query_strings)
        max_seq_tokens = max(max_seq_tokens, len(tokenizer.tokenize(query_text)))
        batch_inputs.append(query_text)
        batch_size += 1
        if (max_seq_tokens * batch_size >= max_batch_tokens):
            flush()
    flush()


# ---------------------------
# Cache recovery (unchanged from your behavior)
# ---------------------------
def load_cached_results(path, index_type,
                        match_threshold=None, is_exact=False):
    match_scores = []
    contaminated_ids = set()
    max_score_per_train_doc = defaultdict(float)

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "num_hits" in d:                           # exact-match
                score_raw = 1 if d["num_hits"] > 0 else 0
            elif "score" in d:                            # n-gram
                score_raw = d["score"]
            else:                                         # vector
                score_raw = d["results"][0]["score"] if d.get("results") else 0.0

            match_scores.append(
                score_raw if match_threshold is None
                else 1 if score_raw > match_threshold else 0
            )

            if "train_docs" in d:                         # exact-match
                for td in d["train_docs"]:
                    contaminated_ids.add(td["original_id"])
            elif "matches" in d:                          # n-gram
                for m in d["matches"]:
                    _id, s = m["source"]["original_id"], m["score"]
                    max_score_per_train_doc[_id] = max(max_score_per_train_doc[_id], s)
            else:                                         # vector
                for r in d.get("results", []):
                    _id, s = r["original_id"], r["score"]
                    max_score_per_train_doc[_id] = max(max_score_per_train_doc[_id], s)

    if match_threshold is not None:
        contaminated_ids.update(
            _id for _id, s in max_score_per_train_doc.items()
            if s > match_threshold
        )
    mean = sum(match_scores) / len(match_scores) if match_scores else 0.0
    return mean, contaminated_ids


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--es_url", type=str, default="http://localhost:9200")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--field", type=str, nargs="+")
    parser.add_argument("--limit", type=int, help="Limit the number of eval instances")
    parser.add_argument("--train_dataset_names", type=str, nargs="+")
    parser.add_argument("--dataset_mixer_config", type=str, help="Path to a train config file in yml format with a `dataset_mixer` field.")
    parser.add_argument("--index_type", type=str, choices=["text", "vector"], default="text")
    parser.add_argument("--search_size", type=int, default=100)
    parser.add_argument("--ngram_size", type=int)
    parser.add_argument("--match_threshold", type=float)
    parser.add_argument("--model", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument("--max_batch_tokens", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--decontaminate", action="store_true")

    # Optional multiprocessing for TEXT mode
    parser.add_argument("--workers", type=int, default=1, help="Workers for TEXT mode (exact/ngram). Default 1.")
    parser.add_argument("--mp_start_method", type=str, default="spawn", choices=["spawn", "fork", "forkserver"], help="Multiprocessing start method (macOS default: spawn).")

    args = parser.parse_args()

    eval_sets = [
        # (dataset, subset, split, fields, limit)
        # Olmo 3 New Dev Evals
        ("livecodebench/code_generation_lite", None, "test", ["question_content"], None),  # v3 is dev, v6 is test, load full for full decontam. Needs trust_remote_code=True
        ("HuggingFaceH4/aime_2024", None, "train", ["problem"], None),
        ("yentinglin/aime_2025", "default", "train", ["problem"], None),
        ("allenai/ZebraLogicBench-private", "grid_mode", "test", ["puzzle"], None),
        ("allenai/multilingual_mbpp", None, "test", ["text"], None),
        # Tulu 3 Dev evals
        ("cais/mmlu", "all", "test", ["question"], None),
        ("openai/openai_humaneval", None, "test", ["prompt"], None),
        ("openai/gsm8k", "main", "test", ["question"], None),
        ("EleutherAI/hendrycks_math", None, "test", ["problem"], None),
        ("google/IFEval", None, "train", ["prompt"], None),
        ("akariasai/PopQA", None, "test", ["subj", "prop", "obj"], None),
        ("tatsu-lab/alpaca_eval", None, "eval", ["instruction"], None),
        ("lukaemon/bbh", None, "test", ["input"], None),
        ("truthfulqa/truthful_qa", "generation", "validation", ["question"], None),
        ("allenai/wildguardmix", "wildguardtest", "test", ["prompt"], None),
        ("allenai/wildjailbreak", "eval", "train", ["adversarial"], None),
        ("allenai/tulu-3-trustllm-jailbreaktrigger-eval", None, "test", ["prompt"], None),
        ("allenai/tulu-3-harmbench-eval", None, "test", ["Behavior"], None),
        ("allenai/tulu-3-do-anything-now-eval", None, "test", ["prompt"], None),
        ("hamishivi/toxigen-prompts", None, "train", ["prompt"], None),
        ("walledai/XSTest", None, "test", ["prompt"], None),
        ("basicv8vc/SimpleQA", None, "test", ["problem"], None),
        ("hamishivi/omega-all-test-prompts", None, "train", ["prompt"], None),
        # Test evals
        ("TIGER-Lab/MMLU-Pro", None, "test", ["question"], None),
        ("Idavidrein/gpqa", "gpqa_extended", "train", ["Question"], None),
        ("lighteval/agi_eval_en", None, "train", ["passage", "question"], None),
        ("bigcode/bigcodebench", None, "v0.1.2", ["instruct_prompt"], None),
        ("deepmind/math_dataset", None, "test", ["question"], 50),
        ("allenai/IFBench_test", None, "train", ["prompt"], None),
        ("cais/hle", None, "test", ["question"], None),
        ("m-a-p/SuperGPQA", None, "train", ["question"], None),
        ("BBEH/bbeh", None, "train", ["input"], None),
        # not currently really evaluated but decontam anyway.
        ("AI-MO/aimo-validation-amc", None, "train", ["problem"], None),
        ("math-ai/olympiadbench", None, "test", ["question"], None),
        ("cruxeval-org/cruxeval", None, "test", ["code"], None),
        ("hamishivi/codeeditorbench_prompts", None, "train", ["prompt"], None),
        ("cais/wmdp", None, "test", ["question"], None),
        ("ByteDance-Seed/BeyondAIME", None, "test", ["problem"], None),
        ("hamishivi/strong_reject_data", None, "train", ["jailbroken_prompt"], None),
        ("hamishivi/bbq_prompts", None, "train", ["prompt"], None),
    ] if args.dataset is None else [
        (args.dataset, args.subset, args.split, args.field, args.limit)
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    # Save password env early (workers will read it)
    es_password = os.environ["ELASTIC_PASSWORD"]

    # Determine datasets/indexes
    if args.dataset_mixer_config is not None:
        print(f"Reading from dataset mixer info from train config: {args.dataset_mixer_config}")
        train_config = yaml.safe_load(open(args.dataset_mixer_config))
        dataset_names = list(train_config["dataset_mixer"].keys())
        index_names = [d.replace("/", "_").lower() + f"_{args.index_type}" for d in dataset_names]
        print(f"Config has {len(dataset_names)} datasets. Looking for corresponding indexes: {index_names}")
    elif args.train_dataset_names is not None:
        dataset_names = args.train_dataset_names
        index_names = [d.replace("/", "_").lower() + f"_{args.index_type}" for d in dataset_names]
    else:
        raise RuntimeError("Specify train_dataset_names or provide a train config file with dataset mixer info.")

    all_index_match_scores = []
    all_index_contaminated_ids = []

    # Configure multiprocessing context
    ctx = get_context(args.mp_start_method)

    for index_name in index_names:
        mean_match_scores = {}
        contaminated_ids = set()

        for dataset, subset, split, fields, limit in eval_sets:
            print(f"Querying {index_name} for {dataset}.")
            output_filename = os.path.join(args.output_dir, f"{index_name}_{dataset.split('/')[-1]}.jsonl")

            if os.path.exists(output_filename):
                print(f"[cache] {output_filename} exists – reusing.")
                mean, cached_ids = load_cached_results(output_filename, args.index_type, match_threshold=args.match_threshold)
                mean_match_scores[dataset] = mean
                contaminated_ids.update(cached_ids)
                print(f"\tNumber of matching train instances: {len(cached_ids)}")
                print(f"\tMean match score: {mean:.6f}")
                continue

            need_trust = any(k in (dataset or "") for k in ("deepmind", "livecodebench", "alpaca", "agi"))
            try:
                query_dataset = iter_split(dataset, subset, split, limit, need_trust)
            except ValueError:
                if args.subset is None:
                    from datasets import get_dataset_config_names
                    query_dataset = itertools.chain.from_iterable(
                        iter_split(dataset, sub, split, limit, need_trust) for sub in get_dataset_config_names(dataset)
                    )
                else:
                    raise

            mean_sum, mean_n = 0.0, 0

            with open(output_filename, "w") as outfile:
                if args.index_type == "text":
                    if args.ngram_size is None:
                        # EXACT MATCH — streaming + optional multiprocessing
                        if args.workers > 1:
                            task = partial(_exact_task, es_url=args.es_url, es_password=es_password,
                                           index_name=index_name, fields=fields, search_size=args.search_size)
                            with ctx.Pool(processes=args.workers, maxtasksperchild=50) as pool:
                                for rec, ids, score in tqdm(pool.imap_unordered(task, query_dataset), desc="exact"):
                                    mean_sum += (1 if (args.match_threshold and score > args.match_threshold) else score)
                                    mean_n += 1
                                    contaminated_ids.update(ids)
                                    json.dump(rec, outfile); outfile.write("\n")
                        else:
                            for datum in tqdm(query_dataset, desc="exact"):
                                rec, ids, score = _exact_task(
                                    datum,
                                    es_url=args.es_url, es_password=es_password,
                                    index_name=index_name, fields=fields, search_size=args.search_size
                                )
                                mean_sum += (1 if (args.match_threshold and score > args.match_threshold) else score)
                                mean_n += 1
                                contaminated_ids.update(ids)
                                json.dump(rec, outfile); outfile.write("\n")

                    else:
                        # NGRAM MATCH — streaming + optional multiprocessing
                        if args.workers > 1:
                            task = partial(_ngram_task, es_url=args.es_url, es_password=es_password,
                                           index_name=index_name, fields=fields, ngram_size=args.ngram_size, search_size=args.search_size)
                            with ctx.Pool(processes=args.workers, maxtasksperchild=20) as pool:
                                for rec, per_doc_scores, top_score in tqdm(pool.imap_unordered(task, query_dataset), desc="ngram"):
                                    if args.match_threshold is not None:
                                        mean_sum += 1 if top_score > args.match_threshold else 0
                                        contaminated_ids.update([_id for _id, s in per_doc_scores.items() if s > args.match_threshold])
                                    else:
                                        mean_sum += top_score
                                    mean_n += 1
                                    json.dump(rec, outfile); outfile.write("\n")
                        else:
                            for datum in tqdm(query_dataset, desc="ngram"):
                                rec, per_doc_scores, top_score = _ngram_task(
                                    datum,
                                    es_url=args.es_url, es_password=es_password,
                                    index_name=index_name, fields=fields, ngram_size=args.ngram_size, search_size=args.search_size
                                )
                                if args.match_threshold is not None:
                                    mean_sum += 1 if top_score > args.match_threshold else 0
                                    contaminated_ids.update([_id for _id, s in per_doc_scores.items() if s > args.match_threshold])
                                else:
                                    mean_sum += top_score
                                mean_n += 1
                                json.dump(rec, outfile); outfile.write("\n")

                else:
                    # VECTOR MATCH — streaming (single-process)
                    if args.workers > 1:
                        print("[warn] --workers>1 is ignored for index_type=vector to avoid GPU/model duplication.")
                    es = get_es(args.es_url, es_password)
                    model, tokenizer = prepare_embedding_model(args.model)
                    mean_acc = {"sum": 0.0, "n": 0}
                    vector_match_streaming(
                        es, index_name, query_dataset, fields, model, tokenizer, args.max_batch_tokens,
                        args.search_size, outfile, args.match_threshold, contaminated_ids, mean_acc
                    )
                    mean_sum += mean_acc["sum"]; mean_n += mean_acc["n"]

            mean_match_score = (mean_sum / mean_n) if mean_n else 0.0
            print(f"\tNumber of matching train instances so far: {len(contaminated_ids)}")
            print(f"\tMean match score: {mean_match_score:.6f}")
            mean_match_scores[dataset] = mean_match_score

        all_index_match_scores.append(mean_match_scores)
        all_index_contaminated_ids.append(contaminated_ids)

    # TSV summary
    output_file = os.path.join(args.output_dir, "contamination_results.tsv")
    print(f"TSV file with all results: {output_file}")
    with open(output_file, "w") as outfile:
        print("\t" + "\t".join(ev[0] for ev in eval_sets), file=outfile)
        for index_name, mean_match_scores in zip(index_names, all_index_match_scores):
            print(index_name + "\t" + "\t".join([f"{mean_match_scores.get(ev[0], 0.0):.4f}" for ev in eval_sets]), file=outfile)

    if args.decontaminate:
        for dataset_name, contaminated_ids in zip(dataset_names, all_index_contaminated_ids):
            print(f"Decontaminating {dataset_name}")
            train_dataset = load_dataset(dataset_name, split="train")
            decontaminated_dataset = []
            num_kept = 0
            num_total = 0
            for i, datum in enumerate(train_dataset):
                num_total += 1
                if i in contaminated_ids:
                    continue
                num_kept += 1
                decontaminated_dataset.append(datum)
            output_path = os.path.join(args.output_dir, dataset_name.replace("/", "_") + "_decontaminated")
            os.makedirs(output_path, exist_ok=True)
            parquet_file_name = os.path.join(output_path, "train.parquet")
            hf_dataset = Dataset.from_list(decontaminated_dataset)
            hf_dataset.to_parquet(parquet_file_name)
            print(f"\tWrote parquet files to {output_path}")
            print(f"\tRemoved {num_total - num_kept} train instances.")
            print(f"\tKept {100 * num_kept / num_total:.2f}% of the original data.")


if __name__ == "__main__":
    main()