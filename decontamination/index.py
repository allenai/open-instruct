"""
You need Elasticsearch up and running. You can run it locally as follows:
https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html
"""

import os
import argparse
import json

import tqdm.autonotebook
from datasets import load_dataset
from elasticsearch import Elasticsearch, helpers



parser = argparse.ArgumentParser()
parser.add_argument("--es_url", type=str, default="http://localhost:9200")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--messages_field", type=str, default="conversations")
parser.add_argument("--query_filter", type=str, default="from:User")
parser.add_argument("--query_field", type=str, default="value")
parser.add_argument("--index_name", type=str, required=True)
parser.add_argument("--index_type", type=str, choices=["text", "vector"], default="text")
parser.add_argument("--text_batch_size", type=int, default=1000, help="Batch size used if the `index_type` is `text`.")
parser.add_argument("--model", type=str, default="nvidia/NV-Embed-v2")
parser.add_argument("--max_batch_tokens", type=int, default=10000, help="Maximum number of tokens per batch if the `index_type` is `vector`.")
parser.add_argument("--data_log", type=str)
args = parser.parse_args()

es = Elasticsearch(
    args.es_url,
    basic_auth=("elastic", os.environ["ELASTIC_PASSWORD"]),
)

if not es.indices.exists(index=args.index_name):
    if args.index_type == "text":
        es_index = {
            "mappings": {
                "properties": {
                    "text": {"type": "text", "index": True},
                    "original_id": {"type": "integer"},
                }
            }
        }
    else:
        es_index = {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "original_id": {"type": "integer"},
                    "vector": {"type": "dense_vector", "dims": 4096, "index": True, "similarity": "dot_product"},
                }
            }
        }

    es.indices.create(index=args.index_name, body=es_index)
    print("Created a new index.")
    index_size = 0
else:
    stats = es.indices.stats(index=args.index_name)
    index_size = stats["indices"][args.index_name]["total"]["docs"]["count"]
    print(f"Index of size {index_size} exists.")

dataset = load_dataset(args.dataset, split=args.split)
data_to_index = []

query_filter_key, query_filter_value = args.query_filter.split(":")

print(f"Reading {args.messages_field} from {args.dataset}")

for i, datum in tqdm.tqdm(enumerate(dataset)):
    for message in datum[args.messages_field]:
        if message[query_filter_key] == query_filter_value:
            data_to_index.append(
                {
                    "text": message[args.query_field],
                    "metadata": datum,
                    "original_id": i,
                }
            )

print(f"Read {args.dataset} for indexing. Has {len(dataset)} instances and {len(data_to_index)} messages.")

if index_size < len(data_to_index):
    idx = index_size
    if args.index_type == "text":
        with tqdm.tqdm(total=len(data_to_index) - idx) as pbar:
            while idx < len(data_to_index):
                bulk_data = []
                for datum in data_to_index[idx: idx+args.text_batch_size]:
                    bulk_data.append(
                        {
                            "_index": args.index_name,
                            "_source": {"text": datum["text"], "original_id": datum["original_id"]},
                        }
                    )

                helpers.bulk(es, bulk_data)
                idx += len(bulk_data)
                pbar.update(len(bulk_data))

    else:
        # Embedding model setup
        import torch
        from transformers import AutoModel, AutoTokenizer
        # Prompt based on the usage example at https://huggingface.co/nvidia/NV-Embed-v2
        query_prefix = "Instruct: Given a user request to a chatbot, retrieve requests that are semantically equivalent to the given request\nQuery: "

        model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
        model.eval()
        model.cuda()
        device = model.device
        print(f"Loaded {args.model} on device:{device}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        if torch.cuda.device_count() > 1:
            print("Found multiple gpus. Will use data parallel.")
            for module_key, module in model._modules.items():
                model._modules[module_key] = torch.nn.DataParallel(module)

        # Indexing
        print("Indexing data (you can stop it by pressing Ctrl+C once):")
        if args.data_log is not None:
            log_ptr = open(args.data_log, "w" if idx == 0 else "a") 
            batch_id = 0
        with tqdm.tqdm(total=len(data_to_index) - idx) as pbar:
            while idx < len(data_to_index):
                batch_data = []
                batch_inputs = []
                max_seq_tokens = 0
                batch_size = 0
                while True:
                    datum = data_to_index[idx] 
                    datum_seq_length = len(tokenizer.tokenize(datum["text"]))
                    if datum_seq_length > args.max_batch_tokens:
                        # One really long instance
                        print(f"Warning: Skipping instance {datum['text']}")
                        idx += 1
                        continue
                    max_seq_tokens = max(max_seq_tokens, datum_seq_length)
                    batch_size += 1
                    if (max_seq_tokens * batch_size) > args.max_batch_tokens:
                        break
                    batch_data.append(datum)
                    batch_inputs.append(datum["text"])
                    idx += 1
                    if idx == len(data_to_index):
                        break
                if args.data_log is not None:
                    print(
                        json.dumps(
                            {
                                "batch_id": batch_id,
                                "batch_size": len(batch_inputs),
                                "seq_length": max_seq_tokens,
                                "inputs": batch_inputs,
                            },
                            indent=2
                        ),
                        file=log_ptr,
                        flush=True
                    )
                    batch_id += 1
                embeddings = model.encode(batch_inputs, instruction=query_prefix)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                bulk_data = []
                for datum, embedding in zip(batch_data, embeddings.cpu().numpy()):
                    bulk_data.append(
                        {
                            "_index": args.index_name,
                            "_source": {"text": datum["text"], "original_id": datum["original_id"], "vector": embedding},
                        }
                    )

                helpers.bulk(es, bulk_data)
                pbar.update(len(batch_data))
else:
    print("All data is already indexed. Nothing to do.")
