import argparse
import os

import yaml
from datasets import Dataset, load_dataset
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

import open_instruct.utils as open_instruct_utils


def create_text_index(es, index_name):
    mappings = {"properties": {"text": {"type": "text", "index": True}, "original_id": {"type": "integer"}}}
    # The default analyzer is a "standard" analyzer which lowercases and splits tokens on all punctuation. This is not a great choice for math and
    # coding datasets where we would lose math operators, equations get split, etc. The following custom analyzer uses a regex pattern that splits on
    # fewer characters. This is not perfect either, but is a better choice across evals.
    settings = {
        "analysis": {
            "analyzer": {"tulu_analyzer": {"type": "pattern", "pattern": '[ ,.?!:;()"-]|\\n|\\\\', "lowercase": True}}
        }
    }
    es.indices.create(index=index_name, mappings=mappings, settings=settings)
    print(f"Created a new text index: {index_name}")


def create_vector_index(es, index_name):
    mappings = {
        "properties": {
            "text": {"type": "text"},
            "original_id": {"type": "integer"},
            "vector": {"type": "dense_vector", "dims": 4096, "index": True, "similarity": "dot_product"},
        }
    }
    es.indices.create(index=index_name, mappings=mappings)
    print(f"Created a new vector index: {index_name}")


def read_dataset(dataset_name, split, messages_field, query_filter, query_field):
    dataset: Dataset = load_dataset(dataset_name, split=split, num_proc=open_instruct_utils.max_num_processes())  # type: ignore[assignment]
    data_to_index = []

    query_filter_key, query_filter_value = query_filter.split(":")

    print(f"Reading {messages_field} from {dataset_name}")

    for i, datum in tqdm(enumerate(dataset)):
        for message in datum[messages_field]:
            if message[query_filter_key] == query_filter_value:
                data_to_index.append({"text": message[query_field], "metadata": datum, "original_id": i})

    print(f"Read {dataset_name} for indexing. Has {len(dataset)} instances and {len(data_to_index)} messages.")
    return data_to_index


def index_dataset_text(data_to_index, es, index_name, text_batch_size):
    stats = es.indices.stats(index=index_name)
    index_size = stats["indices"][index_name]["total"]["docs"]["count"]
    if index_size > 0:
        print(f"Index of size {index_size} exists. Adding data.")

    if index_size < len(data_to_index):
        idx = index_size
        with tqdm(total=len(data_to_index) - idx) as pbar:
            while idx < len(data_to_index):
                bulk_data = []
                for datum in data_to_index[idx : idx + text_batch_size]:
                    bulk_data.append(
                        {"_index": index_name, "_source": {"text": datum["text"], "original_id": datum["original_id"]}}
                    )

                helpers.bulk(es, bulk_data)
                idx += len(bulk_data)
                pbar.update(len(bulk_data))
        print(f"Indexing into {index_name} complete!\n")
    else:
        print("All data is already indexed. Nothing to do.\n")


def index_dataset_vectors(data_to_index, es, index_name, model_name, max_batch_tokens):
    stats = es.indices.stats(index=index_name)
    index_size = stats["indices"][index_name]["total"]["docs"]["count"]
    if index_size > 0:
        print(f"Index of size {index_size} exists. Adding data.")

    if index_size < len(data_to_index):
        # Embedding model setup
        import torch
        from transformers import AutoModel, AutoTokenizer

        # Prompt based on the usage example at https://huggingface.co/nvidia/NV-Embed-v2
        query_prefix = "Instruct: Given a user request to a chatbot, retrieve requests that are semantically equivalent to the given request\nQuery: "

        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.eval()
        model.cuda()
        device = model.device
        print(f"Loaded {model_name} on device:{device}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if torch.cuda.device_count() > 1:
            print("Found multiple gpus. Will use data parallel.")
            for module_key, module in model._modules.items():
                model._modules[module_key] = torch.nn.DataParallel(module)

        # Indexing
        idx = index_size
        print("Indexing data (you can stop it by pressing Ctrl+C once):")
        with tqdm(total=len(data_to_index) - idx) as pbar:
            while idx < len(data_to_index):
                batch_data = []
                batch_inputs = []
                max_seq_tokens = 0
                batch_size = 0
                while True:
                    datum = data_to_index[idx]
                    datum_seq_length = len(tokenizer.tokenize(datum["text"]))
                    if datum_seq_length > max_batch_tokens:
                        # One really long instance
                        print(f"Warning: Skipping instance {datum['text']}")
                        idx += 1
                        continue
                    max_seq_tokens = max(max_seq_tokens, datum_seq_length)
                    batch_size += 1
                    if (max_seq_tokens * batch_size) > max_batch_tokens:
                        break
                    batch_data.append(datum)
                    batch_inputs.append(datum["text"])
                    idx += 1
                    if idx == len(data_to_index):
                        break
                embeddings = model.encode(batch_inputs, instruction=query_prefix)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                bulk_data = []
                for datum, embedding in zip(batch_data, embeddings.cpu().numpy()):
                    bulk_data.append(
                        {
                            "_index": index_name,
                            "_source": {
                                "text": datum["text"],
                                "original_id": datum["original_id"],
                                "vector": embedding,
                            },
                        }
                    )

                helpers.bulk(es, bulk_data)
                pbar.update(len(batch_data))

        print(f"Indexing into {index_name} complete!\n")
    else:
        print("All data is already indexed. Nothing to do.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--es_url", type=str, default="http://localhost:9200")
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--dataset_mixer_config",
        type=str,
        help="Path to a train config file in yml format with a `dataset_mixer` field.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--messages_field", type=str, default="messages")
    parser.add_argument("--query_filter", type=str, default="role:user")
    parser.add_argument("--query_field", type=str, default="content")
    parser.add_argument("--index_type", type=str, choices=["text", "vector"], default="text")
    parser.add_argument(
        "--text_batch_size", type=int, default=1000, help="Batch size used if the `index_type` is `text`."
    )
    parser.add_argument("--model", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=10000,
        help="Maximum number of tokens per batch if the `index_type` is `vector`.",
    )
    args = parser.parse_args()

    if args.dataset_mixer_config is not None:
        print(f"Reading from dataset mixer info from train config: {args.dataset_mixer_config}")
        train_config = yaml.safe_load(open(args.dataset_mixer_config))
        dataset_names = list(train_config["dataset_mixer"].keys())
        print(f"Indexing {len(dataset_names)} datasets: {dataset_names}")
    elif args.dataset is not None:
        dataset_names = [args.dataset]
    else:
        raise RuntimeError("Specify a dataset or provide a train config file with dataset mixer info.")

    es = Elasticsearch(args.es_url, basic_auth=("elastic", os.environ["ELASTIC_PASSWORD"]))
    for i, dataset_name in enumerate(dataset_names):
        print(f"Processing dataset {i+1} / {len(dataset_names)}: {dataset_name}")
        data_to_index = read_dataset(
            dataset_name=dataset_name,
            split=args.split,
            messages_field=args.messages_field,
            query_filter=args.query_filter,
            query_field=args.query_field,
        )
        index_name = dataset_name.replace("/", "_").lower() + f"_{args.index_type}"
        if args.index_type == "text":
            if not es.indices.exists(index=index_name):
                create_text_index(es, index_name=index_name)
            index_dataset_text(
                data_to_index=data_to_index, es=es, index_name=index_name, text_batch_size=args.text_batch_size
            )
        else:
            if not es.indices.exists(index=index_name):
                create_vector_index(es, index_name=index_name)
            index_dataset_vectors(
                data_to_index=data_to_index,
                es=es,
                index_name=index_name,
                model_name=args.model,
                max_batch_tokens=args.max_batch_tokens,
            )


if __name__ == "__main__":
    main()
