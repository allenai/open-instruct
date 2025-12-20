# Scripts for computing overlap between train and test sets

These scripts are for creating Elasticsearch indices over training datasets, particularly instruction tuning datasets, and querying them with test sets to compute overlap. They can be used for quantifying and analyzing training dataset contamination.

## Running Elasticsearch

Elasticsearch needs to up and running for creating and querying indices. You can run it locally by following the steps [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html). Make sure to keep track of the password and save it as an environment variable, `ELASTIC_PASSWORD`, e.g.:

```bash
export ELASTIC_PASSWORD=[password]
```

## Indexing

You can index the training sets either as text or as dense vectors. The indexing script assumes that the training dataset is a Huggingface dataset, and has a field that contains prompt-response pairs in a conversational format, e.g. a `messages` field that looks like

```json
[
    {
        "role": "user",
        "content": "Write me a poem."
    },
    {
        "role": "assistant",
        "content": "Sorry, I cannot help you with that."
    }
]
```

The script indexes each turn as a separate Elasticsearch document, and importantly only indexes the messages of one specific role. The assumption is that you would want to index only the prompts for quantifying contamination. You can control this behavior using the `--messages_field`, `--query_filter`, and `--query_field` options as follows:

```bash
python index.py --messages_field messages --query_filter role:user --query_field content
```

The setting above looks for the `messages` field in the dataset, finds messages where the `role` is `user` and indexes their `content`.

### Indexing multiple datasets

You can index one dataset at a time as follows

```bash
python index.py --dataset HF_DATASET_NAME
```

Alternatively, you can pass a training configuration yaml with a `dataset_mixer` field to index all the datasets in the mix.

```bash
python index.py --dataset_mixer_config config.yaml
```

### Indexing vector representations

By default, the indexing script indexes the text in the datasets. If you want to perform soft matching, you can change `--index_type`, and specify an embedding model (defaults to [NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2)).

```bash
python index.py --index_type vector --model nvidia/NV-Embed-v2
```

The script assumes you are running this on GPUs and uses all the available devices. You can adjust `--max_batch_tokens` to a suitable value if you run into OOM errors or if you want to use your GPUs more effectively.

## Searching

The searching script lets you query one or more `text` or a `vector` indices with a test set. When querying a `text` index, you can perform an ngram match, a full text match, or an embedding-based match of a specified field(s) in the test set. The basic usage looks like

```bash
python search.py --train_dataset_names allenai/tulu-2-sft-mixture allenai/wildchat-1m --dataset tatsu-lab/alpaca_eval --split eval --field instruction --output_dir /path/to/output
```

The command above queries the indices corresponding to the two training sets, `allenai/tulu-2-sft-mixture` and `allenai/wildchat-1m` (assuming these were indexed earlier) with the AlpacaEval dataset, particularly the `instruction` field in the `eval` split.

The script will create in the output directory one `jsonl` file per each pair of index and evaluation dataset with instance-level information about the matches, and a TSV file called `contamination_report.tsv` with a table of contamination scores for all the pairs.

Like with the indexing script, a dataset mixer configuration can be passed with the `--dataset_mixer_config` option instead of `--train_dataset_names`.

### Checking for contamination against the Tulu 3 evaluation suite

If no evaluation dataset is specified using the `--dataset` option, the entire Tulu 3 evaluation suite will be used to query the specified indices.

### Matching ngrams

Text indexes can be queried for ngram matches instead of full field matches (default) as follows

```bash
python search.py --train_dataset_names TRAIN_DATASET_NAME --ngram_size SIZE [--match_threshold THRESHOLD]
```

Matching scores are then computed as follows:
- For each token in the test instance, all matching training documents are retrieved. A training document is considered a match for a token if it is part of an ngram of the specified `SIZE` in the test instance, that also occurs in the training document.
- The single training document that covers the most number of tokens in the test instance is considered the largest match.
- If no threshold is specified, the match score for the test instance is the proportion of the matched tokens. If a threshold is specified, the score is `0` or `1` depending on the threshold.
- The evaluation dataset level match (or contamination) score is the average of instance level match scores.

### Embedding-based matching

If the index is created using `--index_type vector`, the same option needs to be specified for searching as well, along with the same `--model MODEL_NAME`. The searching script also assumes you are running this on GPUs.

You can specify a `--match_threshold` here as well, and the behavior is similar to that in ngram matching, except that the match scores here come from embedding similarity.

### Decontamination

If you need to remove instances from the training sets that match any of the test instances, just pass a `--decontaminate` option to `search.py`. The output directory will contain one decontaminated `jsonl` file per training dataset. If you pass a `--match_treshold`, only those train instances that have a matching score greater than the threshold with *any* of the test instances will be removed.

Note that elasticsearch retrieves a limited number of hits each time you search. You can increase this by requesting a larger number of results by passing a different value to `--search_size` (default is 100). Setting this to a larger number (e.g. 10000) is a good idea if you are decontaminating datasets. Since elasticsearch does not necessarily retrieve all the documents that match, it is not guaranteed that decontamination removes all the matching training instances. You can always check for contamination after decontaminating a dataset to see how effective it was.
