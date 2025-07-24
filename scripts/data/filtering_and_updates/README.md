# Data Filtering and Updates Scripts

* `scripts/data/filtering_and_updates/filter_special_tokens.py`: Removes special tokens like `<think>`, `</think>`, `<answer>`, `</answer>` from dataset text fields
* `scripts/data/filtering_and_updates/filter_wildchat.py`: Filters Wildchat datasets by removing toxic, redacted examples and keeping only English examples
* `scripts/data/filtering_and_updates/filter_cutoff_date.py`: Removes mentions of knowledge cutoff dates from post-training datasets (e.g., "as my last update in April 2023")
* `scripts/data/filtering_and_updates/filter_cots.py`: Filters datasets for proper thinking token format and answer token format, specifically for reasoner model generations
* `scripts/data/filtering_and_updates/filter_dataset_by_keywords.py`: Removes phrases identifying the model as specific entities (e.g., "I am DeepSeek", "OpenAI", "Claude") from datasets
* `scripts/data/filtering_and_updates/filter_datasets_sequential.sh`: Shell script that runs keyword filtering followed by cutoff date filtering sequentially
* `scripts/data/filtering_and_updates/update_subsets.py`: Loads base dataset, removes unwanted sources, adds additional datasets, removes columns, and pushes combined dataset