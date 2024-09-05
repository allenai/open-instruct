from huggingface_hub import HfApi

api = HfApi()

dataset_names = [dd.id for dd in api.list_datasets(author="ai2-adapt-dev")]
# print markdown list where datasets are links
# links look like [dataset_name](https://huggingface.co/datasets/ai2-adapt-dev/dataset_name)

for name in dataset_names:
    print(f"* [{name}](https://huggingface.co/datasets/{name})")
