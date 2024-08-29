import datasets
from huggingface_hub import HfApi
api = HfApi()

ds = datasets.load_dataset("allenai/tulu-2.5-preference-data")

# for each config in the above dict, upload a new private dataset to ai2-adapt-dev 
# with the corresponding subsets, for easy mixing

for split in ds.keys():
    dataset_name = f"tulu-2.5-prefs-{split}"
    print(f"Creating dataset {dataset_name}...")
    ds[split].push_to_hub(f"ai2-adapt-dev/{dataset_name}", private=True)