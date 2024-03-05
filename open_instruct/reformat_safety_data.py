import os
import json
from reformat_datasets import convert_safety_adapt_data
import sys

raw_data_dir = sys.argv[1]
output_dir = sys.argv[2]

convert_safety_adapt_data(
    data_dir=raw_data_dir,
    output_dir=output_dir,
    num_examples=None
)
