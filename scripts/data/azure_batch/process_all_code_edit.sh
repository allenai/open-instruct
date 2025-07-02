#!/bin/bash
set -e

# Process all code edit batches using the batch mapping file
echo "Processing code edit batches from output/batch_files/code-edit-batch-ids.txt"
python process_code_edit_results.py \
    output/code-edit-batch-ids-python.txt \
    --no-upload
#output/code-edit-batch-ids.txt
echo "Completed processing all code edit batches"
