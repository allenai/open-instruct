"""Backport of open-instruct #1904 (99b1ee970) into the 37a119ffd image, applied in the container before tokenizing:
count cached token statistics in Arrow batches instead of a single-process per-row dataset.map (about 6.5 h of a
13.5 h build). Writes open_instruct/dataset_statistics.py verbatim from 99b1ee970 and replaces the block it replaces.
Counts and the cache hash are unchanged."""
import pathlib, sys
root = pathlib.Path(sys.argv[1])
(root / "open_instruct/dataset_statistics.py").write_text('"""Token statistics over logical dataset rows without writing a derived cache."""\n\nfrom datasets import Dataset\nfrom pyarrow import compute\n\n\ndef count_tokens(dataset: Dataset, input_ids_key: str, labels_key: str, masked_token_value: int) -> tuple[int, int]:\n    total_tokens = 0\n    trainable_tokens = 0\n    columns = dataset.select_columns([input_ids_key, labels_key]).with_format("arrow")\n    for batch in columns.iter(batch_size=1024):\n        inputs, labels = batch[input_ids_key], batch[labels_key]\n        if inputs.null_count or labels.null_count:\n            raise ValueError("Token and label sequences must not be null")\n        total_tokens += (\n            compute.call_function("sum", [compute.call_function("list_value_length", [inputs])]).as_py() or 0\n        )\n        # Python\'s original label != mask count includes null scalar labels.\n        trainable = compute.fill_null(\n            compute.call_function("not_equal", [compute.call_function("list_flatten", [labels]), masked_token_value]),\n            True,\n        )\n        trainable_tokens += compute.call_function("sum", [trainable]).as_py() or 0\n    return total_tokens, trainable_tokens\n')
p = root / "open_instruct/dataset_transformation.py"
s = p.read_text()
old_import = "from open_instruct import launch_utils, logger_utils, tokenizer_utils\n"
new_import = "from open_instruct import dataset_statistics as token_statistics\n" + old_import
old_block = '                total_tokens = 0\n                trainable_tokens = 0\n\n                def count_tokens(sample):\n                    token_count = len(sample[INPUT_IDS_KEY])\n                    trainable_tokens = sum(1 for label in sample[LABELS_KEY] if label != MASKED_TOKEN_VALUE)\n                    return {"token_count": token_count, "label_token_count": trainable_tokens}\n\n                token_count_dataset = dataset.map(count_tokens, batched=False)\n                total_tokens = sum(token_count_dataset["token_count"])\n                trainable_tokens = sum(token_count_dataset["label_token_count"])\n'
new_block = '                total_tokens, trainable_tokens = token_statistics.count_tokens(\n                    dataset, INPUT_IDS_KEY, LABELS_KEY, MASKED_TOKEN_VALUE\n                )\n'
assert s.count(old_import) == 1 and s.count(old_block) == 1, "dataset_transformation.py does not match 37a119ffd"
s = s.replace(old_import, new_import).replace(old_block, new_block)
p.write_text(s)
print("COUNT-PATCHED", p)
