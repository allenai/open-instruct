.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = open_instruct

check_dirs := open_instruct 

style:
	uv run ruff format $(check_dirs)

quality:
	uv run ruff check --fix $(check_dirs)
