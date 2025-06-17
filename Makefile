.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = open_instruct

check_dirs := open_instruct 

style:
	python3 -m ruff check --fix $(check_dirs)
	python3 -m ruff format $(check_dirs)

quality:
	python3 -m ruff check $(check_dirs)