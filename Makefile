.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = open_instruct

check_dirs := open_instruct 

style:
	uv run black $(check_dirs)
	uv run isort $(check_dirs)

quality:
	uv run autoflake -r --exclude=wandb --in-place --remove-unused-variables --remove-all-unused-imports $(check_dirs)
	uv run flake8 --ignore E501,W503 $(check_dirs)
