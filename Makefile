.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = open_instruct

check_dirs := open_instruct 

style:
	python -m black --line-length 119 --target-version py310 $(check_dirs)
	python -m isort $(check_dirs) --profile black

quality:
	python -m autoflake -r --exclude=wandb --in-place --remove-unused-variables --remove-all-unused-imports $(check_dirs)
	python -m flake8 --ignore E501,W503 $(check_dirs)