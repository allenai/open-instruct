.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = open_instruct

check_dirs := open_instruct 

style:
	python3 -m black $(check_dirs)
	python3 -m isort $(check_dirs)

quality:
	python3 -m autoflake -r --exclude=wandb --in-place --remove-unused-variables --remove-all-unused-imports $(check_dirs)
	python3 -m flake8 --ignore E501,W503 $(check_dirs)