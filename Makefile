.PHONY: precommit

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = open_instruct

check_dirs := open_instruct 

precommit:
	pre-commit run --all-files