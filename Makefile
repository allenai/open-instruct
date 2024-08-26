.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = open_instruct

check_dirs := open_instruct 
image_name ?= open_instruct_dev

BEAKER_BASE_IMAGE ?= ai2/cuda12.0-cudnn8-dev-ubuntu20.04


style:
	python -m black --line-length 119 --target-version py310 $(check_dirs)
	python -m isort $(check_dirs) --profile black -p open_instruct

quality:
	python -m autoflake -r --exclude=wandb --in-place --remove-unused-variables --remove-all-unused-imports $(check_dirs)
	python -m flake8 --ignore E501,W503 $(check_dirs)

# internally use only
bimage:
	beaker image pull $(BEAKER_BASE_IMAGE)
	$(eval BEAKER_BASE_IMAGE_DOCKER_TAG := $(shell beaker image get ai2/cuda12.0-cudnn8-dev-ubuntu20.04 --format json | jq -r '.[0].dockerTag'))
	docker build --build-arg BEAKER_BASE_IMAGE_DOCKER_TAG=$(BEAKER_BASE_IMAGE_DOCKER_TAG) -f Dockerfile.speed . -t $(image_name)
	beaker image delete $(whoami)/$(image_name)
	beaker image create $(image_name) -n $(image_name) -w ai2/$(whoami)