.PHONY: style quality docker

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = open_instruct

check_dirs := open_instruct 

style:
	uv run ruff format $(check_dirs)

quality:
	uv run ruff check --fix $(check_dirs)

style-check:   ## *fail* if anything needs rewriting
	uv run ruff format --check --diff $(check_dirs)

quality-check: ## *fail* if any rewrite was needed
	uv run ruff check --exit-non-zero-on-fix $(check_dirs)

docker:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile --build-arg UV_CACHE_DIR=$(UV_CACHE_DIR) -t open_instruct_olmo2_retrofit .
	# if you are internally at AI2, you can create an image like this:
	$(eval beaker_user := $(shell beaker account whoami --format json | jq -r '.[0].name'))
	beaker image delete $(beaker_user)/open_instruct_olmo2_retrofit
	beaker image create open_instruct_olmo2_retrofit -n open_instruct_olmo2_retrofit -w ai2/$(beaker_user)
