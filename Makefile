.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = open_instruct

check_dirs := open_instruct *mason.py

style:
	uv run ruff format $(check_dirs)

quality:
	uv run ruff check -q --fix $(check_dirs)
	uv run python -m compileall -qq $(check_dirs)
	uv run ty check

style-check:   ## *fail* if anything needs rewriting
	uv run ruff format --check --diff $(check_dirs)

quality-check: ## *fail* if any rewrite was needed
	uv run ruff check --exit-non-zero-on-fix $(check_dirs)
	uv run ty check
	uv run python -m compileall -qq $(check_dirs)

docker:
	# rsync -a --delete ../oe-eval-internal/ oe-eval-internal/
	DOCKER_BUILDKIT=1 docker build -f Dockerfile \
		--build-arg UV_CACHE_DIR=$(UV_CACHE_DIR) \
		--build-arg GIT_COMMIT="$$(git rev-parse --short HEAD)" \
		--build-arg GIT_BRANCH="$$(git rev-parse --abbrev-ref HEAD)" \
		-t open_instruct .
	# if you are internally at AI2, you can create an image like this:
	$(eval beaker_user := $(shell beaker account whoami --format json | jq -r '.[0].name'))
	beaker image rename $(beaker_user)/open_instruct_test ""
	beaker image create open_instruct_test -n open_instruct_test -w ai2/$(beaker_user)
