#!/bin/bash
image_name=code_perf_penalty

# Build and push the Docker image to Beaker
docker build . -t $image_name
beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')

# Use '|| true' to prevent script from exiting if image doesn't exist to delete
beaker image delete $beaker_user/$image_name || true
# Create the image in the same workspace used for jobs
beaker image create $image_name -n $image_name -w ai2/$beaker_user

bash scripts/train/rlvr/experiment_ace_and_stdio_from_base.sh