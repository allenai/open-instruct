#!/bin/bash
image_name=open-coding-agent

# Build and push the Docker image to Beaker
docker build -f Dockerfile.uv --build-arg UV_CACHE_DIR=$UV_CACHE_DIR -t $image_name .
#docker build . -t $image_name


beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')

# Use '|| true' to prevent script from exiting if image doesn't exist to delete
beaker image delete $beaker_user/$image_name || true
# Create the image in the same workspace used for jobs
beaker image create $image_name -n $image_name -w ai2/$beaker_user
bash coding-agent/launch_view_sft.sh