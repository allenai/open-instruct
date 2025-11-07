export BEAKER_TOKEN=OM268h3N/wpIM2i7
beaker_whoami=$(beaker account whoami --format json | jq -r '.[0].name')
workspace=ai2/$beaker_whoami
itwl() {
    local num_gpus=${1:-0}  # Default to 0 GPU if not specified
    beaker session create \
    --budget ai2/oe-adapt  \
    --bare \
    --mount src=weka,ref=oe-adapt-default,dst=/weka/oe-adapt-default \
    --mount src=weka,ref=oe-training-default,dst=/weka/oe-training-default \
    --mount src=weka,ref=oe-eval-default,dst=/weka/oe-eval-default \
    --image beaker://jacobm/open_instruct_dev_random_rewards10 \
    --mount src=secret,ref=${beaker_whoami}_id_ed25519,dst=/root/.ssh/id_ed25519 \
    --mount src=secret,ref=${beaker_whoami}_bashrc_remote,dst=/root/.bashrc \
    --mount src=secret,ref=${beaker_whoami}_ssh_config,dst=/root/.ssh/config \
    --priority normal \
    --host-networking \
    --workspace $workspace \
    --gpus $num_gpus
}
itwl 1

# workspace=ai2/olmo-instruct

