python mason.py \
    --cluster ai2/neptune-cirrascale \
    --workspace ai2/usable-olmo \
    --priority normal \
    --image nathanl/rewardbench_auto \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 1 -- python scripts/data/preferences/olmo3/extract_responses.py