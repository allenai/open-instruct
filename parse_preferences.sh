python mason.py \
    --cluster ai2/neptune-cirrascale \
    --workspace ai2/usable-olmo \
    --priority normal \
    --image nathanl/rewardbench_auto \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 1 -- python scripts/data/preferences/olmo3/extract_responses.py --parsed /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/grades/workspaces/qwq/32b/parsed_judgments_combined.jsonl \
    --selected /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/selected_models_FINAL.jsonl \
    --combined-dir /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/combined-outputs \

python mason.py \
    --cluster ai2/neptune-cirrascale \
    --workspace ai2/usable-olmo \
    --priority normal \
    --image nathanl/rewardbench_auto \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 1 -- python scripts/data/preferences/olmo3/extract_responses.py --parsed /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/grades/workspaces/gemma3/27b/parsed_judgments_combined.jsonl \
    --selected /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/selected_models_FINAL.jsonl \
    --combined-dir /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/combined-outputs \
    