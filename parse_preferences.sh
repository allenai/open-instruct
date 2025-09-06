# python mason.py \
#     --cluster ai2/neptune-cirrascale \
#     --workspace ai2/usable-olmo \
#     --priority normal \
#     --image nathanl/rewardbench_auto \
#     --preemptible \
#     --num_nodes 1 \
#     --budget ai2/oe-adapt \
#     --gpus 1 -- python scripts/data/preferences/olmo3/extract_responses.py --parsed /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/grades/workspaces/qwq/32b/parsed_judgments_combined.jsonl \
#     --selected /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/selected_models_FINAL.jsonl \
#     --combined-dir /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/combined-outputs \

# python mason.py \
#     --cluster ai2/neptune-cirrascale \
#     --workspace ai2/usable-olmo \
#     --priority normal \
#     --image nathanl/rewardbench_auto \
#     --preemptible \
#     --num_nodes 1 \
#     --budget ai2/oe-adapt \
#     --gpus 1 -- python scripts/data/preferences/olmo3/extract_responses.py --parsed /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/grades/workspaces/gemma3/27b/parsed_judgments_combined.jsonl \
#     --selected /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/selected_models_FINAL.jsonl \
#     --combined-dir /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/combined-outputs \
    
# python mason.py \
#     --cluster ai2/neptune-cirrascale \
#     --workspace ai2/usable-olmo \
#     --priority normal \
#     --image nathanl/rewardbench_auto \
#     --preemptible \
#     --num_nodes 1 \
#     --budget ai2/oe-adapt \
#     --gpus 1 -- python scripts/data/preferences/olmo3/extract_responses_faster.py --parsed /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/grades/workspaces/qwq/32b/parsed_judgments_combined.jsonl \
#     --selected /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/selected_models_FINAL.jsonl \
#     --combined-dir /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/combined-outputs

# python mason.py \
#     --cluster ai2/neptune-cirrascale \
#     --workspace ai2/usable-olmo \
#     --priority normal \
#     --image nathanl/rewardbench_auto \
#     --preemptible \
#     --num_nodes 1 \
#     --budget ai2/oe-adapt \
#     --gpus 1 -- python scripts/data/preferences/olmo3/extract_responses_faster.py --parsed /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/grades/workspaces/gemma3/27b/parsed_judgments_combined.jsonl \
#     --selected /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/selected_models_FINAL.jsonl \
#     --combined-dir /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/combined-outputs

# for i in aa ab ac ad ae af ag ah ai aj; do
# python mason.py \
#     --cluster ai2/neptune-cirrascale ai2/saturn-cirrascale ai2/jupiter-cirrascale-2\
#     --workspace ai2/usable-olmo \
#     --priority normal \
#     --image nathanl/rewardbench_auto \
#     --preemptible \
#     --num_nodes 1 \
#     --budget ai2/oe-adapt \
#     --gpus 1 -- python scripts/data/preferences/olmo3/extract_responses_faster.py --parsed /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/grades/workspaces/qwq/32b/parsed_judgements.part_$i \
#     --out /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/grades/workspaces/qwq/32b/parsed_responses_chunked.part_$i \
#     --selected /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/selected_models_FINAL.jsonl \
#     --combined-dir /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/combined-outputs \
#     --join-workers 1

# python mason.py \
#     --cluster ai2/neptune-cirrascale ai2/saturn-cirrascale ai2/jupiter-cirrascale-2\
#     --workspace ai2/usable-olmo \
#     --priority normal \
#     --image nathanl/rewardbench_auto \
#     --preemptible \
#     --num_nodes 1 \
#     --budget ai2/oe-adapt \
#     --gpus 1 -- python scripts/data/preferences/olmo3/extract_responses_faster.py --parsed /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/grades/workspaces/gemma3/27b/parsed_judgements.part_$i \
#     --out /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/grades/workspaces/gemma3/27b/parsed_responses_chunked.part_$i \
#     --selected /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/selected_models_FINAL.jsonl \
#     --combined-dir /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/combined-outputs \
#     --join-workers 1
# done

for i in aa ab ac ad ae af ag ah ai aj; do
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/usable-olmo \
    --priority normal \
    --image nathanl/open_instruct_auto \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 1 -- python scripts/data/preferences/olmo3/extract_responses_faster_debug.py --parsed /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/grades/workspaces/qwq/32b/parsed_judgements.part_$i \
    --out /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/grades/workspaces/qwq/32b/parsed_responses_chunked.part_${i}_fixed \
    --selected /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/selected_models_FINAL.jsonl \
    --combined-dir /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/combined-outputs \
    --overwrite \
    --allow-prompt-mismatch \
    --join-workers 1

python mason.py \
    --cluster ai2/jupiter-cirrascale-2\
    --workspace ai2/usable-olmo \
    --priority normal \
    --image nathanl/open_instruct_auto \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 1 -- python scripts/data/preferences/olmo3/extract_responses_faster_debug.py --parsed /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/grades/workspaces/gemma3/27b/parsed_judgements.part_$i \
    --out /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/grades/workspaces/gemma3/27b/parsed_responses_chunked.part_${i}_fixed \
    --selected /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/selected_models_FINAL.jsonl \
    --combined-dir /weka/oe-adapt-default/jacobm/rl-sft/olmo3-preferences/combined-outputs \
    --overwrite \
    --allow-prompt-mismatch \
    --join-workers 1
done