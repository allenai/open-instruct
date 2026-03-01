#!/bin/bash
# Convert Think SFT OLMo-core checkpoint to HuggingFace format.
# This runs in OLMo-core, not open-instruct.

gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/olmo-instruct \
    --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
    --install "/opt/conda/bin/pip install -e '.[fla,transformers]'" \
    --weka=oe-adapt-default:/weka/oe-adapt-default \
    --weka=oe-training-default:/weka/oe-training-default \
    --priority urgent \
    --gpus 1 \
    -- /opt/conda/bin/python src/examples/huggingface/convert_checkpoint_to_hf_hybrid.py \
        -i /weka/oe-training-default/ai2-llm/checkpoints/nathanl/olmo-sft/TEST_HYBRIC_SFT_LARGER_LR2.5e-5/step46412 \
        -o /weka/oe-adapt-default/nathanl/checkpoints/TEST_HYBRIC_SFT_LARGER_LR2.5e-5/step46412-hf \
        --max-sequence-length 32768 \
        --skip-validation
