#!/bin/bash
# Convert Instruct SFT sweep OLMo-core checkpoints to HuggingFace format.
# This runs in OLMo-core, not open-instruct.

STEP=step3256
TOKENIZER=allenai/olmo-3-tokenizer-instruct-dev
BEAKER_IMAGE=tylerr/olmo-core-tch291cu128-2025-11-25

LRS=(1e-4 9e-5 8e-5 6e-5 5e-5 2.5e-5)

for LR in "${LRS[@]}"; do
    RUN_NAME="HYBRID_INSTRUCT_SFT_0218_${LR}"
    INPUT="/weka/oe-training-default/ai2-llm/checkpoints/nathanl/olmo-sft/${RUN_NAME}/${STEP}"
    OUTPUT="/weka/oe-adapt-default/nathanl/checkpoints/${RUN_NAME}/${STEP}-hf"

    echo "Converting ${RUN_NAME}..."

    gantry run --cluster ai2/saturn-cirrascale --timeout 0 -y --budget ai2/oe-adapt --workspace ai2/olmo-instruct \
        --beaker-image "${BEAKER_IMAGE}" \
        --install "/opt/conda/bin/pip install -e '.[fla,transformers]'" \
        --weka=oe-adapt-default:/weka/oe-adapt-default \
        --weka=oe-training-default:/weka/oe-training-default \
        --priority urgent \
        --gpus 1 \
        -- /opt/conda/bin/python src/examples/huggingface/convert_checkpoint_to_hf_hybrid.py \
            -i "${INPUT}" \
            -o "${OUTPUT}" \
            -t "${TOKENIZER}" \
            --max-sequence-length 32768 \
            --skip-validation
done
