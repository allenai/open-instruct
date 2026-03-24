# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "vllm>=0.18.0",
#     "transformers>=5.3.0",
#     "torch>=2.10.0",
# ]
#
# [tool.uv]
# override-dependencies = ["transformers>=5.3.0"]
# ///
"""Reproduce vLLM hybrid model dtype serialization bug.

MambaSpec.dtypes is typed as tuple[torch.dtype] (exactly 1 element),
but OlmoHybridForCausalLM has 2 state dtypes, causing:
    msgspec.ValidationError: Expected `array` of length 1, got 2 - at `$.dtypes`

Run on a GPU machine:
    uv run scripts/debug/repro_vllm_hybrid_dtype.py

Or on Beaker:
    ./scripts/debug/repro_vllm_hybrid_dtype_beaker.sh
"""

import logging

import vllm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "allenai/Olmo-Hybrid-Instruct-DPO-7B"

if __name__ == "__main__":
    logger.info("Starting vLLM with %s — expect msgspec.ValidationError", MODEL)
    llm = vllm.LLM(model=MODEL, trust_remote_code=True, enforce_eager=True, gpu_memory_utilization=0.5)
    outputs = llm.generate(["What is 2 + 2?"], vllm.SamplingParams(temperature=0.7, max_tokens=64))
    for output in outputs:
        logger.info("Generated: %s", output.outputs[0].text)
