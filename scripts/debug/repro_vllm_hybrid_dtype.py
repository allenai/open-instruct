# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "vllm>=0.18.0",
#     "transformers>=5.3.0",
#     "torch>=2.10.0",
# ]
# ///
"""Reproduce vLLM hybrid model dtype serialization bug.

MambaSpec.dtypes is typed as tuple[torch.dtype] (exactly 1 element),
but OlmoHybridForCausalLM has 2 state dtypes, causing:
    msgspec.ValidationError: Expected `array` of length 1, got 2 - at `$.dtypes`

Run: uv run scripts/debug/repro_vllm_hybrid_dtype.py
"""

import logging
import typing

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "allenai/Olmo-Hybrid-Instruct-DPO-7B"

from vllm.v1 import kv_cache_interface

hints = typing.get_type_hints(kv_cache_interface.MambaSpec)
dtypes_hint = hints["dtypes"]
args = getattr(dtypes_hint, "__args__", ())
logger.info("MambaSpec.dtypes hint: %s (args=%s)", dtypes_hint, args)

if len(args) == 1:
    logger.error("BUG: dtypes is tuple[torch.dtype] (fixed length 1), hybrid models will fail.")

import vllm

logger.info("Starting vLLM with %s — expect msgspec.ValidationError", MODEL)
llm = vllm.LLM(model=MODEL, trust_remote_code=True, enforce_eager=True, gpu_memory_utilization=0.5)
outputs = llm.generate(["What is 2 + 2?"], vllm.SamplingParams(temperature=0.7, max_tokens=64))
for output in outputs:
    logger.info("Generated: %s", output.outputs[0].text)
