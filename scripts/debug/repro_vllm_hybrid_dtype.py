"""Minimal repro for vLLM hybrid model dtype serialization bug.

vllm 0.18.0 crashes when generating with allenai/Olmo-Hybrid-Instruct-DPO-7B:
  msgspec.ValidationError: Expected `array` of length 1, got 2 - at `$.dtypes`

The hybrid model has two dtypes (transformer + SSM layers) but vLLM's
MambaSpec serialization expects exactly one.

Run on a GPU node:
  python scripts/debug/repro_vllm_hybrid_dtype.py
"""

import vllm

if __name__ == "__main__":
    llm = vllm.LLM(
        model="allenai/Olmo-Hybrid-Instruct-DPO-7B",
        enforce_eager=True,
    )
    output = llm.generate(["Hello, world!"], vllm.SamplingParams(max_tokens=16))
    print(output[0].outputs[0].text)
