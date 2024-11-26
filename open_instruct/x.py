from typing import Optional

from hf_olmo import *
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

from open_instruct.olmo_adapter.olmo_new import OlmoNewForCausalLM


class Args:
    model_name_or_path: str = "/net/nfs.cirrascale/allennlp/akshitab/model-checkpoints/peteish7/step11931-unsharded-hf"
    trust_remote_code: bool = True
    revision: Optional[str] = None


# def main(args: Args):
#     model = AutoModelForCausalLM.from_pretrained(
#         args,
#         trust_remote_code=True,
#     )


if __name__ == "__main__":
    # instead of installing from source, https://github.com/AkshitaB/vllm/blob/c96643ec56da3ab8cefba03cadf7731788e756b5/vllm/model_executor/models/__init__.py#L49
    # here we just register the new model class
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model("OLMoForCausalLM", OlmoNewForCausalLM)
    from vllm import LLM, SamplingParams

    model = AutoModelForCausalLM.from_pretrained(
        "/net/nfs.cirrascale/allennlp/akshitab/model-checkpoints/peteish7/step11931-unsharded-hf",
        trust_remote_code=True,
    )
    from vllm.model_executor.models import ModelRegistry

    from open_instruct.olmo_adapter.modeling_olmo2 import OLMoForSequenceClassification
    from open_instruct.olmo_adapter.olmo_new import OlmoNewForCausalLM

    AutoModelForSequenceClassification.register(OLMoConfig, OLMoForSequenceClassification)

    s = SamplingParams(temperature=0.0)
    llm = LLM(
        model="/net/nfs.cirrascale/allennlp/akshitab/model-checkpoints/peteish7/step11931-unsharded-hf",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )

    vllm_out = llm.generate(["How is the weather today"], sampling_params=s)
    print(vllm_out[0].outputs[0].text)
