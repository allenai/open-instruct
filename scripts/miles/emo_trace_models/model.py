"""SGLang entry point for the diagnostic-only trace package."""

from olmo_sglang.models import olmo3_moe
from scripts.miles.emo_trace_models import TraceMixin


class Olmo3MoeForCausalLM(TraceMixin, olmo3_moe.Olmo3MoeForCausalLM):
    pass


EntryClass = Olmo3MoeForCausalLM
