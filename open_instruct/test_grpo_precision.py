"""Exercise GRPO's actual forward contexts without requiring CUDA or FSDP."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from olmo_core.train.train_module import transformer

from open_instruct import grpo_utils, olmo_core_train_modules
from open_instruct.test_olmo_core_train_modules import _make_batch_data


class ForwardChecked(Exception):
    pass


@pytest.mark.parametrize("forward_path", ["reference", "old_policy", "learner"])
def test_grpo_autocasts_fp32_parameters(monkeypatch, forward_path):
    def init_base(self, **kwargs):
        self.model = kwargs["model"]
        self.device = kwargs["device"]
        self.autocast_precision = kwargs["autocast_precision"]
        self.rank_microbatch_size = kwargs["rank_microbatch_size"]
        self._trainer = SimpleNamespace(dp_process_group=None)

    monkeypatch.setattr(transformer.TransformerTrainModule, "__init__", init_base)
    config = SimpleNamespace(
        model_dtype="bfloat16",
        load_ref_policy=forward_path == "reference",
        num_mini_batches=2 if forward_path == "old_policy" else 1,
        use_vllm_logprobs=False,
        loss_denominator=1,
        record_entropy=False,
        num_epochs=1,
    )
    model = torch.nn.Linear(4, 4)
    module = olmo_core_train_modules.GRPOTrainModule(
        model=model,
        optim=Mock(),
        sample_microbatch_size=1,
        max_sequence_length=5,
        grpo_config=config,
        temperature=1.0,
        tokenizer=SimpleNamespace(pad_token_id=0),
        streaming_config=Mock(),
        attn_implementation="flash_2",
        ref_policy=torch.nn.Linear(4, 4) if config.load_ref_policy else None,
        device=torch.device("cpu"),
        compile_model=False,
    )

    def check_forward(policy, *args, **kwargs):
        assert torch.is_autocast_enabled("cpu")
        assert next(policy.parameters()).dtype == torch.float32
        assert policy(torch.ones(1, 4)).dtype == torch.bfloat16
        raise ForwardChecked

    monkeypatch.setattr(grpo_utils, "compute_logprobs", check_forward)
    monkeypatch.setattr(grpo_utils, "forward_for_logprobs", check_forward)
    with pytest.raises(ForwardChecked):
        module.train_batch({"batch": _make_batch_data(num_samples=1)}, dry_run=True)
    assert model.weight.dtype == torch.float32
