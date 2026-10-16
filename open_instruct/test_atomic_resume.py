import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from olmo_core.optim import LinearWithWarmup
from torch.distributed import checkpoint as distributed_checkpoint

from open_instruct import atomic_resume


def trainer_state():
    return {
        "global_step": 512,
        "global_train_tokens_seen": 536870912,
        "max_steps": 512,
        "world_size": 16,
        "data_loader": {"batches_processed": 512, "seed": 34521, "fingerprint": "native"},
        "rng": {"torch": torch.arange(5, dtype=torch.uint8), "numpy": np.arange(5, dtype=np.uint32)},
        "callbacks": {"atomic_audit": {"mode": "atomic-safe", "horizon": 11768}, "promoted_row_step_check": {}},
    }


def test_only_duration_and_empty_callback_changes_are_allowed():
    saved = trainer_state()
    actual = copy.deepcopy(saved)
    actual["max_steps"] = 11768
    del actual["callbacks"]["promoted_row_step_check"]
    atomic_resume.verify_trainer_state(saved, actual)


@pytest.mark.parametrize("defect", ["rng", "numpy_rng", "data", "step", "world", "callback", "duration"])
def test_resume_rejects_lost_or_changed_training_state(defect):
    saved = trainer_state()
    actual = copy.deepcopy(saved)
    actual["max_steps"] = 11768
    if defect == "rng":
        actual["rng"]["torch"][0] += 1
    elif defect == "numpy_rng":
        actual["rng"]["numpy"][0] += 1
    elif defect == "data":
        actual["data_loader"]["batches_processed"] = 0
    elif defect == "step":
        actual["global_step"] = 0
    elif defect == "world":
        actual["world_size"] = 8
    elif defect == "callback":
        del actual["callbacks"]["atomic_audit"]
    else:
        actual["max_steps"] = 512
    with pytest.raises(ValueError):
        atomic_resume.verify_trainer_state(saved, actual)


def test_next_lr_uses_restored_step_without_mutating_optimizer():
    group = {"lr": torch.tensor(5e-5), "initial_lr": 5e-5}
    scheduler = LinearWithWarmup(warmup=353, t_max=11768, alpha_f=0.0)
    module = SimpleNamespace(scheduler=scheduler, optim=SimpleNamespace(param_groups=[group]))
    trainer = SimpleNamespace(global_step=512, max_steps=11768)
    before = group["lr"].clone()
    value = atomic_resume.verify_next_lr(module, trainer, 11768)[0]
    assert value == pytest.approx(5e-5 * (1 - 160 / 11415))
    assert torch.equal(group["lr"], before)
    scheduler.t_max = 512
    with pytest.raises(ValueError, match="LR schedule"):
        atomic_resume.verify_next_lr(module, trainer, 11768)


def test_native_auxiliary_load_and_master_corruption_detection(tmp_path, monkeypatch):
    def digest(value):
        return hashlib.sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()

    model = {f"p{i}": torch.tensor([0.10001]) for i in range(415)}
    states = {
        f"{name}.{suffix}": value.clone()
        for name, value in model.items()
        for suffix in ("main", "exp_avg", "exp_avg_sq", "step")
    }
    checkpoint = tmp_path / "control" / "step512"
    audit_dir = checkpoint.parent / "atomic-audit"
    audit_dir.mkdir(parents=True)
    (checkpoint / "train").mkdir(parents=True)
    (audit_dir / "state-step512-rank0.json").write_text(
        json.dumps(
            {
                "step": 512,
                "model": {k: digest(v) for k, v in model.items()},
                "optimizer": {k: digest(v) for k, v in states.items()},
            }
        )
    )
    aux = {"__moe_skip_step_losses": [1.0, 2.0], "__moe_skip_step_grad_norms": [3.0]}
    distributed_checkpoint.save(aux, checkpoint_id=checkpoint / "model_and_optim")
    native = trainer_state()
    native["callbacks"] = {"atomic_audit": {"mode": "control", "horizon": 11768, "offline": True}}
    torch.save(native, checkpoint / "train/rank0.pt")
    current = copy.deepcopy(native)
    current["max_steps"] = 11768
    optim = SimpleNamespace(
        states=states,
        param_groups=[{"lr": torch.tensor(5e-5), "initial_lr": 5e-5}],
        LOSSES_STATE_DICT_KEY="__moe_skip_step_losses",
        GRAD_NORMS_STATE_DICT_KEY="__moe_skip_step_grad_norms",
        _losses=[torch.tensor(1.0), torch.tensor(2.0)],
        _grad_norms=[torch.tensor(3.0)],
    )
    module = SimpleNamespace(
        model=SimpleNamespace(state_dict=lambda: model),
        optim=optim,
        scheduler=LinearWithWarmup(warmup=353, t_max=11768, alpha_f=0.0),
    )
    trainer = SimpleNamespace(
        state_dict=lambda: copy.deepcopy(current),
        train_module=module,
        global_step=512,
        max_steps=11768,
        data_loader={512: {key: torch.tensor([1]) for key in ("input_ids", "label_mask", "doc_lens", "index")}},
    )
    monkeypatch.setattr(atomic_resume, "get_rank", lambda: 0)
    monkeypatch.setattr(atomic_resume, "get_world_size", lambda: 16)
    result = atomic_resume.verify(trainer, checkpoint, "control", 11768, digest)
    assert result["optimizer_auxiliary_state_exact"] and result["next_step"] == 513
    original = states["p0.main"].clone()
    states["p0.main"].copy_(states["p0.main"].bfloat16().float())
    with pytest.raises(ValueError, match="FP32 masters"):
        atomic_resume.verify(trainer, checkpoint, "control", 11768, digest)
    states["p0.main"].copy_(original)
    optim._losses = []
    with pytest.raises(ValueError, match="rolling optimizer"):
        atomic_resume.verify(trainer, checkpoint, "control", 11768, digest)
