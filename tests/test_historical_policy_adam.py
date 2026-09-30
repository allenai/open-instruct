"""CPU construction and moment checks for an explicitly authorized legacy reproduction."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "open_instruct"
FIELDS = ("policy_adam_beta1", "policy_adam_beta2", "policy_adam_eps", "policy_adam_weight_decay")


@pytest.fixture
def controls():
    tree = ast.parse((SOURCE / "grpo_utils.py").read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "policy_adam_kwargs")
    env = {"Any": object, "math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "historical-native-optimizer-helper", "exec"), env)
    return env[fn.name]


def declared(**changes):
    values = dict(zip(FIELDS, (0.9, 0.95, 1e-8, 0.0), strict=True))
    return SimpleNamespace(**(values | changes))


def test_absent_and_explicitly_unset_preserve_installed_defaults(controls):
    assert controls(SimpleNamespace()) == {}
    assert controls(SimpleNamespace(**dict.fromkeys(FIELDS))) == {}
    p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    default = torch.optim.AdamW([p], lr=1e-6, **controls(SimpleNamespace()))
    assert default.defaults["betas"] == (0.9, 0.999)
    assert default.defaults["eps"] == 1e-8
    assert default.defaults["weight_decay"] == 0.01


@pytest.mark.parametrize(
    "change",
    [
        {"policy_adam_beta1": None},
        {"policy_adam_beta2": 1.0},
        {"policy_adam_beta1": -0.1},
        {"policy_adam_beta2": float("nan")},
        {"policy_adam_eps": float("inf")},
        {"policy_adam_eps": 0.0},
        {"policy_adam_weight_decay": -1.0},
        {"policy_adam_beta1": True},
        {"policy_adam_beta2": "0.95"},
    ],
)
def test_incomplete_or_invalid_override_rejected(controls, change):
    with pytest.raises(ValueError):
        controls(declared(**change))


@pytest.mark.parametrize("offload", [False, True])
def test_actual_policy_constructor_passes_complete_override(controls, offload):
    # Execute the real constructor's optimizer section; no DeepSpeed/CUDA import.
    tree = ast.parse((SOURCE / "grpo_fast.py").read_text())
    blocks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for index, item in enumerate(node.body):
                if isinstance(item, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "policy_adam" for target in item.targets
                ):
                    blocks = node.body[index : index + 2]
    assert len(blocks) == 2 and isinstance(blocks[1], ast.If)
    calls = []

    def constructor(params, **kwargs):
        calls.append((params, kwargs))
        return object()

    args = declared(learning_rate=1e-6, fused_optimizer=False, deepspeed_offload_optimizer=offload)
    env = {
        "args": args,
        "self": SimpleNamespace(),
        "optim_params": "parameters",
        "grpo_utils": SimpleNamespace(policy_adam_kwargs=controls),
        "DeepSpeedCPUAdam": constructor,
        "torch": SimpleNamespace(optim=SimpleNamespace(AdamW=constructor)),
    }
    exec(compile(ast.Module(body=blocks, type_ignores=[]), "native-policy-optimizer-call", "exec"), env)
    assert len(calls) == 1
    assert calls[0][1] == ({"lr": 1e-6, **controls(args)} | ({} if offload else {"fused": False}))


@pytest.mark.parametrize("decay", [0.0, 0.04])
def test_three_cpu_adam_updates_match_independent_moment_oracle(controls, decay):
    # Unequal gradients expose beta2/epsilon and decay, rather than checking only declarations.
    params = controls(declared(policy_adam_eps=2e-5, policy_adam_weight_decay=decay))
    p = torch.nn.Parameter(torch.tensor([1.25], dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=0.003, **params)
    expected, first, second = 1.25, 0.0, 0.0
    beta1, beta2 = params["betas"]
    for step, gradient in enumerate((0.2, -0.7, 0.4), start=1):
        p.grad = torch.tensor([gradient], dtype=torch.float64)
        opt.step()
        first = beta1 * first + (1 - beta1) * gradient
        second = beta2 * second + (1 - beta2) * gradient**2
        expected *= 1 - 0.003 * decay
        expected -= 0.003 * (first / (1 - beta1**step)) / (math.sqrt(second / (1 - beta2**step)) + params["eps"])
        assert float(p.detach()[0]) == pytest.approx(expected, abs=1e-14)
        state = opt.state[p]
        assert float(state["exp_avg"][0]) == pytest.approx(first)
        assert float(state["exp_avg_sq"][0]) == pytest.approx(second)


def test_config_defaults_and_validation_are_opt_in():
    tree = ast.parse((SOURCE / "grpo_utils.py").read_text())
    config = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GRPOExperimentConfig")
    fields = {node.target.id: node.value for node in config.body if isinstance(node, ast.AnnAssign)}
    assert all(isinstance(fields[name], ast.Constant) and fields[name].value is None for name in FIELDS)
    post = next(node for node in config.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    assert ast.unparse(post.body[0]) == "policy_adam_kwargs(self)"


def test_parameter_group_decay_must_agree_with_explicit_override(controls):
    with pytest.raises(ValueError, match="parameter-group decay"):
        controls(declared(set_weight_decay_on_bias_and_norm=True, weight_decay=0.02))
    assert controls(declared(set_weight_decay_on_bias_and_norm=True, weight_decay=0.0))["weight_decay"] == 0.0
