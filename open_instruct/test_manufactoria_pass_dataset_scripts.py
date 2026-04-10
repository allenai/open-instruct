import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_module(relative_path: str, module_name: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


manufactoria_pass_module = load_module(
    "scripts/data/rlvr/manufactoria_pass_at_k_dataset.py", "manufactoria_pass_at_k_dataset"
)


def test_compute_max_model_len_adds_max_prompt_token_length_and_response_length():
    assert manufactoria_pass_module.compute_max_model_len(max_prompt_token_length=2048, response_length=8192) == 10240
