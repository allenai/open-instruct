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


aime_pass_module = load_module("scripts/data/rlvr/aime_pass_at_k_dataset.py", "aime_pass_at_k_dataset")
aime_quartiles_module = load_module("scripts/data/rlvr/create_aime_pass_rate_quartiles.py", "aime_pass_rate_quartiles")


def test_extract_generation_messages_prefers_prompt_field():
    sample = {
        "prompt": "Solve x^2 = 1",
        "messages": [
            {"role": "user", "content": "stale user prompt"},
            {"role": "assistant", "content": "stale answer"},
        ],
    }

    assert aime_pass_module.extract_generation_messages(sample) == [{"role": "user", "content": "Solve x^2 = 1"}]


def test_extract_generation_messages_strips_assistant_reference_answer():
    sample = {
        "messages": [{"role": "user", "content": "Find 2+2"}, {"role": "assistant", "content": "The answer is 4"}]
    }

    assert aime_pass_module.extract_generation_messages(sample) == [{"role": "user", "content": "Find 2+2"}]


def test_build_output_rows_sets_pass_metadata_and_sanitized_messages():
    sample_ds = aime_pass_module.Dataset.from_list(
        [{"messages": [{"role": "user", "content": "Find 2+2"}], "ground_truth": "4", "dataset": "math"}]
    )
    generation_messages = [[{"role": "user", "content": "Find 2+2"}]]
    completions_by_prompt = [["work\n\\boxed{4}", "work\n\\boxed{5}"]]
    args = aime_pass_module.argparse.Namespace(
        num_samples=2,
        model="allenai/Olmo-3-1025-7B",
        chat_template="olmo_thinker_rlzero",
        temperature=1.0,
        top_p=1.0,
        max_tokens=32768,
    )

    rows = aime_pass_module.build_output_rows(sample_ds, generation_messages, completions_by_prompt, args, "test_2024")

    assert rows[0]["messages"] == generation_messages[0]
    assert rows[0]["pass_count"] == 1
    assert rows[0]["pass_rate"] == "1/2"
    assert rows[0]["source_split"] == "test_2024"


def test_split_to_dataset_name_preserves_year():
    assert aime_quartiles_module.split_to_dataset_name("test_2024", 0) == "math_aime_2024_quartile0"
    assert aime_quartiles_module.split_to_dataset_name("test_2025", 3) == "math_aime_2025_quartile3"
