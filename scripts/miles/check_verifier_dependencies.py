"""Offline build-time checks for task graders shipped in the MILES image."""

from importlib import metadata

from code_execution_client import livecodebench, verifier
from rdkit import Chem

from open_instruct.miles.rewards import reasoning_gym, structured_outputs


def main():
    assert metadata.version("reasoning-gym") == "0.1.25"
    assert metadata.version("rdkit") == "2025.9.3"
    assert Chem.MolFromSmiles("CCO") is not None
    gym = reasoning_gym.ReasoningGymVerifier()
    target = {"question": "What is 20 + 22?", "answer": "42", "metadata": {"source_dataset": "basic_arithmetic"}}
    assert gym([], "42", target).score == 1
    assert gym([], "43", target).score == 0
    structured = structured_outputs.StructuredOutputVerifier()
    target = {
        "format": "xml",
        "schema_policy": "as_declared",
        "schema": {"type": "object", "properties": {"answer": {"const": "42"}}, "required": ["answer"]},
    }
    assert structured([], "<answer>42</answer>", target).score == 1
    assert structured([], "<answer>43</answer>", target).score == 0
    assert callable(verifier.CodeVerifier)
    mode, tests = livecodebench.prepare_nemotron_tests(["1\n"], ["2\n"])
    assert mode == "stdio" and tests == [{"input": "1\n", "output": "2\n", "comparison": "nemo_v1"}]


if __name__ == "__main__":
    main()
