"""Compare Core with independently generated release-era Olmo 3 reference logits."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from olmo_core.nn.hf import convert
from transformers import AutoModelForCausalLM

from open_instruct.miles import models
from open_instruct.miles.config import CoreConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.reference / "reference.json").read_text())
    if metadata["transformers"] != "4.57.0":
        raise ValueError("Reference must come from release-era Transformers 4.57.0")
    torch.set_num_threads(2)
    reference = AutoModelForCausalLM.from_pretrained(args.reference / "hf", torch_dtype=torch.bfloat16)
    hf = reference.config
    cfg = models.model_config_from_hf(hf, CoreConfig(attention_backend="torch"))
    model = cfg.build(init_device="cpu").eval()
    model.load_state_dict(convert.convert_state_from_hf(hf, reference.state_dict(), model_type="olmo3"), strict=True)
    results = []
    for record in torch.load(args.reference / "reference.pt", weights_only=True):
        with torch.no_grad():
            actual = model(record["tokens"]).float()
        expected = record["logits"]
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError("Olmo 3 reference comparison contains non-finite logits")
        error = actual - expected
        relative_l2 = (error.norm() / expected.norm()).item()
        max_abs = error.abs().max().item()
        # BF16 forwards through independently implemented kernels are not bit exact.
        # Also require agreement at the distribution level, without unstable near-zero ratios.
        actual_lp, expected_lp = actual.log_softmax(-1), expected.log_softmax(-1)
        mean_lp_error = (actual_lp - expected_lp).abs().mean().item()
        if relative_l2 > 0.025 or max_abs > 0.08 or mean_lp_error > 0.015:
            raise AssertionError(
                f"Olmo 3 reference mismatch: length={actual.shape[1]} rel_l2={relative_l2} max_abs={max_abs} logprob_mae={mean_lp_error}"
            )
        results.append(
            {
                "length": actual.shape[1],
                "relative_l2": relative_l2,
                "max_abs": max_abs,
                "logprob_mean_abs": mean_lp_error,
            }
        )
    exported = models.export_state(SimpleNamespace(model=model, _miles_model_backend="standard"), hf)
    assert exported.keys() == reference.state_dict().keys()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(exported[name], value, rtol=0, atol=0)
    result = {"reference": metadata, "core": results, "hf_weight_roundtrip_exact": True}
    (args.reference / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": results, "hf_weight_roundtrip_exact": True}))


if __name__ == "__main__":
    main()
