"""Native EP stress gate: unequal token counts and active global clipping.

Captures canonical FP32 gradients on both sides of the actual optimizer clip.
The independent checks count each parameter once, verify Adam moments, and
compare every rank's reconstructed state before comparing EP1 with EP2.
"""

import argparse
import json
import math
from pathlib import Path

import ep_contract
import torch


def inspect(evidence):
    before, after, state = (evidence[key] for key in ("before", "after", "state"))
    if not before or set(before) != set(after):
        raise ValueError("Missing or mismatched gradient inventory")
    expected_state = {
        name.removesuffix(".grad") + suffix for name in before for suffix in (".main", ".exp_avg", ".exp_avg_sq")
    }
    if set(state) != expected_state:
        raise ValueError("Optimizer and gradient inventories differ")
    values = [*before.values(), *after.values(), *state.values()]
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("Non-finite gradient or optimizer state")
    norm = math.sqrt(sum(float(value.double().square().sum()) for value in before.values()))
    reported = evidence["norm"]
    if not math.isfinite(reported) or not math.isclose(norm, reported, rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError(f"Global gradient norm differs: independently {norm}, optimizer {reported}")
    clip = evidence["clip_grad"]
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError("Invalid clipping threshold")
    factor = min(1.0, clip / (reported + 1e-6))
    beta1, beta2 = evidence["betas"]
    for name, gradient in before.items():
        clipped = after[name]
        if gradient.shape != clipped.shape:
            raise ValueError(f"Gradient shape differs: {name}")
        torch.testing.assert_close(clipped, gradient * factor, rtol=2e-6, atol=1e-9)
        prefix = name.removesuffix(".grad")
        torch.testing.assert_close(state[prefix + ".exp_avg"], clipped * (1 - beta1), rtol=2e-6, atol=1e-9)
        torch.testing.assert_close(
            state[prefix + ".exp_avg_sq"], clipped.square() * (1 - beta2), rtol=3e-6, atol=1e-12
        )
    return {
        "independent_grad_norm": norm,
        "optimizer_grad_norm": reported,
        "clip_factor": factor,
        "parameter_tensors": len(before),
    }


def compare(root):
    report = {
        "passed": False,
        "scope": "Native tiny EP1/EP2 token reduction and active clipping; not Megatron parity",
        "arms": [],
        "failures": [],
    }
    for variant in ("token", "clipped"):
        folder = root / variant
        loaded = {}
        records = []
        for world in (1, 2):
            for rank in range(world):
                try:
                    evidence = torch.load(folder / f"stress-ep{world}-rank{rank}.pt", weights_only=True)
                    if (
                        evidence["world"] != world
                        or evidence["rank"] != rank
                        or evidence["token_average"] != (variant == "token")
                    ):
                        raise ValueError("Arm identity or reduction differs")
                    measured = inspect(evidence)
                    if variant == "clipped" and not 0 < measured["clip_factor"] < 0.5:
                        raise ValueError("Clipping was not substantially active")
                    if variant == "token" and measured["clip_factor"] != 1.0:
                        raise ValueError("Token reduction arm unexpectedly clipped")
                    loaded[world, rank] = evidence
                    records.append(dict(world=world, rank=rank, **measured))
                except (OSError, RuntimeError, ValueError, AssertionError, KeyError) as exc:
                    report["failures"].append(f"{variant}/EP{world}/rank{rank}: {exc}")
        comparisons = []
        if len(loaded) == 3:
            # Reconstructed replicated and sharded states must agree on both EP ranks.
            for section in ("before", "after", "state"):
                for name, value in loaded[2, 0][section].items():
                    if not torch.equal(value, loaded[2, 1][section][name]):
                        report["failures"].append(f"{variant}: reconstructed rank disagreement in {section}/{name}")
                errors, error = ep_contract._state_errors(loaded[1, 0][section], loaded[2, 0][section])
                comparisons.append(dict(section=section, errors=errors, error=error))
                if error or any(value["relative_l2_error"] >= 0.05 for value in errors.values()):
                    report["failures"].append(f"{variant}: EP1/EP2 {section} disagreement")
        report["arms"].append(dict(variant=variant, ranks=records, comparisons=comparisons))
    report["passed"] = not report["failures"]
    (root / "ep-stress.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passed"]:
        raise AssertionError("; ".join(report["failures"]))
    print("EP_STRESS_PASSED", json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bootstrap", "run", "compare"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--variant", choices=("token", "clipped"), default="token")
    args = parser.parse_args()
    if args.command == "bootstrap":
        for variant in ("token", "clipped"):
            ep_contract.bootstrap(args.root / variant)
    elif args.command == "run":
        ep_contract.run(
            args.root / args.variant,
            "combined",
            False,
            token_average=args.variant == "token",
            clip_grad=1e-3 if args.variant == "clipped" else 1e9,
            capture_gradients=True,
        )
    else:
        compare(args.root)


if __name__ == "__main__":
    main()
