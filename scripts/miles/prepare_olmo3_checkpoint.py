"""Download the pinned pre-RL Olmo 3 snapshot on WEKA and verify GSM8K preparation."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from scripts.miles import stage_olmo3

from open_instruct.ground_truth_utils import GSM8KVerifier
from open_instruct.miles import run_data, workflow
from open_instruct.miles.run_spec import RunSpec

MODEL = "allenai/Olmo-3-7B-Think-DPO"
REVISION = "7b18bf927b430ff06376fdfa5610eb3b1b6a5c38"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    spec = RunSpec.from_dict(json.loads(args.config.read_text()), config_path=args.config)
    target = Path(spec.model["source"])
    raw = target.parent / f"olmo3-7b-think-dpo-{REVISION}"
    snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=raw,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "LICENSE*", "README.md"],
    )
    index = json.loads((raw / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(not (raw / name).is_file() or not (raw / name).stat().st_size for name in shards):
        raise ValueError("Pinned snapshot is missing one or more indexed weight shards")
    if target.exists():
        marker = json.loads((target / "olmo3-staging.json").read_text())
        if marker["source"] != workflow.model_identity(raw):
            raise ValueError("Existing staged checkpoint differs from the pinned snapshot")
        if (target / "chat_template.jinja").read_text() != stage_olmo3.TEMPLATE.read_text():
            raise ValueError("Existing staged checkpoint has a different chat template")
    else:
        stage_olmo3.stage(raw, target)
    hf = workflow.prepare_model(spec)
    config = spec.compile()
    prepared = run_data.prepare_data(
        spec.data,
        Path(hf),
        Path(spec.output["root"]) / "prepared/data",
        max_prompt_length=config.miles["rollout_max_prompt_len"],
        seed=config.miles["seed"],
    )
    paths = [prepared["prompt_data"], *prepared["eval_prompt_data"][1::2]]
    verifier = GSM8KVerifier()
    count = 0
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if verifier([], f"The answer is {row['label']}", row["label"]).score != 1:
                raise ValueError("Positive GSM8K reward canary failed")
            if verifier([], "No answer is available.", row["label"]).score != 0:
                raise ValueError("Negative GSM8K reward canary failed")
            count += 2
    report = dict(
        model=MODEL,
        revision=REVISION,
        source=str(raw),
        staged=str(target),
        shards=shards,
        weight_bytes=sum((raw / name).stat().st_size for name in shards),
        reward_canaries=count,
        prepared=prepared,
    )
    workflow.write_json(Path(spec.output["root"]) / "checkpoint-preparation.json", report)
    workflow.write_json("/output/checkpoint-preparation.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
