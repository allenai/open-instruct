"""Read-only audit of frozen judge inputs, using the judge's own tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from open_instruct.miles import general_judge
from open_instruct.miles.run_spec import RunSpec


def audit(spec):
    service = spec.judges["judges"]["general"]
    prepared = json.loads((Path(service["prepared_dir"]) / "prepared.json").read_text())
    snapshot = Path(prepared["snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.chat_template = Path(prepared["template"]).read_text()
    if "audit canary" not in tokenizer.apply_chat_template(
        [{"role": "user", "content": "audit canary"}], tokenize=False
    ):
        raise ValueError("Judge template discarded the input prompt")
    report = {"model_config": json.loads((snapshot / "config.json").read_text()), "files": {}}
    paths = [spec.data["prompt_data"], *spec.data["eval_prompt_data"][1::2]]
    for path in paths:
        counts = []
        with Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                meta = row["metadata"]
                for verifier in meta["verifiers"]:
                    if verifier["name"] not in ("general-quality", "general-quality_ref"):
                        continue
                    prompt, _ = general_judge.build_judge_prompt(
                        verifier["name"], query=meta["judge_query"], prediction="", target=verifier["target"]
                    )
                    tokens = len(
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=False,
                        )
                    )
                    counts.append(
                        {
                            "sample_id": meta.get("prepared_sample_id"),
                            "verifier": verifier["name"],
                            "static_prompt_tokens": tokens,
                            "reference_tokens": len(
                                tokenizer.encode(str(verifier["target"]), add_special_tokens=False)
                            ),
                            "query_tokens": len(tokenizer.encode(meta["judge_query"], add_special_tokens=False)),
                        }
                    )
        report["files"][path] = {
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "judged_rows": len(counts),
            "largest_static_prompts": sorted(counts, key=lambda r: r["static_prompt_tokens"], reverse=True)[:10],
            "static_prompt_alone_over_budget": sum(
                r["static_prompt_tokens"] + 2048 > service["max_context_length"] for r in counts
            ),
        }
        print(json.dumps({"path": path, **report["files"][path]}), flush=True)
    Path("/output/judge-budget.json").write_text(json.dumps(report, indent=2))
    print("JUDGE_BUDGET_AUDIT_COMPLETED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    audit(RunSpec.load(parser.parse_args().config))
