"""Fixed unhinted panels, initial length-cap checks and final full evaluation."""

import copy
import dataclasses
import json
from pathlib import Path

from open_instruct.miles.rollout import bread


def evaluation_state(state):
    """Allow evaluation to use serving capacity without widening training samples."""
    result = copy.copy(state)
    result.args = copy.copy(state.args)
    result.args.rollout_max_context_len = state.args.sglang_context_length or state.args.rollout_max_context_len
    return result


def mark_final(args, rollout_id):
    root = Path(args.save) / "bread"
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "final-eval.tmp"
    temporary.write_text(json.dumps({"rollout_id": rollout_id}))
    temporary.replace(root / "final-eval.json")


async def evaluate(args, input, run_native, *, initial):
    root = Path(args.save) / "bread"
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "final-eval.json"
    final = marker.exists() and json.loads(marker.read_text())["rollout_id"] == input.rollout_id
    output, metrics = {}, {}
    for config in args.eval_datasets:
        rows = Path(config.path).read_text().splitlines()
        panel_size = min(args.olmo_core.bread_eval_panel_size, len(rows))
        panel_file = root / f"eval-panel-{config.name}.jsonl"
        payload = "\n".join(rows[:panel_size]) + "\n"
        if panel_file.exists() and panel_file.read_text() != payload:
            raise ValueError("BREAD evaluation panel changed within a run")
        panel_file.write_text(payload)
        panel_name = f"{config.name}_panel"
        full_name = f"{config.name}_full"
        selected = dataclasses.replace(
            config,
            name=full_name if initial or final else panel_name,
            path=config.path if initial or final else str(panel_file),
        )
        result = await run_native(selected)
        output.update(result)
        if initial or final:
            count = panel_size * config.n_samples_per_eval_prompt
            output[panel_name] = {key: value[:count] for key, value in result[full_name].items()}
        if initial and args.olmo_core.bread_eval_long_response_len:
            long_config = dataclasses.replace(
                config,
                name=f"{config.name}_panel_long",
                path=str(panel_file),
                max_response_len=args.olmo_core.bread_eval_long_response_len,
            )
            output.update(await run_native(long_config))
    for name, result in output.items():
        descriptions = bread.responses(result["samples"], args.eval_reward_key or args.reward_key)
        finished = [r for r in descriptions if r["status"] == "completed"]
        metrics[f"eval/{name}/finished_count"] = len(finished)
        metrics[f"eval/{name}/accuracy_finished"] = (
            sum(r["reward"] for r in finished) / len(finished) if finished else 0.0
        )
        with (root / "eval-responses.jsonl").open("a") as stream:
            for sample, description in zip(result["samples"], descriptions, strict=True):
                stream.write(
                    json.dumps(
                        {
                            "rollout_id": input.rollout_id,
                            "initial": initial,
                            "final": final,
                            "dataset": name,
                            "sample_id": sample.metadata.get("prepared_sample_id"),
                            "sample_index": sample.index,
                            **description,
                        }
                    )
                    + "\n"
                )
    return output, metrics
