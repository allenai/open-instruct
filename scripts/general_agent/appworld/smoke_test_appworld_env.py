"""End-to-end smoke test for AppWorldEnv (requires Docker + the AppWorld image + data).

Drives a single AppWorld task through the env the way the rollout loop would:
reset -> a few execute_python steps -> completion -> reward. This starts a real
AppWorld container and talks to it over HTTP, so it validates the full container +
HTTP path without importing the (pydantic-1) appworld package in this process.

Prereqs:
    docker pull ghcr.io/stonybrooknlp/appworld:latest
    # AppWorld data root on the host (contains data/ and experiments/outputs/):
    #   pip install appworld && appworld install && appworld download data --root $APPWORLD_ROOT

Example:
    uv run python scripts/general_agent/appworld/smoke_test_appworld_env.py --data_root $APPWORLD_ROOT
"""

import argparse
import asyncio
import os

from open_instruct.environments.appworld_env import AppWorldEnv
from open_instruct.environments.base import EnvCall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, help="Host AppWorld root (has data/ and experiments/outputs/).")
    parser.add_argument("--split", default="train", help="Split file under data/datasets to pick a task from.")
    parser.add_argument("--task_id", default="", help="Specific task id; default = first id in the split file.")
    parser.add_argument("--image", default="ghcr.io/stonybrooknlp/appworld:latest", help="AppWorld docker image.")
    return parser.parse_args()


def _first_task_id(data_root: str, split: str) -> str:
    split_path = os.path.join(data_root, "data", "datasets", f"{split}.txt")
    with open(split_path, encoding="utf-8") as fh:
        for line in fh:
            tid = line.strip().split(":")[-1]  # strip optional tag prefix
            if tid:
                return tid
    raise RuntimeError(f"No task ids found in {split_path}")


def _call(code: str) -> EnvCall:
    return EnvCall(id="0", name="execute_python", args={"code": code})


async def run(args: argparse.Namespace) -> None:
    os.makedirs(os.path.join(args.data_root, "experiments", "outputs"), exist_ok=True)
    task_id = args.task_id or _first_task_id(args.data_root, args.split)
    print(f"Using task_id={task_id}")

    env = AppWorldEnv(image=args.image, data_root=args.data_root, max_interactions=10)
    await env.setup()
    try:
        result, tools = await env.reset(task_id=task_id)
        print(f"reset -> tools={[t['function']['name'] for t in tools]}, base_url={env._base_url}")

        # Probe the API surface, then complete the task (no real solution; this just
        # exercises execute -> task_completed -> evaluate -> reward end to end).
        for code in [
            "print(apis.api_docs.show_app_descriptions())",
            "print(apis.supervisor.show_profile())",
            "apis.supervisor.complete_task()",
        ]:
            step = await env.step(_call(code))
            head = step.result if len(step.result) < 600 else step.result[:600] + " ...[truncated]"
            print(f"\n>>> {code}\n{head}\n[reward={step.reward} done={step.done}]")
            if step.done:
                break

        print(f"\nmetrics={env.get_metrics()}")
    finally:
        await env.close()
    print("Smoke test complete.")


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
