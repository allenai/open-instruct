"""CPU integration probe in each pinned image; HTTP response is explicitly fake."""

import asyncio
import importlib
import json
from pathlib import Path

from aiohttp import web
from scripts.miles import benchmark_kda_rollout as benchmark


async def main():
    args = benchmark.native_args("/tmp/plan.json", Path("/tmp/out"))
    ray = importlib.import_module("ray")
    http = importlib.import_module("miles.utils.http_utils")
    ray.init(num_cpus=24, num_gpus=0, include_dashboard=False, log_to_driver=False)
    http.init_http_client(args)
    seen = []

    async def generate(request):
        payload = await request.json()
        assert payload["return_logprob"] and payload["top_logprobs_num"] == 20
        assert payload["return_routed_experts"]
        seen.append(payload)
        n = payload["sampling_params"]["max_new_tokens"]
        return web.json_response(
            {
                "text": "\\boxed{0}",
                "meta_info": {
                    "finish_reason": {"type": "length"},
                    "completion_tokens": n,
                    "prompt_tokens": len(payload["input_ids"]),
                    "output_token_logprobs": [[-1.0, 15, "0"]] * n,
                    "output_top_logprobs": [[[-1.0, i, str(i)] for i in range(20)]] * n,
                },
            }
        )

    app = web.Application()
    app.router.add_post("/generate", generate)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 31000).start()
    rows = json.loads(Path("/tmp/inputs.json").read_text())
    try:
        result = await benchmark.measure(args, rows, Path("/tmp/out/contract.json"), length=2, repeat=0)
        assert result["responses"] == 64 and result["output_tokens"] == 128
        assert len(seen) == 64 and len(result["groups"]) == 4
        assert all(s["status"] == "truncated" and "reward_seconds" in s for s in result["samples"])
        assert all(
            s["submitted"] <= s["admitted"] <= s["response_processed"] <= s["assembled"] for s in result["samples"]
        )
        print(
            json.dumps(
                {
                    "passed": True,
                    "fake_engine": True,
                    "native_miles_pipeline": True,
                    "http_workers": 16,
                    "responses": 64,
                    "tokens": 128,
                }
            )
        )
    finally:
        await runner.cleanup()
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
