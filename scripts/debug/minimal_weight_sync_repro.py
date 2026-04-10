"""
Minimal reproduction of NaN after NCCLWeightTransferEngine weight sync.

Replicates our GRPO setup:
- AsyncLLMEngine (not offline LLM)
- TP=2
- packed=False
- Real base weights (not dummy)
- Qwen2.5-7B (or smaller Qwen for faster testing)

Usage (3 GPUs: 1 trainer + 2 for TP=2):
  ray start --head --num-gpus=3
  python scripts/debug/minimal_weight_sync_repro.py

Or on a single machine with 3+ GPUs:
  python scripts/debug/minimal_weight_sync_repro.py
"""

import asyncio
import os
import uuid
from dataclasses import asdict

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer

import vllm
from vllm import SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
    NCCLWeightTransferUpdateInfo,
)
from vllm.utils.network_utils import get_ip, get_open_port
from vllm.v1.executor import Executor

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
TP_SIZE = int(os.environ.get("TP_SIZE", "2"))
USE_PACKED = os.environ.get("USE_PACKED", "false").lower() == "true"


class MyAsyncLLM(vllm.AsyncLLMEngine):
    def __init__(self, **kwargs):
        bundle_indices = kwargs.pop("bundle_indices", None)
        if bundle_indices is not None:
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(
                map(str, bundle_indices)
            )
        engine_args = vllm.AsyncEngineArgs(**kwargs)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        super().__init__(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=False,
            log_stats=False,
        )

    async def do_generate(self, prompt_token_ids, sampling_params):
        output = None
        async for request_output in self.generate(
            {"prompt_token_ids": prompt_token_ids},
            sampling_params,
            request_id=str(uuid.uuid4()),
        ):
            output = request_output
        return output


@ray.remote(num_gpus=1)
class TrainModel:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to("cuda:0")
        self.port = get_open_port()
        self.master_address = get_ip()

    def get_master_address_and_port(self):
        return self.master_address, self.port

    def get_weight_metadata(self):
        names = []
        dtype_names = []
        shapes = []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        return names, dtype_names, shapes

    def init_weight_transfer_group(self, world_size):
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.master_address,
                master_port=self.port,
                world_size=world_size,
            ),
        )

    def broadcast_weights(self, packed):
        trainer_args = NCCLTrainerSendWeightsArgs(
            group=self.model_update_group,
            packed=packed,
        )
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=self.model.named_parameters(),
            trainer_args=trainer_args,
        )


def main():
    ray.init(
        runtime_env={
            "env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_ENV_VAR": "1"}
        }
    )

    print(f"Model: {MODEL_NAME}, TP={TP_SIZE}, packed={USE_PACKED}")

    train_model = TrainModel.remote(MODEL_NAME)

    pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * TP_SIZE)
    ray.get(pg_inference.ready())
    scheduling_inference = PlacementGroupSchedulingStrategy(
        placement_group=pg_inference,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )

    bundle_indices = list(range(TP_SIZE))

    llm = ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=scheduling_inference,
    )(MyAsyncLLM).remote(
        model=MODEL_NAME,
        enforce_eager=True,
        tensor_parallel_size=TP_SIZE,
        distributed_executor_backend="ray",
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
        dtype="bfloat16",
        bundle_indices=bundle_indices,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompt_token_ids = [
        tokenizer.encode(p, add_special_tokens=False) for p in prompts
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=50)

    print("\n" + "=" * 60)
    print("PHASE 1: Generate BEFORE weight sync (should be normal)")
    print("=" * 60)
    outputs_before = ray.get(
        [llm.do_generate.remote(ptids, sampling_params) for ptids in prompt_token_ids]
    )
    for output in outputs_before:
        text = tokenizer.decode(output.outputs[0].token_ids)
        print(f"  {output.prompt!r} -> {text!r}")

    print("\n" + "=" * 60)
    print("PHASE 2: Weight sync")
    print("=" * 60)

    ray.get(llm.sleep.remote(level=0, mode="keep"))

    master_address, master_port = ray.get(
        train_model.get_master_address_and_port.remote()
    )
    world_size = TP_SIZE + 1

    inference_handle = llm.init_weight_transfer_engine.remote(
        WeightTransferInitRequest(
            init_info=asdict(
                NCCLWeightTransferInitInfo(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=1,
                    world_size=world_size,
                )
            )
        )
    )
    train_handle = train_model.init_weight_transfer_group.remote(world_size)
    ray.get([train_handle, inference_handle])

    names, dtype_names, shapes = ray.get(
        train_model.get_weight_metadata.remote()
    )
    print(f"  Sending {len(names)} parameters, packed={USE_PACKED}")

    inference_handle = llm.update_weights.remote(
        WeightTransferUpdateRequest(
            update_info=asdict(
                NCCLWeightTransferUpdateInfo(
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    packed=USE_PACKED,
                )
            )
        )
    )
    train_handle = train_model.broadcast_weights.remote(packed=USE_PACKED)
    ray.get([train_handle, inference_handle])
    print("  Weight sync complete!")

    ray.get(llm.wake_up.remote(tags=["scheduling"]))

    print("\n" + "=" * 60)
    print("PHASE 3: Generate AFTER weight sync (NaN bug would show here)")
    print("=" * 60)
    outputs_after = ray.get(
        [llm.do_generate.remote(ptids, sampling_params) for ptids in prompt_token_ids]
    )

    has_nan = False
    for output in outputs_after:
        text = tokenizer.decode(output.outputs[0].token_ids)
        print(f"  {output.prompt!r} -> {text!r}")
        if "nan" in text.lower() or not output.outputs[0].token_ids:
            has_nan = True

    print("\n" + "=" * 60)
    if has_nan:
        print("RESULT: NaN detected after weight sync - BUG REPRODUCED")
    else:
        print("RESULT: Generation looks normal after weight sync")
    print("=" * 60)

    ray.get(llm.shutdown.remote())
    ray.shutdown()


if __name__ == "__main__":
    main()
