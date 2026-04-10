"""
Minimal reproduction of NaN after NCCLWeightTransferEngine weight sync.

Based directly on vllm/examples/rl/rlhf_nccl.py but with:
- packed=False (matching our GRPO setup)
- Real base weights (not load_format="dummy")
- Qwen2.5-1.5B (same family as our Qwen2.5-7B)

Usage (2 GPUs):
  python scripts/debug/minimal_weight_sync_repro.py

Set USE_PACKED=true to test with packed=True (should match working example).
Set LOAD_FORMAT=dummy to test with dummy weights (should match working example).
"""

import os

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)
from vllm.utils.network_utils import get_ip, get_open_port

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
TP_SIZE = int(os.environ.get("TP_SIZE", "2"))
USE_PACKED = os.environ.get("USE_PACKED", "false").lower() == "true"
LOAD_FORMAT = os.environ.get("LOAD_FORMAT", "auto")
SEND_DATA_ONLY = os.environ.get("SEND_DATA_ONLY", "false").lower() == "true"


class MyLLM(LLM):
    def __init__(self, *args, bundle_indices=None, **kwargs):
        if bundle_indices is not None:
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        super().__init__(*args, **kwargs)


@ray.remote(num_gpus=1)
class TrainModel:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
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

    def broadcast_weights(self, packed, send_data_only=False):
        trainer_args = NCCLTrainerSendWeightsArgs(
            group=self.model_update_group,
            packed=packed,
        )
        if send_data_only:
            mapped_params = [(n, p.data) for n, p in self.model.named_parameters()]
            iterator = iter(mapped_params)
        else:
            iterator = self.model.named_parameters()
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=iterator,
            trainer_args=trainer_args,
        )


ray.init()

print(f"Model: {MODEL_NAME}, TP={TP_SIZE}, packed={USE_PACKED}, load_format={LOAD_FORMAT}, send_data_only={SEND_DATA_ONLY}")

train_model = TrainModel.remote(MODEL_NAME)

pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * TP_SIZE)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(MyLLM).remote(
    model=MODEL_NAME,
    enforce_eager=True,
    tensor_parallel_size=TP_SIZE,
    distributed_executor_backend="ray",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
    load_format=LOAD_FORMAT,
    bundle_indices=list(range(TP_SIZE)),
)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0, max_tokens=50)

print("\n" + "=" * 60)
print("PHASE 1: Generate BEFORE weight sync")
print("=" * 60)
outputs_before = ray.get(llm.generate.remote(prompts, sampling_params))
for output in outputs_before:
    print(f"  {output.prompt!r} -> {output.outputs[0].text!r}")

print("\n" + "=" * 60)
print("PHASE 2: Weight sync")
print("=" * 60)

ray.get(llm.sleep.remote(level=0))

master_address, master_port = ray.get(
    train_model.get_master_address_and_port.remote()
)
world_size = ray.get(llm.get_world_size.remote()) + 1

inference_handle = llm.init_weight_transfer_engine.remote(
    dict(
        init_info=dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,
        )
    )
)
train_handle = train_model.init_weight_transfer_group.remote(world_size)
ray.get([train_handle, inference_handle])

names, dtype_names, shapes = ray.get(train_model.get_weight_metadata.remote())
print(f"  Sending {len(names)} parameters, packed={USE_PACKED}")

inference_handle = llm.update_weights.remote(
    dict(
        update_info=dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=USE_PACKED,
        )
    )
)
train_handle = train_model.broadcast_weights.remote(packed=USE_PACKED, send_data_only=SEND_DATA_ONLY)
ray.get([train_handle, inference_handle])
print("  Weight sync complete!")

ray.get(llm.wake_up.remote(tags=["scheduling"]))

print("\n" + "=" * 60)
print("PHASE 3: Generate AFTER weight sync")
print("=" * 60)
outputs_after = ray.get(llm.generate.remote(prompts, sampling_params))

has_nan = False
for output in outputs_after:
    text = output.outputs[0].text
    print(f"  {output.prompt!r} -> {text!r}")
    if not text or "nan" in text.lower():
        has_nan = True

print("\n" + "=" * 60)
if has_nan:
    print("RESULT: NaN detected after weight sync - BUG REPRODUCED")
else:
    print("RESULT: Generation looks normal after weight sync")
print("=" * 60)
