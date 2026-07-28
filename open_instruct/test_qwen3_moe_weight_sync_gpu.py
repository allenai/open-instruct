"""End-to-end GPU test for Transformers 5 Qwen3-MoE live weight synchronization.

This pytest file is the sole isolated launch target for the test. Run it on
Beaker with:

    GPU_COUNT=3 ./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh \
        open_instruct/test_qwen3_moe_weight_sync_gpu.py

The tests use a tiny fused Qwen3-MoE learner under DeepSpeed ZeRO-3 and a vLLM
engine initialized with dummy weights. The single-GPU test exhaustively covers
the packed IPC tensor mapping. The three-GPU test uses two learner ranks and a
separate vLLM engine to exercise the packed NCCL path and its collectives.
"""

import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import datasets
import deepspeed
import ray
import torch
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM
from vllm.distributed.weight_transfer.base import WeightTransferInitRequest
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

from open_instruct import logger_utils, utils, vllm_utils
from open_instruct.data_types import GenerationResult, PromptRequest
from open_instruct.ground_truth_utils import RewardConfig
from open_instruct.vllm_utils import SamplingConfig

logger = logger_utils.setup_logger(__name__)

TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
NUM_EXPERTS = 2
PROMPT_TOKEN_IDS = [1, 2, 3, 4]


@ray.remote
class Qwen3MoeZero3Trainer:
    """Own one partitioned learner rank and drive the production sync path."""

    def __init__(self, model_path: str, distributed_port: int, rank: int = 0, world_size: int = 1):
        self.rank = rank
        os.environ.update(
            {
                "LOCAL_RANK": "0",
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(distributed_port),
            }
        )
        torch.cuda.set_device(0)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=f"tcp://127.0.0.1:{distributed_port}",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=2),
            )

        model = Qwen3MoeForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda()
        self.original_parameters = {
            name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
        }
        self.engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config={
                "train_batch_size": 1,
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "bf16": {"enabled": True},
                "zero_optimization": {"stage": 3, "stage3_param_persistence_threshold": 0},
            },
            dist_init_required=False,
        )
        self.model = self.engine.module
        self.model_update_group = None

    def setup_nccl_transfer(self, engine: ray.actor.ActorHandle, master_address: str, master_port: int) -> None:
        if self.rank == 0:
            master_info = {"master_address": master_address, "master_port": master_port, "world_size": 2}
            init_ref = engine.init_weight_transfer_engine.remote(
                WeightTransferInitRequest(init_info=master_info | {"rank_offset": 1})
            )
            torch.cuda.set_device(0)
            self.model_update_group = NCCLWeightTransferEngine.trainer_init(master_info)
            ray.get(init_ref)
        torch.distributed.barrier()

    def export_description(self) -> dict:
        adapter = vllm_utils.resolve_weight_export_adapter(self.model)
        metadata = vllm_utils._collect_weight_metadata(self.model, name_mapper=None, adapter=adapter)
        return {
            "adapter": adapter.name,
            "all_parameters_are_zero3": all(hasattr(parameter, "ds_id") for parameter in self.model.parameters()),
            "original_parameter_count": metadata.original_parameter_count,
            "exported_tensor_count": len(metadata.specs),
            "expanded_expert_tensor_count": metadata.expanded_expert_tensor_count,
            "exported_names": metadata.names,
        }

    def reset_and_perturb(self, projection: str, expert_index: int | None, seed: int) -> None:
        target_name = {
            "attention": "model.layers.0.self_attn.q_proj.weight",
            "embedding": "model.embed_tokens.weight",
            "output": "lm_head.weight",
            "router": "model.layers.0.mlp.gate.weight",
            "gate": "model.layers.0.mlp.experts.gate_up_proj",
            "up": "model.layers.0.mlp.experts.gate_up_proj",
            "down": "model.layers.0.mlp.experts.down_proj",
        }[projection]
        target_found = False

        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                with deepspeed.zero.GatheredParameters([parameter], modifier_rank=0, enabled=True):
                    parameter.copy_(self.original_parameters[name].to(device=parameter.device, dtype=parameter.dtype))
                    if name != target_name:
                        continue

                    target_found = True
                    target = parameter
                    if projection == "embedding":
                        target = parameter[1:5]
                    elif projection in {"gate", "up"}:
                        assert expert_index is not None
                        intermediate_size = parameter.shape[1] // 2
                        target = (
                            parameter[expert_index, :intermediate_size]
                            if projection == "gate"
                            else parameter[expert_index, intermediate_size:]
                        )
                    elif projection == "down":
                        assert expert_index is not None
                        target = parameter[expert_index]

                    generator = torch.Generator(device=target.device).manual_seed(seed)
                    target.add_(
                        torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=generator) * 0.5
                    )

        if not target_found:
            raise AssertionError(f"Did not find Qwen3-MoE parameter {target_name!r}")

    def sync(self, engine: ray.actor.ActorHandle, model_step: int) -> None:
        refs = vllm_utils.broadcast_weights_to_vllm(
            model=self.model, vllm_engines=[engine], model_update_group=self.model_update_group, model_step=model_step
        )
        if refs:
            raise AssertionError(f"Weight sync returned unfinished Ray references: {refs}")
        if self.rank == 0:
            ray.get(engine.wake_up.remote())
        torch.distributed.barrier()


def _write_tiny_checkpoint(path: Path) -> None:
    config = Qwen3MoeConfig(
        vocab_size=151936,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS,
        head_dim=16,
        tie_word_embeddings=False,
    )
    Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
    AutoTokenizer.from_pretrained(TOKENIZER_NAME).save_pretrained(path)


def _get_fingerprint(
    prompt_queue: ray_queue.Queue, results_queue: ray_queue.Queue, request_number: int
) -> tuple[tuple[int, ...], tuple[float, ...], int]:
    prompt_queue.put(
        PromptRequest(
            prompt=PROMPT_TOKEN_IDS,
            index=0,
            prompt_id=f"qwen3_moe_sync_{request_number}",
            generation_config=SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=1, seed=42, logprobs=20),
        )
    )
    result = results_queue.get(timeout=120)
    if not isinstance(result, GenerationResult):
        raise AssertionError(f"Expected GenerationResult, got {type(result)}")
    return (tuple(result.responses[0]), tuple(round(logprob, 6) for logprob in result.logprobs[0]), result.model_step)


def validate_qwen3_moe_live_weight_sync() -> None:
    """Run the isolated ZeRO-3 to vLLM Qwen3-MoE synchronization test."""

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")

    ray_started_here = not ray.is_initialized()
    if ray_started_here:
        ray.init()

    pg = None
    trainer = None
    engines = []
    try:
        with tempfile.TemporaryDirectory(prefix="qwen3_moe_sync_") as temp_dir:
            model_path = Path(temp_dir)
            _write_tiny_checkpoint(model_path)

            prompt_queue = ray_queue.Queue(maxsize=32)
            results_queue = ray_queue.Queue(maxsize=32)
            eval_results_queue = ray_queue.Queue(maxsize=1)
            train_dataset = datasets.Dataset.from_dict(
                {"ground_truth": [["4"]], "dataset": ["test"], "prompt": ["test"], "index": [0]}
            )

            pg = placement_group([{"GPU": 1, "CPU": 2}], strategy="PACK")
            ray.get(pg.ready())
            engines = vllm_utils.create_vllm_engines(
                num_engines=1,
                tensor_parallel_size=1,
                enforce_eager=True,
                tokenizer_name_or_path=str(model_path),
                pretrain=str(model_path),
                revision=None,
                seed=42,
                enable_prefix_caching=False,
                max_model_len=32,
                vllm_gpu_memory_utilization=0.35,
                single_gpu_mode=True,
                pg=pg,
                prompt_queue=prompt_queue,
                results_queue=results_queue,
                eval_results_queue=eval_results_queue,
                reward_config=RewardConfig(),
                train_dataset=train_dataset,
                load_format="dummy",
            )
            engine = engines[0]
            ray.get(engine.init_weight_transfer_engine.remote(WeightTransferInitRequest(init_info={})))

            scheduling = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )
            trainer = Qwen3MoeZero3Trainer.options(num_cpus=0.5, num_gpus=0.5, scheduling_strategy=scheduling).remote(
                str(model_path), utils.find_free_port()
            )

            description = ray.get(trainer.export_description.remote())
            assert description["adapter"] == "qwen3_moe"
            assert description["all_parameters_are_zero3"]
            assert description["expanded_expert_tensor_count"] == NUM_EXPERTS * 3
            assert description["exported_tensor_count"] > description["original_parameter_count"]

            exported_names = set(description["exported_names"])
            assert "model.layers.0.mlp.experts.gate_up_proj" not in exported_names
            assert "model.layers.0.mlp.experts.down_proj" not in exported_names
            for expert_index in range(NUM_EXPERTS):
                expert_prefix = f"model.layers.0.mlp.experts.{expert_index}"
                assert f"{expert_prefix}.gate_proj.weight" in exported_names
                assert f"{expert_prefix}.up_proj.weight" in exported_names
                assert f"{expert_prefix}.down_proj.weight" in exported_names

            ray.get(trainer.sync.remote(engine, 0))
            baseline = _get_fingerprint(prompt_queue, results_queue, request_number=0)
            baseline_repeat = _get_fingerprint(prompt_queue, results_queue, request_number=1)
            assert baseline == baseline_repeat, "Baseline vLLM generation was not deterministic"
            assert baseline[2] == 0

            perturbations = [
                ("attention", None),
                ("embedding", None),
                ("output", None),
                ("router", None),
                *[
                    (projection, expert_index)
                    for expert_index in range(NUM_EXPERTS)
                    for projection in ("gate", "up", "down")
                ],
            ]
            for model_step, (projection, expert_index) in enumerate(perturbations, start=1):
                ray.get(
                    trainer.reset_and_perturb.remote(
                        projection=projection, expert_index=expert_index, seed=1000 + model_step
                    )
                )
                ray.get(trainer.sync.remote(engine, model_step))
                updated = _get_fingerprint(prompt_queue, results_queue, request_number=model_step + 1)
                target = projection if expert_index is None else f"expert {expert_index} {projection}"
                assert updated[:2] != baseline[:2], f"{target} did not change live vLLM output"
                assert updated[2] == model_step, f"{target} did not stamp model step {model_step}"
                logger.info("Verified Qwen3-MoE live weight update for %s", target)
    finally:
        if trainer is not None:
            ray.kill(trainer, no_restart=True)
        for engine in engines:
            ray.kill(engine, no_restart=True)
        if pg is not None:
            remove_placement_group(pg)
        if ray_started_here:
            ray.shutdown()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestQwen3MoeLiveWeightSync(unittest.TestCase):
    def test_zero3_packed_ipc_updates_every_expert_projection(self):
        validate_qwen3_moe_live_weight_sync()

    @unittest.skipUnless(torch.cuda.device_count() >= 3, "Three GPUs are required for the multi-rank NCCL test")
    def test_zero3_packed_nccl_uses_every_learner_rank(self):
        ray_started_here = not ray.is_initialized()
        if ray_started_here:
            ray.init()

        trainers = []
        engines = []
        try:
            with tempfile.TemporaryDirectory(prefix="qwen3_moe_nccl_sync_") as temp_dir:
                model_path = Path(temp_dir)
                _write_tiny_checkpoint(model_path)

                prompt_queue = ray_queue.Queue(maxsize=16)
                results_queue = ray_queue.Queue(maxsize=16)
                eval_results_queue = ray_queue.Queue(maxsize=1)
                train_dataset = datasets.Dataset.from_dict(
                    {"ground_truth": [["4"]], "dataset": ["test"], "prompt": ["test"], "index": [0]}
                )
                engines = vllm_utils.create_vllm_engines(
                    num_engines=1,
                    tensor_parallel_size=1,
                    enforce_eager=True,
                    tokenizer_name_or_path=str(model_path),
                    pretrain=str(model_path),
                    revision=None,
                    seed=42,
                    enable_prefix_caching=False,
                    max_model_len=32,
                    vllm_gpu_memory_utilization=0.35,
                    single_gpu_mode=False,
                    prompt_queue=prompt_queue,
                    results_queue=results_queue,
                    eval_results_queue=eval_results_queue,
                    reward_config=RewardConfig(),
                    train_dataset=train_dataset,
                    load_format="dummy",
                )
                engine = engines[0]

                learner_port = utils.find_free_port()
                trainers = [
                    Qwen3MoeZero3Trainer.options(num_cpus=1, num_gpus=1).remote(str(model_path), learner_port, rank, 2)
                    for rank in range(2)
                ]
                transfer_address = ray._private.services.get_node_ip_address()
                transfer_port = utils.find_free_port()
                ray.get(
                    [
                        trainer.setup_nccl_transfer.remote(engine, transfer_address, transfer_port)
                        for trainer in trainers
                    ],
                    timeout=300,
                )

                ray.get([trainer.sync.remote(engine, 0) for trainer in trainers], timeout=300)
                baseline = _get_fingerprint(prompt_queue, results_queue, request_number=100)
                self.assertEqual(baseline[2], 0)

                perturbations = [("attention", None), ("router", None), ("gate", 0), ("down", 1)]
                for model_step, (projection, expert_index) in enumerate(perturbations, start=1):
                    ray.get(
                        [
                            trainer.reset_and_perturb.remote(
                                projection=projection, expert_index=expert_index, seed=2000 + model_step
                            )
                            for trainer in trainers
                        ],
                        timeout=300,
                    )
                    ray.get([trainer.sync.remote(engine, model_step) for trainer in trainers], timeout=300)
                    updated = _get_fingerprint(prompt_queue, results_queue, request_number=100 + model_step)
                    target = projection if expert_index is None else f"expert {expert_index} {projection}"
                    self.assertNotEqual(updated[:2], baseline[:2], f"{target} did not change live vLLM output")
                    self.assertEqual(updated[2], model_step, f"{target} did not stamp model step {model_step}")
        finally:
            for trainer in trainers:
                ray.kill(trainer, no_restart=True)
            for engine in engines:
                ray.kill(engine, no_restart=True)
            if ray_started_here:
                ray.shutdown()


if __name__ == "__main__":
    unittest.main()
