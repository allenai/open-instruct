"""Optional one-GPU smoke test that proves Qwen3-MoE runtime weights refresh.

Run in the CUDA image with:
    uv run python scripts/test/validate_qwen3_moe_live_weight_sync.py

The test builds a tiny Transformers 5 Qwen3-MoE checkpoint, loads a dummy
instance in vLLM, performs packed IPC synchronization through Open-Instruct,
and verifies deterministic next-token logprobs change after independently
perturbing dense, router, gate, up, and down learner weights.
"""

import os
import tempfile
from pathlib import Path

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM
from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig

from open_instruct import vllm_utils

GPU_FRACTION = 0.48
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"


@ray.remote(num_cpus=0, num_gpus=GPU_FRACTION)
class ValidationLLM:
    def __init__(self, model_path: str):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(GPU_FRACTION)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0"
        self.model_step = -1
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            load_format="dummy",
            dtype="bfloat16",
            enforce_eager=True,
            distributed_executor_backend="ray",
            enable_sleep_mode=True,
            gpu_memory_utilization=0.35,
            max_model_len=32,
            weight_transfer_config=WeightTransferConfig(backend="ipc"),
        )
        self.llm.init_weight_transfer_engine({"init_info": {}})

    def sleep(self) -> None:
        self.llm.sleep(level=0, mode="keep")

    def wake_up(self) -> None:
        self.llm.wake_up(tags=["scheduling"])

    def start_weight_update(self) -> None:
        self.llm.start_weight_update()

    def update_weights(self, request: dict) -> None:
        self.llm.update_weights(request)

    def finish_weight_update(self) -> None:
        self.llm.finish_weight_update()

    def set_model_step(self, model_step: int) -> None:
        self.model_step = model_step

    def fingerprint(self) -> tuple[int, tuple[tuple[int, float], ...]]:
        output = self.llm.generate(
            [{"prompt_token_ids": [1, 2, 3, 4]}],
            SamplingParams(temperature=0, max_tokens=1, logprobs=20),
            use_tqdm=False,
        )[0].outputs[0]
        logprobs = tuple(sorted((token_id, round(value.logprob, 7)) for token_id, value in output.logprobs[0].items()))
        return output.token_ids[0], logprobs


@ray.remote(num_cpus=0, num_gpus=GPU_FRACTION)
class ValidationTrainer:
    def __init__(self, model_path: str):
        self.model = Qwen3MoeForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
        self.original = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}

    def sync(self, engine, model_step: int) -> None:
        refs = vllm_utils.broadcast_weights_to_vllm(
            model=self.model, vllm_engines=[engine], model_update_group=None, model_step=model_step
        )
        assert refs == []
        ray.get(engine.wake_up.remote())

    def perturb(self, kind: str) -> None:
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                parameter.copy_(self.original[name])

            parameters = dict(self.model.named_parameters())
            generator = torch.Generator(device="cuda").manual_seed(1234)

            if kind == "attention":
                target = parameters["model.layers.0.self_attn.q_proj.weight"]
            elif kind == "embedding":
                target = parameters["model.embed_tokens.weight"]
            elif kind == "output":
                target = parameters["lm_head.weight"]
            elif kind == "router":
                target = parameters["model.layers.0.mlp.gate.weight"]
            elif kind in {"gate", "up"}:
                fused = parameters["model.layers.0.mlp.experts.gate_up_proj"]
                intermediate_size = fused.shape[1] // 2
                target = fused[:, :intermediate_size] if kind == "gate" else fused[:, intermediate_size:]
            elif kind == "down":
                target = parameters["model.layers.0.mlp.experts.down_proj"]
            else:
                raise ValueError(f"Unknown perturbation kind: {kind}")

            target.add_(
                torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=generator) * 0.25
            )


def _write_tiny_checkpoint(path: Path) -> None:
    config = Qwen3MoeConfig(
        vocab_size=151936,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        head_dim=16,
        tie_word_embeddings=False,
    )
    Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
    AutoTokenizer.from_pretrained(TOKENIZER_NAME).save_pretrained(path)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This validation requires a CUDA GPU")

    with tempfile.TemporaryDirectory(prefix="qwen3_moe_sync_") as temp_dir:
        model_path = Path(temp_dir)
        _write_tiny_checkpoint(model_path)

        ray.init()
        pg = placement_group([{"GPU": 1, "CPU": 1}], strategy="PACK")
        ray.get(pg.ready())
        scheduling = PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_capture_child_tasks=True)
        engine = ValidationLLM.options(scheduling_strategy=scheduling).remote(str(model_path))
        trainer = ValidationTrainer.options(scheduling_strategy=scheduling).remote(str(model_path))

        ray.get(trainer.sync.remote(engine, 0))
        baseline = ray.get(engine.fingerprint.remote())
        for model_step, kind in enumerate(
            ("attention", "embedding", "output", "router", "gate", "up", "down"), start=1
        ):
            ray.get(trainer.perturb.remote(kind))
            ray.get(trainer.sync.remote(engine, model_step))
            updated = ray.get(engine.fingerprint.remote())
            if updated == baseline:
                raise AssertionError(f"{kind} perturbation did not change vLLM next-token logprobs")
            print(f"PASS: {kind} weights changed runtime logprobs")

        ray.shutdown()


if __name__ == "__main__":
    main()
