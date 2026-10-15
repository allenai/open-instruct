"""Exercise the GRPO CPUAdam factory through a real ZeRO-2 offloaded optimizer step."""

import contextlib
import datetime
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import distributed as dist

from open_instruct import grpo_utils, utils

# CPU CI does not install DeepSpeed; CUDA runs must have the real dependency.
deepspeed = None
with contextlib.suppress(Exception):
    import deepspeed


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestGRPOCPUAdamOffload(unittest.TestCase):
    def test_zero2_offload_updates_cuda_model_with_cpu_optimizer_state(self):
        self.assertIsNotNone(deepspeed, "DeepSpeed is required for the CPU-offload GPU smoke test")
        self.assertIsNotNone(grpo_utils.DeepSpeedCPUAdam)
        owns_process_group = not dist.is_initialized()
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            if owns_process_group:
                dist.init_process_group(
                    backend="nccl",
                    init_method=pathlib.Path(tmpdir, "distributed-init").as_uri(),
                    rank=0,
                    world_size=1,
                    timeout=datetime.timedelta(seconds=60),
                )
            try:
                torch.cuda.set_device(0)
                model = torch.nn.Linear(8, 4, bias=False).cuda()
                with torch.no_grad():
                    model.weight.zero_()
                initial_weight = model.weight.detach().clone()
                args = grpo_utils.GRPOExperimentConfig(
                    use_cpu_adam=True,
                    deepspeed_stage=2,
                    deepspeed_offload_optimizer=True,
                    deepspeed_zpg=1,
                    learning_rate=0.05,
                    weight_decay=0.0,
                    fused_optimizer=False,
                )
                optimizer = grpo_utils.create_optimizer(model.parameters(), args)
                self.assertIsInstance(optimizer, grpo_utils.DeepSpeedCPUAdam)
                config = utils.get_train_ds_config(
                    offload=args.deepspeed_offload_param,
                    adam_offload=args.deepspeed_offload_optimizer,
                    stage=args.deepspeed_stage,
                    bf16=True,
                    zpg=args.deepspeed_zpg,
                )
                config.update(train_micro_batch_size_per_gpu=1, gradient_accumulation_steps=1)
                engine, zero_optimizer, _, _ = deepspeed.initialize(
                    model=model, optimizer=optimizer, config=config, dist_init_required=False
                )
                self.assertIs(zero_optimizer.optimizer, optimizer)
                self.assertTrue(zero_optimizer.cpu_offload)
                inputs = torch.ones(1, 8, device="cuda", dtype=engine.module.weight.dtype)
                target = torch.ones(1, 4, device="cuda")
                losses = []
                for _ in range(3):
                    loss = (engine(inputs).float() - target).square().mean()
                    self.assertTrue(torch.isfinite(loss).item())
                    losses.append(loss.item())
                    engine.backward(loss)
                    engine.step()
                self.assertLess(losses[-1], losses[0])
                self.assertFalse(torch.equal(engine.module.weight.float(), initial_weight))
                self.assertTrue(all(p.device.type == "cuda" for p in engine.module.parameters()))
                self.assertTrue(all(p.device.type == "cpu" for g in optimizer.param_groups for p in g["params"]))
                self.assertTrue(optimizer.state, "Optimizer state must exist after a real optimizer step")
                for state in optimizer.state.values():
                    for key in ("exp_avg", "exp_avg_sq"):
                        self.assertEqual(state[key].device.type, "cpu")
                        self.assertTrue(torch.isfinite(state[key]).all().item())
                self.assertEqual(engine.global_steps, 3)
                print(f"CPU_ADAM_OFFLOAD_SMOKE_PASS: steps={engine.global_steps}, losses={losses}")
            finally:
                if owns_process_group:
                    dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
