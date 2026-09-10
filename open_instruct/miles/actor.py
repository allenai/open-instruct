"""MILES TrainRayActor implemented with OLMo-core train modules."""

import contextlib
import json
import time
from pathlib import Path

import ray
import torch
from miles.backends.fsdp_utils import update_weight_utils
from miles.backends.training_utils import data as miles_data
from miles.backends.training_utils import loss as miles_loss
from miles.backends.training_utils import parallel
from miles.ray.train_actor import TrainRayActor
from miles.utils import distributed_utils
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.hf_config import HF_EXPORT_COMPLETE_MARKER
from olmo_core.nn.moe.v2 import replay
from safetensors import torch as safetensors_torch
from torch import distributed as dist
from transformers import AutoTokenizer

from open_instruct import logger_utils
from open_instruct.miles import checkpoint, data, models, publication, scheduler
from open_instruct.miles.state import PolicyClock

logger = logger_utils.setup_logger(__name__)


class OLMoCoreTrainRayActor(TrainRayActor):
    def init(self, args, role, *, with_ref=False, with_opd_teacher=False, recv_ckpt_src_rank=None, indep_dp_info=None):
        if role != "actor" or with_opd_teacher or recv_ckpt_src_rank is not None:
            raise ValueError("Core backend supports the policy actor without trainer-cell recovery")
        super().init(args, role, with_ref=with_ref, with_opd_teacher=False)
        torch.manual_seed(args.seed)
        world = dist.get_world_size()
        rank = dist.get_rank()
        # Expert parallel ranks own different prompt samples, like Megatron's EP-inside-DP layout.
        dp = GroupInfo(rank=rank, size=world, group=dist.group.WORLD, gloo_group=distributed_utils.get_gloo_group())
        trivial = GroupInfo(rank=0, size=1, group=None)
        parallel.set_parallel_state(
            parallel.ParallelState(
                intra_dp=dp,
                intra_dp_cp=dp,
                cp=trivial,
                tp=trivial,
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
            )
        )
        self.train_parallel_config = {"dp_size": world}
        self.train_module, self.hf_config, self.model_config = models.build_train_module(args)
        self.model = self.train_module.model
        self.optimizer = self.train_module.optim
        self.lr_scheduler = scheduler.CoreLRScheduler(args, self.optimizer)
        self.clock = PolicyClock()
        self.ref_module = models.build_train_module(args, source=args.ref_load)[0] if with_ref else None
        if self.ref_module is not None:
            self.ref_module.model.requires_grad_(False)
        checkpoint.restore(self)
        self.train_module._trainer.global_step = self.clock.completed_steps
        updater = (
            update_weight_utils.UpdateWeightFromTensor
            if args.colocate
            else update_weight_utils.UpdateWeightFromDistributed
        )
        if not args.colocate and args.olmo_core.weight_sync_mode == "flattened":
            updater = publication.FlattenedDistributedUpdater
        self.weight_updater = updater(args, self.model)
        return self.clock.next_rollout_id

    def _get_parallel_config(self):
        return self.train_parallel_config

    def _agree(self, operation):
        error = None
        result = None
        try:
            result = operation()
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            error = f"rank {dist.get_rank()}: {exc}"
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error, group=distributed_utils.get_gloo_group())
        if any(errors):
            raise RuntimeError("Core RL validation failed: " + "; ".join(e for e in errors if e))
        return result

    def _replay_context(self, module, batch):
        if not self.args.use_rollout_routing_replay:
            return contextlib.nullcontext()
        return replay.replay_routes(module.model, data.router_routes(module.model, batch))

    def _forward(self, module, batch):
        forward = getattr(module, "model_forward_no_pipeline", None) or module.model_forward
        # One unpadded sample per forward avoids auxiliary losses on artificial padding.
        return forward(batch["tokens"], loss_div_factor=batch.get("aux_loss_div_factor", batch["tokens"].numel()))

    def _score(self, module, batches, *, use_replay):
        scores = []
        module.model.eval()
        with torch.no_grad():
            for batch in batches:
                with self._replay_context(module, batch) if use_replay else contextlib.nullcontext():
                    logits = self._forward(module, batch)
                    result = miles_loss.get_log_probs_and_entropy(
                        logits,
                        args=self.args,
                        unconcat_tokens=batch["unconcat_tokens"],
                        total_lengths=batch["total_lengths"],
                        response_lengths=batch["response_lengths"],
                        max_seq_lens=batch["max_seq_lens"],
                    )
                    scores.extend(result["log_probs"])
        return scores

    def train(self, rollout_id, rollout_data_ref, *, witness_info=None, attempt=0, external_data=None):
        if attempt or witness_info is not None or external_data is not None:
            raise ValueError("Trainer retry, witnesses, and external critic data are not supported")
        rollout, store = miles_data.get_rollout_data(self.args, rollout_data_ref)
        with store:
            batches = self._agree(lambda: data.sample_batches(rollout, self.args.olmo_core.max_sequence_length))
            versions = self._agree(lambda: data.policy_versions(rollout))
            self._agree(lambda: self.clock.validate_versions(versions, self.args.olmo_core.max_policy_lag))
            rollout["log_probs"] = self._score(self.train_module, batches, use_replay=True)
            if self.ref_module is not None:
                rollout["ref_log_probs"] = self._score(self.ref_module, batches, use_replay=False)
            miles_loss.compute_advantages_and_returns(self.args, rollout)
            batches = data.sample_batches(rollout, self.args.olmo_core.max_sequence_length)
            local_batch = self.args.global_batch_size // dist.get_world_size()
            self._agree(lambda: self._validate_step_batches(batches, local_batch))
            for start in range(0, len(batches), local_batch):
                step_batches = batches[start : start + local_batch]
                self._agree(
                    lambda current=step_batches: self.clock.validate_versions(
                        [v for batch in current for v in data.policy_versions(batch)],
                        self.args.olmo_core.max_policy_lag,
                    )
                )
                denominator = sum(mask.sum().clamp_min(1) for batch in step_batches for mask in batch["loss_masks"])
                dist.all_reduce(denominator)
                auxiliary_tokens = torch.tensor(
                    sum(batch["tokens"].numel() for batch in step_batches), device=denominator.device
                )
                dist.all_reduce(auxiliary_tokens)
                for batch in step_batches:
                    batch["aux_loss_div_factor"] = auxiliary_tokens / dist.get_world_size()
                self.train_module.zero_grads()

                count = len(step_batches)

                def objective(module, batch, count=count, token_denominator=denominator):
                    logits = self._forward(module, batch)
                    loss, _, metrics = miles_loss.loss_function(
                        self.args, batch, count, logits, apply_megatron_loss_scaling=False
                    )
                    if self.args.calculate_per_token_loss:
                        loss = loss * dist.get_world_size() / token_denominator
                    self._agree(lambda: self._validate_loss(loss))
                    return loss, dict(zip(metrics["keys"], metrics["values"][1:], strict=True))

                metrics = self.train_module.train_batch_with_loss(step_batches, objective, self._replay_context)
                self.train_module.optim_step()
                self._agree(
                    lambda: self.clock.optimizer_step(not bool(getattr(self.optimizer, "step_skipped", False)))
                )
                self.lr_scheduler.step()
                self.train_module._trainer.global_step = self.clock.completed_steps
                logger.info(
                    "Core optimizer step %s: %s",
                    self.clock.completed_steps,
                    {key: sum(float(m[key]) for m in metrics) / len(metrics) for key in metrics[0]},
                )
            self.clock.next_rollout_id = rollout_id + 1
        self._heartbeat.bump()

    @staticmethod
    def _validate_step_batches(batches, local_batch):
        if not local_batch or len(batches) % local_batch:
            raise ValueError("Rollout collection must contain complete optimizer batches on every rank")

    @staticmethod
    def _validate_loss(loss):
        if not bool(torch.isfinite(loss)):
            raise ValueError("Non-finite policy loss")

    def save_model(self, rollout_id, force_sync=False):
        checkpoint.save(self, rollout_id)

    def finalize_checkpoint(self, rollout_id):
        self._agree(lambda: checkpoint.finalize(self, rollout_id))

    def update_weights(self, info):
        torch.cuda.synchronize()
        started = time.perf_counter()
        updater = self.weight_updater
        if info.has_new_engines:
            updater.connect_rollout_engines(
                info.rollout_engines,
                info.rollout_engine_lock,
                engine_gpu_counts=info.engine_gpu_counts,
                engine_gpu_offsets=info.engine_gpu_offsets,
            )
        engines = info.rollout_engines
        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in engines])
            ray.get([engine.begin_weight_update.remote() for engine in engines])
        dist.barrier()
        pause_done = time.perf_counter()
        transfer_seconds, tensor_count, byte_count, bucket_count = 0.0, 0, 0, 0

        def send(bucket):
            nonlocal transfer_seconds, bucket_count
            torch.cuda.synchronize()
            before = time.perf_counter()
            updater.update_bucket_weights(bucket, weight_version=self.clock.completed_steps)
            torch.cuda.synchronize()
            transfer_seconds += time.perf_counter() - before
            bucket_count += 1

        bucket, size = [], 0
        for name, tensor in models.iter_export_state(
            self.train_module, self.hf_config, stream_moe=self.args.olmo_core.stream_moe_export
        ):
            tensor_count += 1
            byte_count += tensor.nbytes
            if bucket and size + tensor.nbytes > self.args.update_weight_buffer_size:
                send(bucket)
                bucket, size = [], 0
            bucket.append((name, tensor.to(device="cuda", dtype=torch.bfloat16).contiguous()))
            size += tensor.nbytes
        if bucket:
            send(bucket)
        export_done = time.perf_counter()
        dist.barrier()
        if dist.get_rank() == 0:
            ray.get([engine.flush_cache.remote() for engine in engines])
            ray.get([engine.end_weight_update.remote() for engine in engines])
            ray.get(self.rollout_manager.set_weight_version.remote(self.clock.completed_steps))
            ray.get(self.rollout_manager.clear_updatable_has_new_engines.remote())
            ray.get([engine.continue_generation.remote() for engine in engines])
        dist.barrier()
        self.clock.published()
        if dist.get_rank() == 0:
            timings = dict(
                version=self.clock.completed_steps,
                pause_connect_seconds=pause_done - started,
                export_pack_seconds=export_done - pause_done - transfer_seconds,
                transport_load_seconds=transfer_seconds,
                finalize_seconds=time.perf_counter() - export_done,
                total_seconds=time.perf_counter() - started,
                tensors=tensor_count,
                bytes=byte_count,
                buckets=bucket_count,
                transport_collectives=bucket_count
                if self.args.colocate or self.args.olmo_core.weight_sync_mode == "flattened"
                else tensor_count,
                stream_moe_export=self.args.olmo_core.stream_moe_export,
                transport="ipc" if self.args.colocate else self.args.olmo_core.weight_sync_mode,
            )
            logger.info("Core weight publication: %s", json.dumps(timings, sort_keys=True))
            if self.args.save:
                path = Path(self.args.save) / "publication.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as output:
                    output.write(json.dumps(timings) + "\n")

    def export_hf(self, rollout_id, path):
        state = models.export_state(self.train_module, self.hf_config)
        if dist.get_rank() == 0:
            target = Path(path)
            target.mkdir(parents=True, exist_ok=False)
            self.hf_config.save_pretrained(target)
            AutoTokenizer.from_pretrained(self.args.hf_checkpoint).save_pretrained(target)
            safetensors_torch.save_file(
                {name: value.detach().cpu().contiguous().clone() for name, value in state.items()},
                target / "model.safetensors",
            )
            (target / HF_EXPORT_COMPLETE_MARKER).touch()
        dist.barrier()

    def sleep(self, tags=None):
        raise NotImplementedError("Core trainer offload is not qualified; use --no-offload-train")

    def wake_up(self, tags=None):
        if self.args.offload_train:
            raise NotImplementedError("Core trainer offload is not qualified")

    def reconcile_adapters(self):
        if self.args.multi_lora:
            raise ValueError("Core backend does not support multi-LoRA")

    def shutdown(self):
        """Release trainer process groups after the producer and all writes have stopped."""
        if dist.is_initialized():
            dist.destroy_process_group()
