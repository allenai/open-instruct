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
from safetensors import torch as safetensors_torch
from torch import distributed as dist
from transformers import AutoTokenizer

from open_instruct import logger_utils
from open_instruct.miles import checkpoint, config, contract, data, models, publication, replay_diagnostics, scheduler
from open_instruct.miles import metrics as training_metrics
from open_instruct.miles.state import PolicyClock
from open_instruct.miles.timing import startup_stage

logger = logger_utils.setup_logger(__name__)


class OLMoCoreTrainRayActor(TrainRayActor):
    def init(self, args, role, *, with_ref=False, with_opd_teacher=False, recv_ckpt_src_rank=None, indep_dp_info=None):
        if role != "actor" or with_opd_teacher or recv_ckpt_src_rank is not None:
            raise ValueError("Core backend supports the policy actor without trainer-cell recovery")
        with startup_stage(args, "distributed_init"):
            super().init(args, role, with_ref=with_ref, with_opd_teacher=False)
        self._agree(lambda: training_metrics.init_tracking(args))
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
        with startup_stage(args, "native_restore", device=torch.cuda):
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
        self._scoring_pass()
        return self.clock.next_rollout_id

    def _scoring_pass(self):
        """Resolve once whether every update runs the standalone scoring pass."""
        decision = getattr(self, "_scoring_decision", None)
        if decision is not None:
            return decision
        args = self.args
        decision = config.scoring_pass(
            args.olmo_core,
            {
                "global_batch_size": args.global_batch_size,
                "rollout_batch_size": getattr(args, "rollout_batch_size", None) or 0,
                "n_samples_per_prompt": getattr(args, "n_samples_per_prompt", None) or 1,
                "kl_coef": getattr(args, "kl_coef", 0) or 0,
                "use_rollout_logprobs": bool(getattr(args, "use_rollout_logprobs", False)),
            },
        )
        hf_config = getattr(self, "hf_config", None)
        if not decision.standalone and hf_config is not None:
            stochastic = config.stochastic_fields(hf_config.to_dict())
            if stochastic:
                decision = config.ScoringPass(
                    True,
                    "stochastic model configuration: " + ", ".join(stochastic),
                    decision.optimizer_steps_per_collection,
                )
        logger.info("Core scoring pass: %s", json.dumps(decision.as_dict(), sort_keys=True))
        self._scoring_decision = decision
        self._scoring_checks = 0
        return decision

    def _scoring_check_due(self, decision):
        if decision.standalone:
            return False
        return config.scoring_check_due(self.args.olmo_core, self._scoring_checks, self.clock.completed_steps)

    @contextlib.contextmanager
    def _actor_forward_skipped(self, skipped):
        """Let MILES take old log-probabilities from the training forward for this update.

        MILES' parser only accepts its native flag for the Megatron backend; this
        adapter implements the same contract, so the attribute is set here for the
        duration of the update and restored afterwards, including on failure.
        """
        previous = getattr(self.args, "skip_actor_forward_only", False)
        self.args.skip_actor_forward_only = skipped
        try:
            yield
        finally:
            self.args.skip_actor_forward_only = previous

    def _training_log_probs(self, logits, batch):
        with torch.no_grad():
            result = miles_loss.get_log_probs_and_entropy(
                logits.detach(),
                args=self.args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=batch["total_lengths"],
                response_lengths=batch["response_lengths"],
                max_seq_lens=batch["max_seq_lens"],
            )
        return [value.detach() for value in result["log_probs"]]

    def _score_contract(self, rollout, rollout_id, source):
        """Validate the behavior-policy agreement gate and profile before any weight change."""
        self._agree(lambda: contract.validate_training_data(rollout))
        agreement = self._agree(lambda: data.score_agreement(rollout))
        dist.all_reduce(agreement)
        difference = self._agree(
            lambda: data.validate_score_agreement(agreement, self.args.olmo_core.max_train_rollout_logprob_abs_diff)
        )
        logger.info("Core behavior-policy agreement: mean_abs=%s active_tokens=%s", difference, int(agreement[1]))
        profile = contract.probability_profile(rollout)
        logger.info(
            "Core score contract: %s",
            contract.record(self.args, {"event": "scores", "rollout_id": rollout_id, "source": source, **profile}),
        )
        return difference, agreement, profile

    def _check_training_scores(self, rollout, training_scores):
        # Agree on local shape/finite-value failures before entering tensor collectives.
        sums, maximum = self._agree(
            lambda: contract.scoring_check(rollout["log_probs"], training_scores, rollout["loss_masks"])
        )
        dist.all_reduce(sums)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        return self._agree(
            lambda: contract.validate_scoring_check(sums, maximum, self.args.olmo_core.scoring_check_tolerance)
        )

    def _get_parallel_config(self):
        return self.train_parallel_config

    def _agree(self, operation):
        error = None
        result = None
        try:
            result = operation()
        except Exception as exc:
            error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error, group=distributed_utils.get_gloo_group())
        if any(errors):
            raise RuntimeError("Core RL validation failed: " + "; ".join(e for e in errors if e))
        return result

    def _replay_context(self, module, batch):
        context = models.replay_context(module, batch, enabled=self.args.use_rollout_routing_replay)
        if getattr(self.args.olmo_core, "replay_diagnostics", False):
            return replay_diagnostics.checked_context(self, module, batch, context)
        return context

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
            local_batch = self.args.global_batch_size // dist.get_world_size()
            self._agree(lambda: self._validate_step_batches(batches, local_batch))
            contract.validate_batch_schedule(len(batches), local_batch, batches[0]["tokens"].device)
            if self.args.use_rollout_routing_replay:
                self._agree(lambda: self._validate_replay_batches(batches))
            versions = self._agree(lambda: data.policy_versions(rollout))
            self._agree(lambda: self.clock.validate_versions(versions, self.args.olmo_core.max_policy_lag))
            decision = self._scoring_pass()
            checked = self._scoring_check_due(decision)
            standalone = decision.standalone or checked
            if not decision.standalone and len(batches) != local_batch:
                raise ValueError("Skipping the standalone scoring pass requires one optimizer step per collection")
            difference = agreement = profile = None
            if standalone:
                torch.cuda.synchronize()
                score_started = time.perf_counter()
                rollout["log_probs"] = self._score(self.train_module, batches, use_replay=True)
                torch.cuda.synchronize()
                contract.record(
                    self.args,
                    {
                        "event": "score_timing",
                        "rollout_id": rollout_id,
                        "seconds": time.perf_counter() - score_started,
                        "model_tokens": sum(batch["tokens"].numel() for batch in batches),
                        "row_specialization": self.args.olmo_core.row_specialization,
                    },
                )
                difference, agreement, profile = self._score_contract(rollout, rollout_id, "standalone")
            if self.ref_module is not None:
                rollout["ref_log_probs"] = self._score(self.ref_module, batches, use_replay=False)
            with self._actor_forward_skipped(not standalone):
                self._train_steps(rollout, rollout_id, local_batch, decision, checked, difference, agreement, profile)
            self.clock.next_rollout_id = rollout_id + 1
        self._heartbeat.bump()

    def _train_steps(self, rollout, rollout_id, local_batch, decision, checked, difference, agreement, profile):
        # Without the standalone pass, old log-probabilities come from the training
        # forward itself. That is exact only at unchanged weights, which the caller
        # guarantees by admitting exactly one optimizer step per collection.
        capture = not decision.standalone
        scoring_mode = "standalone" if decision.standalone else ("checked" if checked else "skipped")
        miles_loss.compute_advantages_and_returns(self.args, rollout)
        self._agree(lambda: contract.validate_training_data(rollout))
        batches = data.sample_batches(rollout, self.args.olmo_core.max_sequence_length)
        for start in range(0, len(batches), local_batch):
            step_batches = batches[start : start + local_batch]
            step_versions = [version for batch in step_batches for version in data.policy_versions(batch)]
            self._agree(
                lambda current=step_versions: self.clock.validate_versions(current, self.args.olmo_core.max_policy_lag)
            )
            normalization = contract.step_normalization(step_batches, self.args.global_batch_size)
            for batch in step_batches:
                batch["aux_loss_div_factor"] = normalization.auxiliary_denominator
            interval = self.args.olmo_core.diagnostic_interval
            diagnostic = interval > 0 and self.clock.completed_steps % interval == 0
            probe = contract.ParameterProbe(self.model) if diagnostic else None
            lr_used = self.lr_scheduler.get_last_lr()
            self._agree(lambda: contract.validate_schedule(self.clock, self.lr_scheduler))
            started = time.perf_counter()
            contract.auxiliary_metrics(self.model, reset=True)
            self.train_module.zero_grads()

            count = len(step_batches)

            def objective(module, batch, count=count, normalization=normalization):
                logits = self._forward(module, batch)
                if capture:
                    batch["training_log_probs"] = self._training_log_probs(logits, batch)
                loss, _, metrics = miles_loss.loss_function(
                    self.args, batch, count, logits, apply_megatron_loss_scaling=False
                )
                if self.args.calculate_per_token_loss:
                    loss = normalization.scale_token_loss(loss)
                self._agree(lambda: self._validate_loss(loss))
                return loss, training_metrics.loss_metrics(metrics, loss)

            metrics = self.train_module.train_batch_with_loss(step_batches, objective, self._replay_context)
            if capture:
                training_scores = self._agree(
                    lambda step_batches=step_batches: [
                        score for batch in step_batches for score in batch.pop("training_log_probs")
                    ]
                )
                if checked:
                    # The standalone pass anchored this update; measure how far the
                    # gradient-enabled forward drifted from it before any weight change.
                    report = self._check_training_scores(rollout, training_scores)
                    logger.info(
                        "Core scoring check: %s",
                        contract.record(
                            self.args,
                            {
                                "event": "scoring_check",
                                "rollout_id": rollout_id,
                                "step": self.clock.completed_steps,
                                **report,
                            },
                        ),
                    )
                    self._scoring_checks += 1
                else:
                    rollout["log_probs"] = training_scores
                    difference, agreement, profile = self._score_contract(rollout, rollout_id, "training_forward")
            gradient_stats = self._agree(probe.gradients) if probe is not None else None
            aux_metrics = self._agree(lambda: contract.auxiliary_metrics(self.model))
            self.train_module.optim_step()
            try:
                contract.validate_step_transition(self.clock, self.optimizer, step_batches[0]["tokens"].device)
            except ValueError:
                contract.record(
                    self.args,
                    {
                        "event": "optimizer_rejected",
                        "step": self.clock.completed_steps,
                        "local_optimizer_skipped": bool(getattr(self.optimizer, "step_skipped", False)),
                    },
                )
                raise
            self.clock.optimizer_step(True)
            self.lr_scheduler.step()
            update_stats = self._agree(probe.updates) if probe is not None else None
            logger.info(
                "Core step contract: %s",
                contract.record(
                    self.args,
                    {
                        "event": "optimizer",
                        "step": self.clock.completed_steps,
                        "rollout_id": rollout_id,
                        "normalization": vars(normalization),
                        "auxiliary_denominator": normalization.auxiliary_denominator,
                        "reduction": "token" if self.args.calculate_per_token_loss else "response",
                        "scoring_pass": scoring_mode,
                        "local_policy_objective": sum(float(m["normalized_policy_objective"]) for m in metrics),
                        "local_auxiliary_objective": aux_metrics,
                        "local_behavior_versions": sorted(set(step_versions)),
                        "local_microbatches": count,
                        "lr_used": lr_used,
                        "lr_next": self.lr_scheduler.get_last_lr(),
                        "published_step": self.clock.published_step,
                        "optimizer_skipped": False,
                        "elapsed_seconds": time.perf_counter() - started,
                        "local_pre_optimizer_gradients": gradient_stats,
                        "sampled_model_updates": update_stats,
                    },
                ),
            )
            self.train_module._trainer.global_step = self.clock.completed_steps
            losses = training_metrics.aggregate_losses(metrics)
            summary = training_metrics.step_summary(metrics, aux_metrics, time.perf_counter() - started)
            logged = self._agree(
                lambda losses=losses,
                summary=summary,
                lr_used=lr_used,
                gradient_stats=gradient_stats,
                difference=difference,
                agreement=agreement,
                profile=profile: training_metrics.log_step(
                    self.args,
                    losses=losses,
                    summary=summary,
                    scores=training_metrics.score_metrics(difference, int(agreement[1]), profile),
                    clock=self.clock,
                    rollout_id=rollout_id,
                    lr_used=lr_used,
                    lr_next=self.lr_scheduler.get_last_lr(),
                    optimizer_metrics=self.train_module._trainer.metrics,
                    gradient_stats=gradient_stats,
                )
            )
            if logged is not None:
                logger.info("Core optimizer step %s: %s", self.clock.completed_steps, logged)

    def _validate_replay_batches(self, batches):
        # Validate every rank's external routes before any EP forward collective.
        for batch in batches:
            with self._replay_context(self.train_module, batch):
                pass

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
        bucket_details = []

        def send(bucket):
            nonlocal transfer_seconds, bucket_count
            torch.cuda.synchronize()
            before = time.perf_counter()
            updater.update_bucket_weights(bucket, weight_version=self.clock.completed_steps)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - before
            transfer_seconds += elapsed
            bucket_count += 1
            detail = getattr(updater, "last_bucket_timing", None)
            if detail is not None:
                bucket_details.append({**detail, "seconds": elapsed})

        bucket, size = [], 0
        for name, tensor in models.iter_export_state(
            self.train_module,
            self.hf_config,
            stream_moe=self.args.olmo_core.stream_moe_export,
            fused_experts=self.args.olmo_core.expert_publication == "fused",
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
        repeated_version = self.clock.published_step == self.clock.completed_steps
        self.clock.published()
        if dist.get_rank() == 0:
            timings = dict(
                version=self.clock.completed_steps,
                repeated_version=repeated_version,
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
                buffer_bytes=self.args.update_weight_buffer_size,
                expert_publication=self.args.olmo_core.expert_publication,
            )
            if bucket_details:
                timings.update(
                    broadcast_seconds=sum(d["broadcast_seconds"] for d in bucket_details),
                    engine_seconds=sum(d["engine_seconds"] for d in bucket_details),
                    bucket_details=bucket_details,
                )
            logger.info(
                "Core weight publication: %s",
                json.dumps({k: v for k, v in timings.items() if k != "bucket_details"}, sort_keys=True),
            )
            if self.args.save:
                path = Path(self.args.save) / "publication.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as output:
                    output.write(json.dumps(timings) + "\n")

    def configure_publication(self, buffer_bytes):
        """Set the publication bucket size for subsequent updates; used by profiling drivers."""
        if type(buffer_bytes) is not int or buffer_bytes < 1:
            raise ValueError("Publication buffer size must be a positive integer number of bytes")
        self.args.update_weight_buffer_size = buffer_bytes
        return buffer_bytes

    def close_weight_transport(self):
        """Collectively retire the serving communicator before engines are stopped."""
        updater = getattr(self, "weight_updater", None)
        if updater is None:
            return
        group = getattr(updater, "_model_update_groups", None)
        if group is None:
            return
        pending = [
            engine.destroy_weights_update_group.remote(updater._group_name) for engine in updater.rollout_engines
        ]
        try:
            dist.destroy_process_group(group)
        finally:
            ray.get(pending)
        updater._model_update_groups = None

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
