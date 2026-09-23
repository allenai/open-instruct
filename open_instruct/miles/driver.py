"""MILES orchestration with explicit Core publication and checkpoint boundaries."""

import asyncio
import os
from functools import partial

import wandb
from miles.ray import placement_group, wiring
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.utils import object_store
from miles.utils.data import remove_rollout_data_refs
from miles.utils.hf_config import HF_EXPORT_COMPLETE_MARKER
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking

from open_instruct import logger_utils
from open_instruct.miles import evaluation as background_eval
from open_instruct.miles import startup_cache, throughput
from open_instruct.miles.rolling_publication import RollingPublication
from open_instruct.miles.timing import evaluation_stage, stage

logger = logger_utils.setup_logger(__name__)


async def train(args, *, export_hf=None):
    for warning in throughput.report(vars(args), args.olmo_core)["warnings"]:
        logger.warning("Throughput [%s]: %s", warning["code"], warning["message"])
    with stage(args, "startup_cache_prepare"):
        startup_cache.prepare(args)
    with stage(args, "placement"):
        worker_manager = wiring.launch_worker_manager(
            args, transform_specs=partial(startup_cache.configure_specs, args)
        )
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)
    manager = None
    inference = None
    learner = None
    failure = None
    completed = []
    rolling = None
    refresh = args.olmo_core.publication_mode == "refresh"
    try:
        with stage(args, "serving_startup"):
            inference, manager, rollouts_per_epoch = await placement_group.create_rollout_components(args)
        with stage(args, "trainer_startup"):
            learner, _ = await placement_group.create_training_models(args, inference, manager)

        async def publish(rollout_id=None):
            if args.fully_async:
                await manager.core_publication_boundary.remote(True, **({"refresh": True} if refresh else {}))
            if args.offload_rollout:
                await inference.onload_weights()
            await asyncio.wait_for(
                placement_group.update_weights(learner, manager, rollout_id=rollout_id),
                timeout=args.olmo_core.engine_update_timeout if refresh else None,
            )
            interval = args.olmo_core.diagnostic_interval
            fresh_initial = rollout_id is None and args.start_rollout_id == 0
            diagnostic = interval > 0 and (rollout_id is None or (rollout_id + 1) % interval == 0)
            if args.check_weight_update_equal and (rollout_id is None or diagnostic):
                if not fresh_initial:
                    # The startup snapshot is the original HF checkpoint. It
                    # cannot validate trained/restored weights. Instead measure
                    # an exact current-state publication round trip, including
                    # reset so an omitted tensor cannot pass as unchanged.
                    await inference.check_weights(action="snapshot", selector=args.check_weight_update_selector)
                    await inference.check_weights(
                        action="reset_tensors",
                        selector=args.check_weight_update_selector,
                        skip_list=args.check_weight_update_skip_list,
                    )
                    await asyncio.wait_for(
                        placement_group.update_weights(learner, manager, rollout_id=rollout_id),
                        timeout=args.olmo_core.engine_update_timeout if refresh else None,
                    )
                await inference.check_weights(
                    action="compare",
                    allow_quant_error=args.check_weight_update_allow_quant_error,
                    selector=args.check_weight_update_selector,
                    skip_list=args.check_weight_update_skip_list,
                )
            if args.offload_rollout:
                await inference.onload_kv()
            if args.fully_async:
                await manager.core_publication_boundary.remote(False, **({"refresh": True} if refresh else {}))

        with stage(args, "initial_publication"):
            await publish()
        if args.olmo_core.publication_mode == "engine_drain":
            rolling = RollingPublication(args, learner, manager, inference)
            with stage(args, "engine_drain_startup"):
                await rolling.initialize()
        background = getattr(args, "background_evaluation", None)
        evaluation = None if background else EvalDispatcher(args, learner, manager)
        coordinator = None
        if background:
            tracking = {
                "id": getattr(args, "wandb_run_id", None),
                "entity": args.wandb_team,
                "project": args.wandb_project,
                "mode": args.wandb_mode,
            }
            if args.use_wandb and wandb.run is not None:
                tracking.update(
                    id=wandb.run.id,
                    entity=wandb.run.entity,
                    project=wandb.run.project,
                    mode="offline" if wandb.run.settings.mode in {"offline", "dryrun"} else "online",
                )
            coordinator = background_eval.Coordinator(
                background,
                {
                    "name": background["name"],
                    "root": background["root"],
                    "wandb": tracking,
                    "beaker_experiment_id": os.environ.get("BEAKER_EXPERIMENT_ID"),
                },
            )
            if args.start_rollout_id == 0 and background["initial"]:
                coordinator.dispatch(0, args.hf_checkpoint)
        if evaluation is not None and args.eval_interval is not None and not args.skip_eval_before_train:
            with evaluation_stage(args, args.start_rollout_id, initial=True):
                await evaluation.dispatch(
                    args.start_rollout_id, hf_dir=args.hf_checkpoint if args.start_rollout_id == 0 else None
                )
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            # In async mode the managed producer fills the bounded queue while
            # learning runs; dequeue happens only after the preceding publication.
            with stage(args, "generation_wait", rollout_id):
                if rolling is None:
                    await inference.prepare_rollout(rollout_id)
                batch = await manager.get.remote(rollout_id)
            if args.offload_rollout:
                await inference.offload()
            try:
                with stage(args, "training", rollout_id):
                    await learner.train(rollout_id, batch)
                completed.append(rollout_id)
                if coordinator is not None:
                    per_collection = args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
                    for update in range(rollout_id * per_collection + 1, (rollout_id + 1) * per_collection + 1):
                        target = background_eval.snapshot(background["root"], update)
                        if (target / HF_EXPORT_COMPLETE_MARKER).is_file():
                            coordinator.dispatch(update, target)
                if rolling is not None:
                    await rolling.optimizer_step_completed()
            finally:
                remove_rollout_data_refs(args, batch)
            if rolling is not None:
                # Publish the completed optimizer version before the next collection.
                with stage(args, "publication", rollout_id):
                    await rolling.publish()
            if refresh:
                with stage(args, "publication", rollout_id):
                    await publish(rollout_id)
            sentinel = args.save_trigger_sentinel and os.path.exists(args.save_trigger_sentinel)
            if sentinel or should_run_periodic_action(
                rollout_id, args.save_interval, rollouts_per_epoch, args.num_rollout
            ):
                # The async data source snapshots its cursor and pristine pending
                # prompt ledger under one lock. Completed-but-unused and in-flight
                # groups regenerate on resume; live inference need not finish or
                # pause. No new training batch is consumed until this save commits.
                with stage(args, "checkpoint", rollout_id):
                    await manager.save.remote(rollout_id)
                    await learner.save_model(rollout_id, force_sync=True)
                    await learner.finalize_checkpoint(rollout_id)
                if sentinel:
                    os.remove(args.save_trigger_sentinel)
                # Warm compiler caches now exist; publish them in the background
                # so a later preemption does not discard this process's warmup.
                startup_cache.publish_progress(args, rollout_id)
            if rolling is None and not refresh and (rollout_id + 1) % args.update_weights_interval == 0:
                with stage(args, "publication", rollout_id):
                    await publish(rollout_id)
            if evaluation is not None and should_run_periodic_action(
                rollout_id, args.eval_interval, rollouts_per_epoch, args.num_rollout
            ):
                with evaluation_stage(args, rollout_id):
                    await evaluation.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)
            if (
                args.debug_exit_after_rollout is not None
                and rollout_id - args.start_rollout_id + 1 >= args.debug_exit_after_rollout
            ):
                break
        if evaluation is not None:
            await evaluation.drain()
        # A deliberate debug stop leaves a resumable workflow, not a final export.
        # Otherwise the first process creates the final directory and the resumed
        # process cannot export its newer weights there (the exporter is exclusive).
        reached_end = (completed[-1] + 1 if completed else args.start_rollout_id) == args.num_rollout
        if export_hf is not None and reached_end:
            if args.fully_async:
                await manager.core_publication_boundary.remote(True)
            with stage(args, "final_hf_export"):
                await learner.export_hf(completed[-1] if completed else args.start_rollout_id - 1, export_hf)
    except BaseException as error:
        failure = error
        raise
    finally:
        cleanup_error = None
        if rolling is not None:
            try:
                await rolling.close(failed=failure is not None)
            except BaseException as error:
                cleanup_error = error
                logger.exception("Engine drain cleanup failed")
        for component, operation, timeout, cleanup_stage in (
            # Async generation must stop, but the servers must remain alive while
            # both sides collectively destroy the weight-update NCCL group.
            (
                manager if args.fully_async else None,
                lambda: manager.core_publication_boundary.remote(True),
                (
                    args.olmo_core.engine_drain_timeout + args.olmo_core.engine_update_timeout
                    if rolling or refresh
                    else 60
                ),
                "final_generation_drain",
            ),
            (learner, lambda: learner.execute_workers("close_weight_transport"), 60, "close_weight_transport"),
            (manager, lambda: manager.dispose.remote(), 120, "rollout_dispose"),
            (learner, lambda: learner.dispose(), 60, "trainer_dispose"),
            (inference, lambda: inference.dispose(), 60, "inference_dispose"),
            (worker_manager, lambda: worker_manager.dispose.remote(), 120, "worker_dispose"),
        ):
            if component is None:
                continue
            try:
                with stage(args, cleanup_stage):
                    await asyncio.wait_for(operation(), timeout=timeout)
            except BaseException as error:
                cleanup_error = cleanup_error or error
                logger.exception("Core RL cleanup %s failed (deadline=%ss)", cleanup_stage, timeout)
        try:
            await startup_cache.finish(args, success=failure is None and cleanup_error is None)
        except Exception:
            logger.exception("Optional compiler-cache finalization failed")
        finish_tracking()
        if failure is None and cleanup_error is not None:
            raise cleanup_error
    return {"completed_rollout_ids": completed, "start_rollout_id": args.start_rollout_id, "export_hf": export_hf}
