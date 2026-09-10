"""MILES orchestration with explicit Core publication and checkpoint boundaries."""

import asyncio
import os

from miles.ray import placement_group
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.utils import object_store
from miles.utils.data import remove_rollout_data_refs
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


async def train(args):
    groups = placement_group.create_placement_groups(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)
    manager = None
    learner = None
    failure = None
    try:
        manager, rollouts_per_epoch = placement_group.create_rollout_manager(args, groups["rollout"])
        learner, _ = await placement_group.create_training_models(args, groups, manager)

        async def publish(rollout_id=None):
            if args.fully_async:
                await manager.core_publication_boundary.remote(True)
            if args.offload_rollout:
                await manager.onload_weights.remote()
            await learner.update_weights(rollout_id)
            interval = args.olmo_core.diagnostic_interval
            fresh_initial = rollout_id is None and args.start_rollout_id == 0
            diagnostic = interval > 0 and (rollout_id is None or (rollout_id + 1) % interval == 0)
            if args.check_weight_update_equal and (rollout_id is None or diagnostic):
                if not fresh_initial:
                    # The startup snapshot is the original HF checkpoint. It
                    # cannot validate trained/restored weights. Instead measure
                    # an exact current-state publication round trip, including
                    # reset so an omitted tensor cannot pass as unchanged.
                    await manager.check_weights.remote(action="snapshot", selector=args.check_weight_update_selector)
                    await manager.check_weights.remote(
                        action="reset_tensors",
                        selector=args.check_weight_update_selector,
                        skip_list=args.check_weight_update_skip_list,
                    )
                    await learner.update_weights(rollout_id)
                await manager.check_weights.remote(
                    action="compare",
                    allow_quant_error=args.check_weight_update_allow_quant_error,
                    selector=args.check_weight_update_selector,
                    skip_list=args.check_weight_update_skip_list,
                )
            if args.offload_rollout:
                await manager.onload_kv.remote()
            if args.fully_async:
                await manager.core_publication_boundary.remote(False)

        await publish()
        evaluation = EvalDispatcher(args, learner, manager)
        if args.eval_interval is not None and not args.skip_eval_before_train:
            await evaluation.dispatch(
                args.start_rollout_id, hf_dir=args.hf_checkpoint if args.start_rollout_id == 0 else None
            )
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            # In async mode the managed producer fills the bounded queue while
            # learning runs; dequeue happens only after the preceding publication.
            batch = await manager.generate.remote(rollout_id)
            if args.offload_rollout:
                await manager.offload.remote()
            try:
                await learner.train(rollout_id, batch)
            finally:
                remove_rollout_data_refs(args, batch)
            sentinel = args.save_trigger_sentinel and os.path.exists(args.save_trigger_sentinel)
            if sentinel or should_run_periodic_action(
                rollout_id, args.save_interval, rollouts_per_epoch, args.num_rollout
            ):
                await manager.save.remote(rollout_id)
                await learner.save_model(rollout_id, force_sync=True)
                await learner.finalize_checkpoint(rollout_id)
                if sentinel:
                    os.remove(args.save_trigger_sentinel)
            if (rollout_id + 1) % args.update_weights_interval == 0:
                await publish(rollout_id)
            if should_run_periodic_action(rollout_id, args.eval_interval, rollouts_per_epoch, args.num_rollout):
                await evaluation.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)
            if (
                args.debug_exit_after_rollout is not None
                and rollout_id - args.start_rollout_id + 1 >= args.debug_exit_after_rollout
            ):
                break
        await evaluation.drain()
    except BaseException as error:
        failure = error
        raise
    finally:
        cleanup_error = None
        for component, operation, timeout in (
            # Async generation must stop, but the servers must remain alive while
            # both sides collectively destroy the weight-update NCCL group.
            (manager if args.fully_async else None, lambda: manager.core_publication_boundary.remote(True), 60),
            (learner, lambda: learner._broadcast("close_weight_transport"), 60),
            (manager, lambda: manager.dispose.remote(), 120),
            (learner, lambda: learner.dispose(), 60),
        ):
            if component is None:
                continue
            try:
                await asyncio.wait_for(operation(), timeout=timeout)
            except BaseException as error:
                cleanup_error = cleanup_error or error
                logger.exception("Core RL component cleanup failed")
        finish_tracking()
        if failure is None and cleanup_error is not None:
            raise cleanup_error
