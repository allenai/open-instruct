"""Core actor selecting the application router-freeze hook in its own process."""

from miles.backends.core_utils import actor, moe_models

from open_instruct.miles.execution import frozen_router


class FrozenRouterTrainActor(actor.OLMoCoreTrainRayActor):
    def init(self, args, *positional, **kwargs):
        if not args.olmo_core.freeze_router:
            raise ValueError("FrozenRouterTrainActor requires core.freeze_router")
        frozen_router.install(moe_models)
        return super().init(args, *positional, **kwargs)
