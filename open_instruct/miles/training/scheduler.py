"""MILES learning-rate policy for Core's native optimizer parameter groups."""

from miles.backends.fsdp_utils import lr_scheduler


class CoreLRScheduler:
    """Reuse MILES' LR calculation without requiring a torch Optimizer subclass.

    OLMoDDPOptimizer owns its fused update and exposes parameter groups directly.
    The Core train module remains the only owner of optimizer stepping.
    """

    def __init__(self, args, optimizer):
        self.optimizer = optimizer
        args.train_iters = (
            args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
        )
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        self.init_lr = args.lr_warmup_init
        self.max_lr = float(args.lr)
        self.min_lr = args.min_lr
        self.lr_decay_steps = args.lr_decay_iters
        self.lr_warmup_steps = (
            args.lr_warmup_fraction * self.lr_decay_steps
            if args.lr_warmup_fraction is not None
            else args.lr_warmup_iters
        )
        self.lr_decay_style = args.lr_decay_style
        self.wsd_decay_steps = args.lr_wsd_decay_iters
        self.lr_wsd_decay_style = args.lr_wsd_decay_style
        if not 0 <= self.min_lr <= self.max_lr or self.init_lr > self.max_lr:
            raise ValueError("Invalid learning-rate bounds")
        if not 0 <= self.lr_warmup_steps < self.lr_decay_steps:
            raise ValueError("Learning-rate warmup must end before decay")
        if self.lr_decay_style == "WSD" and self.wsd_decay_steps is None:
            raise ValueError("WSD requires lr_wsd_decay_iters")
        if args.override_lr_scheduler and args.use_checkpoint_lr_scheduler:
            raise ValueError("Cannot both override and use checkpoint learning-rate scheduler")
        self.last_epoch = -1
        self.step()

    def get_lr(self):
        return [lr_scheduler.FSDPLRScheduler._get_lr_for_group(self, group) for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self):
        self.last_epoch += 1
        for group, value in zip(self.optimizer.param_groups, self.get_lr(), strict=True):
            group["lr"] = value

    def state_dict(self):
        return {key: value for key, value in vars(self).items() if key != "optimizer"}

    def load_state_dict(self, state):
        self.__dict__.update({key: value for key, value in state.items() if key != "optimizer"})
        for group, value in zip(self.optimizer.param_groups, self.get_lr(), strict=True):
            group["lr"] = value
