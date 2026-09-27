"""Rolling retention of committed native Core checkpoints."""

from open_instruct.miles.configuration import config


def test_retention_does_not_change_native_writer_options():
    defaults = config.CoreConfig().checkpoint_save_options()
    retained = config.CoreConfig(checkpoint_keep_last=1, checkpoint_keep_every=20)
    assert retained.checkpoint_save_options() == defaults
    threaded = config.CoreConfig(checkpoint_thread_count=2, checkpoint_keep_last=1)
    assert threaded.checkpoint_save_options() == {**defaults, "thread_count": 2}
