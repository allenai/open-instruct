"""Runs MaxText train_sft with optional perf-arm patches (H211 mk_perf.py), then argv as train_sft would get it.
  SKIP_FINAL_SAVE=1  no forced final checkpoint (Tunix _save_last_checkpoint), for short perf arms.
  FORCE_NNX_CACHE=1  train with Tunix cache_nnx_graph=True (MaxText turns it off for MoE models), so the NNX graph is not
                     re-split/merged on the host every step. Experimental: the reason it is off is a scan metadata issue.
  DEFER_AUX_LOG=1    log LOSS-PARTS / REQUIRED_RBF one step late: reading the current step's aux on host 0 blocks until
                     the step finishes, so host work for the next step cannot overlap it. Same lines, same values.
  SKIP_IDLE_SAVE=1   call Tunix CheckpointManager.save only on steps the save interval selects (or forced saves). Tunix
                     calls it every step, and each call flattens the whole NNX model and optimizer state on the host
                     (~0.5 s per step, up to ~3 s when a GC pass lands inside; x1d profile, 2026-11-14). Checkpoints
                     written are the same; Orbax's preemption-triggered saves on other steps are not attempted.
                     With MAXTEXT_DATA_STATE_DIR set, each save first writes this host's data-stream position
                     (fast resume, see maxtext input_pipeline/multihost_dataloading.write_data_state).
  GC_FREEZE=1        gc.collect() + gc.freeze() once before the train loop, so the cyclic GC's full passes no longer
                     rescan the long-lived model/optimizer object graph (the every-~7-step ~2.6 s host stall).
  GC_MANUAL=N        with GC_FREEZE: also gc.disable() and run gc.collect() on every N-th Tunix save() call (once per
                     step), so no automatic collection lands inside a step's host work (x1e still had ~1.6 s every ~12).
"""
import gc
import os
import sys
import time

from absl import app
from maxtext.input_pipeline import multihost_dataloading
from maxtext.trainers.post_train.sft import train_sft as T
from tunix.sft import checkpoint_manager, peft_trainer


def main():
    if os.environ.get("SKIP_FINAL_SAVE") == "1":
        peft_trainer.PeftTrainer._save_last_checkpoint = lambda self: None

    if os.environ.get("DEFER_AUX_LOG") == "1":
        orig_post, orig_close = T.MaxTextPeftTrainer._post_process_train_step, T.MaxTextPeftTrainer.close

        def _flush(self):
            prev = getattr(self, "_deferred_aux", None)
            if prev is not None:
                aux, step = prev
                saved, self._train_steps = self._train_steps, step
                try:
                    orig_post(self, aux)
                finally:
                    self._train_steps = saved
                self._deferred_aux = None

        def deferred_post(self, aux):
            _flush(self)
            self._deferred_aux = (aux, self._train_steps) if aux else None

        def close(self):
            _flush(self)
            return orig_close(self)

        T.MaxTextPeftTrainer._post_process_train_step = deferred_post
        T.MaxTextPeftTrainer.close = close
        print("perf_wrapper: DEFER_AUX_LOG on", flush=True)

    if os.environ.get("SKIP_IDLE_SAVE") == "1":
        orig_save = checkpoint_manager.CheckpointManager.save

        # Tunix resolves save_interval_steps into a save_decision_policy, so take the period from the MaxText argv.
        periods = [int(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("checkpoint_period=")]
        period = periods[-1] if periods else None
        print(f"perf_wrapper: SKIP_IDLE_SAVE period={period}", flush=True)

        def save(self, step, model, optimizer=None, save_only_lora_params=False, force=False, custom_metadata=None):
            if not force and period and step % period != 0:
                return False
            if os.environ.get("MAXTEXT_DATA_STATE_DIR"):
                # Fast-resume data state (raw rows consumed through this step), written before the checkpoint.
                multihost_dataloading.write_data_state(step)
            return orig_save(self, step, model, optimizer, save_only_lora_params=save_only_lora_params, force=force,
                             custom_metadata=custom_metadata)

        checkpoint_manager.CheckpointManager.save = save
        print("perf_wrapper: SKIP_IDLE_SAVE on", flush=True)

    if os.environ.get("GC_FREEZE") == "1":
        orig_train = peft_trainer.PeftTrainer.train

        def train(self, *args, **kwargs):
            gc.collect()
            gc.freeze()
            print(f"perf_wrapper: GC_FREEZE froze {gc.get_freeze_count()} objects", flush=True)
            if gc_every > 0:
                gc.disable()
                print(f"perf_wrapper: GC_MANUAL gc.disable(), collect every {gc_every} steps", flush=True)
            return orig_train(self, *args, **kwargs)

        peft_trainer.PeftTrainer.train = train

        gc_every = int(os.environ.get("GC_MANUAL", "0") or 0)
        if gc_every > 0:
            inner_save = checkpoint_manager.CheckpointManager.save  # SKIP_IDLE_SAVE's wrapper when on (patched above)

            def gc_save(self, step, *args, **kwargs):
                if step % gc_every == 0:
                    t0 = time.perf_counter()
                    n = gc.collect()
                    if step % (gc_every * 10) == 0:
                        print(f"perf_wrapper: GC_MANUAL step {step} collected {n} in {time.perf_counter() - t0:.3f} s", flush=True)
                return inner_save(self, step, *args, **kwargs)

            checkpoint_manager.CheckpointManager.save = gc_save

    if os.environ.get("FORCE_NNX_CACHE") == "1":

        def train_model(mt_config, trainer, mesh):
            with T.jax.set_mesh(mesh), T.logical_axis_rules(mt_config.logical_axis_rules):
                trainer.train(trainer.data_hooks.train_data_iterator, trainer.data_hooks.eval_data_iterator, cache_nnx_graph=True)
            return trainer

        T.train_model = train_model
        print("perf_wrapper: FORCE_NNX_CACHE on", flush=True)

    sys.argv[0] = "train_sft"
    app.run(T.main)


# Guarded: grain's data worker is a spawned process that re-imports __main__; unguarded, it re-ran the whole
# trainer and re-joined jax.distributed with the same task id (p7, 2026-11-12: "different incarnation" abort).
if __name__ == "__main__":
    main()
