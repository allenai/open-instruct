# H235 binary-reference normalization

Experimental arm `B512-Bernoulli`, based on the original H235 application commit
`a87a0feac620d96094cbab241e04eb402c058780` and immutable image
`01M7A0QM57N9499PGFAT73SMF0`. No current-main defaults are imported.

The denominator is `sqrt(n/(n-1) * mean_reward * (1-mean_reward)) + 1e-6`.
It replaces only the native sample standard deviation; prompt/rollout grouping,
centering, epsilon, raw reward recording, reward gates and filtering are retained.
Requires finite rewards in [0,1]. Binary groups agree with ordinary GRPO within
floating-point tolerance. Constant groups provide no contrast. No global
advantage-mass correction or learning-rate adjustment is applied.

Use both existing B512-Noff and B512-Non as controls. This arm starts from the
same SFT checkpoint (not a trained control checkpoint): Dolci-only priority
sampling, seed 17, eight responses per prompt, batch 512, LR 1e-6, 400 updates,
response-mean loss, refresh publication and lag 12. Evaluations and checkpoint
exports retain the original 100-update schedule. Generation/consumption is
asynchronous, so matching seeds does not imply paired trajectories.

Deployment uses a committed Beaker source dataset with a SHA256 manifest, copied
to a fresh content-addressed WEKA directory by a CPU qualification job. BASH_ENV
installs it into the job-local original image, checking the original native
module SHA256 before replacing its one standard-deviation expression. This
avoids unrelated runtime changes. No existing job or shared runtime is patched.
The native MILES source is pinned at `22d2fb4ca0f87255b56ec0cd733f487cca22ebc7`.

Run tests with `OI_BERNOULLI_NATIVE_SOURCE=/path/to/pinned/train_data_conversion.py
python -m pytest scripts/miles/bernoulli/test_bernoulli.py`. The integration tests
execute the actual pinned native reward functions, original and patched.

The launch config, source manifest, control diff and receipts are retained in
`runs/` locally and in the immutable experiment source dataset. Submit through
`MILES_EXISTING_IMAGE=01M7A0QM57N9499PGFAT73SMF0
./scripts/train/build_image_and_launch.sh --miles runs/bernoulli.json`.

## Launch record (2026-11-16 UTC)

- Implementation commit: `66460e4ecf74ce4f56def1ec1d8592bacf82c4f9`.
- Training: https://beaker.org/ex/01M7HKY55CSC5ZDES401SNQG4G
- Runtime qualification: https://beaker.org/ex/01M7HKWCYKEC818FQ1MK0ZMM2N
  (exit 0; 16 tests in the original application image; input paths and LR checked).
- Immutable source, original control config, arm config and explicit diff:
  https://beaker.org/ds/01M7HKV3AEM4ZW6FBGYNG9J84N
- Source manifest SHA256:
  `41a3de9d48dc575dbe952da4207ada4d5d4cf63cd9650ac037842cde4636e500`.
- Patched native module SHA256:
  `9e95471f74cd0c940b12c87e674aa30dec2b349650f824b670ddf86a6ce64c38`.
- Submitted RunSpec SHA256:
  `9c7bb6d3761efa0bf4635ed7cc136d00cf77d1395bc72348ff0b99b151cbd1aa`.
- Local validation: 139 tests passed (normalization and RunSpec), repository Ruff
  check passed, changed Python formatting passed, keyword check passed.
- Requested topology: one 8-GPU Holmes task, 2 trainers / 5 rollout GPUs /
  1 managed judge GPU; normal priority, 2h protection, auto-resume, 8h timeout.
- Submitted status: queued; scheduler reports insufficient available slots and
  higher-ranked jobs. No optimizer steps or evaluation results established yet.
- Full compiled-option comparison: only `grpo_std_normalization`, `save`,
  `save_debug_rollout_data`, `wandb_dir`, and `wandb_run_name` differ from Noff.
  The verified installer changes the normalization denominator in the runtime.

The original 7h graceful segment limit is retained. A successful segment exit
before update 400 is not completion of this arm; it needs a continuation from
its own checkpoint. Beaker auto-resume handles preemption, not this graceful
budget stop. Compare evaluation truncation counts, not filtered trainer batches.
