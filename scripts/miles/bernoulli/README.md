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
