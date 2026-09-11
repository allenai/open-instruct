"""Write a tiny native Mega checkpoint after the existing fixed-input optimizer gate.

Run inside the qualified Megatron image through its public MILES entrypoint.
The fixture directory must be a new private copy; source fixture data stays intact.
"""

import os
from pathlib import Path
from unittest import mock

from megatron.training import get_args
from megatron.training.checkpointing import save_checkpoint
from miles.backends.megatron_utils import model as megatron_model
from olmo_miles.evaluation import policy_contract


def main():
    root = Path(os.environ["OLMO_POLICY_CONTRACT_ROOT"])
    destination = root / "native"
    if destination.exists():
        raise FileExistsError(destination)
    original = megatron_model.train

    def train_and_save(*args, **kwargs):
        outcome = original(*args, **kwargs)
        if outcome != megatron_model.TrainStepOutcome.NORMAL:
            raise RuntimeError("Fixed-input native-save fixture update failed")
        runtime = get_args()
        runtime.save = str(destination)
        runtime.async_save = False
        runtime.use_persistent_ckpt_worker = False
        runtime.no_save_optim = False
        save_checkpoint(1, args[1], args[2], args[3], 0)
        return outcome

    with mock.patch.object(megatron_model, "train", side_effect=train_and_save):
        policy_contract.run(root)


if __name__ == "__main__":
    main()
