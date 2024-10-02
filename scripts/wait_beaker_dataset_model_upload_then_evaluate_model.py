import sys
import time
from dataclasses import dataclass

from open_instruct.utils import (
    ArgumentParserPlus,
    BeakerRuntimeConfig,
    beaker_experiment_succeeded,
    get_beaker_dataset_ids,
    submit_beaker_eval_jobs,
)

"""
example usage
python scripts/wait_beaker_dataset_model_upload_then_evaluate_model.py --beaker_workload_id 01J6ASGRDTH9G9DWF9FPDWM6C1 --model_name "eval_beaker_experiment_until_finished"
"""


@dataclass
class Args:
    model_name: str
    max_wait_time_for_beaker_dataset_upload_seconds: int = 60 * 30  # 30 minutes
    check_interval_seconds: int = 60


def main(args: Args, beaker_runtime_config: BeakerRuntimeConfig):
    print(args)

    start_time = time.time()
    while time.time() - start_time < args.max_wait_time_for_beaker_dataset_upload_seconds:
        if beaker_experiment_succeeded(beaker_runtime_config.beaker_workload_id):
            print("Experiment succeeded")
            # NOTE: we are assuming the first beaker dataset has the model
            # I have checked a couple of beaker jobs and found the first dataset is the model
            # but we should check this assumption
            beaker_dataset_ids = get_beaker_dataset_ids(beaker_runtime_config.beaker_workload_id, sort=True)
            submit_beaker_eval_jobs(
                model_name=args.model_name,
                location=beaker_dataset_ids[-1],
                run_oe_eval_experiments=True,
                run_safety_evaluations=True,
                skip_oi_evals=True,
            )
            return
        time.sleep(args.check_interval_seconds)
    # If we reach here, the experiment failed
    print("Experiment failed")
    sys.exit(1)  # submit eval failed


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, BeakerRuntimeConfig))
    main(*parser.parse())
