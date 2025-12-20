import subprocess
import sys
import time
from dataclasses import dataclass

from open_instruct.utils import (
    ArgumentParserPlus,
    BeakerRuntimeConfig,
    beaker_experiment_succeeded,
    get_beaker_dataset_ids,
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
    upload_to_hf: str = "allenai/tulu-3-evals"
    run_id: str | None = None


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
            command = f"""
            python scripts/submit_eval_jobs.py \
                --model_name {args.model_name} \
                --location {beaker_dataset_ids[-1]} \
                --is_tuned \
                --workspace tulu-3-results \
                --preemptible \
                --use_hf_tokenizer_template \
                --beaker_image nathanl/open_instruct_auto \
                --skip_oi_evals \
                --run_oe_eval_experiments \
                --upload_to_hf {args.upload_to_hf}"""
            if args.run_id:
                command += f" --run_id {args.run_id}"

            process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            print(f"Beaker evaluation jobs: Stdout:\n{stdout.decode()}")
            print(f"Beaker evaluation jobs: Stderr:\n{stderr.decode()}")
            print(f"Beaker evaluation jobs: process return code: {process.returncode}")

            safety_command = f"""
            python scripts/submit_eval_jobs.py \
                --model_name {args.model_name} \
                --location {beaker_dataset_ids[-1]} \
                --is_tuned \
                --workspace tulu-3-results \
                --preemptible \
                --use_hf_tokenizer_template \
                --beaker_image nathanl/open_instruct_auto \
                --skip_oi_evals \
                --run_oe_eval_experiments \
                --oe_eval_task_suite "SAFETY_EVAL" \
                --upload_to_hf {args.upload_to_hf}"""
            if args.run_id:
                safety_command += f" --run_id {args.run_id}"

            safety_process = subprocess.Popen(
                ["bash", "-c", safety_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            safety_stdout, safety_stderr = safety_process.communicate()

            print(f"Beaker safety evaluation jobs: Stdout:\n{safety_stdout.decode()}")
            print(f"Beaker safety evaluation jobs: Stderr:\n{safety_stderr.decode()}")
            print(f"Beaker safety evaluation jobs: process return code: {safety_process.returncode}")

            return
        time.sleep(args.check_interval_seconds)
    # If we reach here, the experiment failed
    print("Experiment failed")
    sys.exit(1)  # submit eval failed


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, BeakerRuntimeConfig))
    main(*parser.parse())
