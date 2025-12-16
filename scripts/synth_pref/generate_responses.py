import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from open_instruct import logger_utils
from scripts.synth_pref.utils.model_configs import MODELS

logger = logger_utils.setup_logger(__name__)

load_dotenv()


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser("Generate responses for a set of prompts given a model.")
    parser.add_argument("-n", "--name", type=str, required=True, help="Name of the dataset for ID purposes.")
    parser.add_argument("-f", "--source_file", nargs="+", required=True, help="Directory that contains the JSONL files you need to generate responses for.")
    parser.add_argument("--target_dir", type=Path, required=True, help="Directory to save the output inference results.")
    parser.add_argument("--batch_size", default=128, type=int, help="Set the number of prompts to send to vLLM at a time.")
    parser.add_argument("--config_output_dir", default="configs/", help="Path to save the configuration yaml files.")
    parser.add_argument("-x", "--ignore_model", nargs="+", help="Ignore a model and don't create a config from it.")
    parser.add_argument("-y", "--include_model", nargs="+", help="List of models to include.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    model_names = list(MODELS.keys())
    if args.include_model:
        model_names = [model for model in model_names if model in args.include_model]
    if args.ignore_model:
        model_names = [model for model in model_names if model not in args.ignore_model]

    job_yaml_paths = []
    logger.info("Creating job files...")
    for model in tqdm(model_names):
        model_config = MODELS.get(model)

        model_id = model.replace("/", "___")
        task_name = f"birr-{model_id}-{args.name}"
        default_pipeline_params = {
            "pipeline": {
                "input_file_dir": args.source_file,
                "output_file_dir": args.target_dir,
                "num_workers": 1,
                "generation_batch_size": args.batch_size,
            }
        }

        job = {**model_config, **default_pipeline_params}

        # Save config to file
        output_dir = Path(args.config_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task_name}.yaml"
        with open(output_path, "w") as file:
            yaml.dump(job, file)

        job_yaml_paths.append(output_path)


if __name__ == "__main__":
    main()
