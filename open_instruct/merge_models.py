import copy
import subprocess
import yaml
from datetime import datetime
import argparse
import re 
import shlex

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def main():
    parser = argparse.ArgumentParser(description="Run experiment with Beaker config")
    # TODO: new, need to complete
    parser.add_argument("--merge_method", type=str, default="linear", help="Merge method to use")


    # TODO: old, need to prune
    parser.add_argument("--default_beaker_config", default="configs/beaker_configs/default_finetune.yaml", 
                        help="Path to the default Beaker config file")
    parser.add_argument("--config", default=None, 
                        help="Path to an additional config file to override default settings")
    # parser.add_argument("--wandb_api_key", required=False, help="Weights & Biases API key")
    parser.add_argument("--cluster", type=str, default="ai2/allennlp-cirrascale", help="Beaker cluster to use")
    parser.add_argument("--priority", type=str, default="high", help="Priority of the job")
    parser.add_argument("--preemptible", type=bool, default=True, help="Whether to use preemptible instances")
    parser.add_argument("--workspace", type=str, default="ai2/tulu-3-dev", help="Beaker workspace to use.")

    # Structure:
    # 1. Parse model inputs
    # 2. Build merge config
    # 3. Launch merge
        # wait for it to complete successfully
    # 4. If --run_evals, require model name (check at beginning)
    # 5. Launch evals

"""
Mergekit Options:
  -v, --verbose                   Verbose logging
  --allow-crimes / --no-allow-crimes
                                  Allow mixing architectures  [default: no-
                                  allow-crimes]
  --transformers-cache TEXT       Override storage path for downloaded models
  --lora-merge-cache TEXT         Path to store merged LORA models
  --cuda / --no-cuda              Perform matrix arithmetic on GPU  [default:
                                  no-cuda]
  --low-cpu-memory / --no-low-cpu-memory
                                  Store results and intermediate values on
                                  GPU. Useful if VRAM > RAM  [default: no-low-
                                  cpu-memory]
  --out-shard-size SIZE           Number of parameters per output shard
                                  [default: 5B]
  --copy-tokenizer / --no-copy-tokenizer
                                  Copy a tokenizer to the output  [default:
                                  copy-tokenizer]
  --clone-tensors / --no-clone-tensors
                                  Clone tensors before saving, to allow
                                  multiple occurrences of the same layer
                                  [default: no-clone-tensors]
  --trust-remote-code / --no-trust-remote-code
                                  Trust remote code from huggingface repos
                                  (danger)  [default: no-trust-remote-code]
  --random-seed INTEGER           Seed for reproducible use of randomized
                                  merge methods
  --lazy-unpickle / --no-lazy-unpickle
                                  Experimental lazy unpickler for lower memory
                                  usage  [default: no-lazy-unpickle]
  --write-model-card / --no-write-model-card
                                  Output README.md containing details of the
                                  merge  [default: write-model-card]
  --safe-serialization / --no-safe-serialization
                                  Save output in safetensors. Do this, don't
                                  poison the world with more pickled models.
                                  [default: safe-serialization]
  --quiet / --no-quiet            Suppress progress bars and other non-
                                  essential output  [default: no-quiet]
  --read-to-gpu / --no-read-to-gpu
                                  Read model weights directly to GPU
                                  [default: no-read-to-gpu]
  --help                          Show this message and exit.
  """