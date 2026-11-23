"""Generate the v1 configs (4+4 GPUs, 8k context) from the completed v0 config by exact, asserted edits.

  test : 3-update qualification (weight equality every update, stop after 2, resume for 3)
  a    : control, full dolci-math-base-r1-v1, no filtering
  b    : offline base-model pass-band filtered data (--b-data)

Usage: python make_v1_configs.py RUNS_DIR --b-data /path/to/filtered/dataset/dir
"""

import argparse
from pathlib import Path

V0 = "olmicro-2.7b-math-rl-v0-r1-20261118"
FULL = "/weka/oe-adapt-default/jacobm/olmicro/data/dolci-math-base-r1-v1"

SHARED = [
    ("[trainer]\n# Native checkpoints only resume on the same trainer world size; keep this fixed for the run.\ngpus = 2",
     "[trainer]\n# Native checkpoints only resume on the same trainer world size; keep this fixed for the run.\ngpus = 4"),
    ("packing_max_tokens = 8192", "packing_max_tokens = 16384"),
    ("gpus = 6\n", "gpus = 4\n"),
    ("max_response_length = 4096\nmax_context_length = 4608", "max_response_length = 7680\nmax_context_length = 8192"),
    ("check_weight_update_equal = false", "check_weight_update_equal = true"),
    ("async_max_concurrent_samples = 384", "async_max_concurrent_samples = 256"),
    ("n_samples_per_eval_prompt = 1", "n_samples_per_eval_prompt = 4"),
    ("eval_max_response_len = 4096", "eval_max_response_len = 7680"),
]


def edit(text, pairs):
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"expected exactly one occurrence of {old!r}")
        text = text.replace(old, new)
    return text


def header(lines):
    return "".join(f"# {line}\n" for line in lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--b-data", type=Path)
    args = parser.parse_args()
    source = (args.runs / f"{V0}.toml").read_text()
    body = source[source.index("schema_version") :]
    base = edit(body, SHARED)

    configs = {}
    name = "olmicro-2.7b-math-4x4-8k-test-20261118"
    configs[name] = header([
        "Qualification for the v1 topology/length: 4 trainer + 4 inference GPUs, 8,192 context, 16k packing.",
        "Weight equality at startup and after every update; stop after update 2, relaunch the same file to resume for 3.",
    ]) + edit(base.replace(V0, name), [
        ("export_hf = true", "export_hf = false"),
        ("# Observability only (never changes admission or training)", "# Records off for the qualification run.\n# Observability only (never changes admission or training)"),
        ("[records]\nenabled = true", "[records]\nenabled = false"),
        ('timeout = "24h"', 'timeout = "2h"'),
        ("max_retries = 3", "max_retries = 0"),
        ("num_rollouts = 400", "num_rollouts = 3"),
        ("skip_eval_before_train = false", "skip_eval_before_train = true"),
        ("eval_interval = 50", "eval_interval = 100"),
        ("save_interval = 50", "save_interval = 1"),
        ("diagnostic_interval = 0", "diagnostic_interval = 1"),
        ("[miles]\n", "[miles]\ndebug_exit_after_rollout = 2\n"),
    ])

    common = [
        "v1: R1-template RL-zero on the 2.7B base, 4 trainer + 4 inference B300s, 7,680-token responses / 8,192 context,",
        "4 eval samples per prompt; weight equality checked at startup. Same settings as arm B except the training data.",
    ]
    name = "olmicro-2.7b-math-rl-v1a-full-20261118"
    configs[name] = header(["Arm A (control): full dolci-math-base-r1-v1 training set, no difficulty filtering.", *common]) + base.replace(V0, name)

    if args.b_data:
        name = "olmicro-2.7b-math-rl-v1b-passband-20261118"
        b = base.replace(V0, name).replace(FULL, str(args.b_data))
        if (args.b_data / "train.jsonl").is_file() is False:
            raise SystemExit(f"missing {args.b_data}/train.jsonl")
        configs[name] = header([f"Arm B: offline base-model pass-band filtered training set {args.b_data.name}.", *common]) + b

    for name, text in configs.items():
        path = args.runs / f"{name}.toml"
        if path.exists():
            if path.read_text() != text:
                raise SystemExit(f"refusing to overwrite a different {path}")
            print("unchanged", path)
            continue
        path.write_text(text)
        print("wrote", path)


if __name__ == "__main__":
    main()
