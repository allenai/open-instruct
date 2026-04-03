# NEED TO CHANGE:
# 3. model path
# 5. identify which evals are missing, and map back to og task
# 6. then convert that into a script to launch missing evals

"""
Simple script to pull experiment results from the OE eval datalake into a CSV.
One row per model (per group), one column per task.

For each model, also queries <MODEL_NAME>-2 and <MODEL_NAME>-3 variants.
If all 3 runs have complete results, a <MODEL_NAME>-average row is added.

Edit MODEL_CONFIGS below to configure your query.
"""

import csv
import os
import requests
from collections import OrderedDict
from dataclasses import dataclass, field

BASE_URL = "https://oe-eval-datalake.allen.ai"
FROM_DATE = "2024-07-01"
DEBUG = False
COMPUTE_AVERAGE_ROWS = False

# Suffixes for repeated runs (first run has no suffix)
RUN_SUFFIXES = [""] #, "-2", "-3"]

# --- Configure your query here ---

@dataclass
class ModelConfig:
    model_name: str
    short_description: str = ""
    long_description: str = ""
    model_path: str = ""
    groups: list[str] = field(default_factory=lambda: ["default"])

MODEL_CONFIGS = [
    ModelConfig(
        model_name="olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr_step_350",
        short_description="Base model",
        long_description="OLMo2 Flex base model fine-tuned with Tulu3 pipeline, excluding code and math data, with DPO(???) and RLVR at step 350.",
        groups=["default"],
    ),
    ModelConfig(
        model_name="math-base",
        short_description="Base 2x7B Flex model",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-2x7b-math-sft",
        short_description="Flex 2x7B Math ONLY",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-2x7b-math-sft-mixed",
        short_description="Flex 2x7B Olmo 2 Math Mixed",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head_step_500",
        short_description="Flex 2x7B Olmo 2 Math Mixed + RL",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-2x7b-math_base-olmo3_safety",
        short_description="Flex 2x7B Safety ONLY",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-2x7b-math_base-olmo3_safety-general-mix",
        short_description="FlexOLMo 2x7B, safety mix",
        long_description="FlexOLMo 2x7B without annealing, trained on safety general mix.",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-2x7b-code-sft-mixed",
        short_description="FlexOLMo 2x7B, olmo 2 anneal + sft code",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-2x7b-olmo3_code_anneal-olmo3_coding",
        short_description="FlexOLMo 2x7B, Olmo 3 code Anneal + Olmo 3 Code only sft",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7_step_200",
        short_description="FlexOLMo 2x7B, Olmo 3 Code SFT Expert + Olmo 3 Code RL",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-2x7b-no_anneal-tool_use_general_mix-unf-lm-head",
        short_description="FlexOLMo 2x7B, tool use mix",
        long_description="FlexOLMo 2x7B without annealing, trained on tool use general mix with unfrozen LM head.",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-2x7b-math_base-olmo3_tool_use-FIXED",
        short_description="FlexOLMo 2x7B, tool use only, working",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-2x7b-no_anneal-tool_use_general_mix",
        short_description="FlexOLMo 2x7B, tool use general mix, broken",
        long_description="",
        groups=["default"],
    ),
    # ModelConfig(
    #     model_name="flexolmo-2x7b-olmo3_50b_math_anneal-general-olmo3_math-mix-4k",
    #     short_description="FlexOLMo 2x7B, olmo 3 anneal + olmo 3 math general sft mix",
    #     long_description="",
    #     groups=["default"],
    # ),
    # ModelConfig(
    #     model_name="flexolmo-2x7b-olmo3_50b_math_anneal-olmo3_math-mix-4k",
    #     short_description="FlexOLMo 2x7B, olmo 3 anneal + olmo 3 math sft",
    #     long_description="",
    #     groups=["default"],
    # ),
    # ModelConfig(
    #     model_name="grpo_math_only_flex-2x7b-50b_ol3_ann-ol3_sft_math-6e-7-unf_step_200",
    #     short_description="",
    #     long_description="",
    #     groups=["default"],
    # ),
    # ModelConfig(
    #     model_name="grpo_math_only_flex-2x7b-50b_ol3_ann-ol3_sft_math-6e-7-unf_step_250",
    #     short_description="",
    #     long_description="",
    #     groups=["default"],
    # ),
    ModelConfig(
        model_name="flexolmo-5x7B-math_rl-olmo2_code_sft-tool_use_sft-safety_sft-0.05-1e-4",
        short_description="best 5x7b, olmo 2 code expert instead of 3",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4",
        short_description="best 5x7b",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4-4-active-2",
        short_description="5x7b, 5% RT, 4 active experts",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4-3-active-2",
        short_description="5x7b, 5% RT, 3 active experts",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4-2-active-2",
        short_description="5x7b, 5% RT, 2 active experts",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4-1-active-2",
        short_description="5x7b, 5% RT, 1 active expert",
        long_description="",
        groups=["default"],
    ),
    # ModelConfig(
    #     model_name="flexolmo-5x7B-olmo3_sft_all-0.05-1e-4",
    #     short_description="",
    #     long_description="",
    #     groups=["default"],
    # ),
    # ModelConfig(
    #     model_name="flexolmo-5x7B-olmo3_sft_4-math_rl-0.05-1e-4",
    #     short_description="",
    #     long_description="",
    #     groups=["default"],
    # ),
    # ModelConfig(
    #     model_name="flexolmo-5x7B-olmo3_sft_4-code_rl-0.05-1e-4",
    #     short_description="",
    #     long_description="",
    #     groups=["default"],
    # ),
    # ModelConfig(
    #     model_name="flexolmo-5x7B-olmo3_sft_3-math_code_rl-0.05-1e-4",
    #     short_description="",
    #     long_description="",
    #     groups=["default"],
    # ),
    #####################
    ##### BASELINES #####
    #####################
    ModelConfig(
        model_name="olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr_step_350",
        short_description="Olmo 7B Baselines - General",
        long_description="",
        groups=["default"],
    ),
    # ModelConfig(
    #     model_name="grpo_math_only_flex-base-7b-mixed-all-sft-6e-7_step_400",
    #     short_description="Olmo 7B Baselines - Math RL'd",
    #     long_description="",
    #     groups=["default"],
    # ),
    ModelConfig(
        model_name="grpo_math_only_flex-base-7b-final-6e-7_step_400",
        short_description="Olmo 7B Baselines - Math",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="grpo_code_only_flex-base-7b-ol3_code-6e-7-unf_step_100",
        short_description="Olmo 7B Baselines - Code",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="olmo2-7b-BASE-general-olmo3_tool_use-mix-2",
        short_description="Olmo 7B Baselines - Tool Use",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="olmo2-7b-BASE-general-olmo3_safety-mix-2",
        short_description="Olmo 7B Baselines - Safety",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-5e-4-4-active",
        short_description="Olmo 7B Baselines - Safety",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-5e-4-3-active",
        short_description="Olmo 7B Baselines - Safety",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-5e-4-2-active",
        short_description="Olmo 7B Baselines - Safety",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-5e-4-1-active",
        short_description="Olmo 7B Baselines - Safety",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.25-1e-4-4-active",
        short_description="Olmo 7B Baselines - Safety",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.25-1e-4-3-active",
        short_description="Olmo 7B Baselines - Safety",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-1.0-1e-4-4-active",
        short_description="Olmo 7B Baselines - Safety",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-1.0-1e-4-3-active",
        short_description="Olmo 7B Baselines - Safety",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="grpo_math_only_flex-base-7b-mixed-all-sft-6e-7_step_200",
        short_description="Full post-train baseline",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="BTX-5x7B-Test-5-Domains-0.05-1e-4",
        short_description="Full btx baseline, 0.05%",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="BTX-5x7B-Test-5-Domains-1.0-1e-4",
        short_description="Full btx baseline, tool use first, 0.05%",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="BTX-5x7B-Test-5-Domains-tool-first-0.05-1e-4",
        short_description="Full btx baseline, tool use first, 0.05%",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="BTX-5x7B-Test-5-Domains-tool-first-FIXED-0.05-1e-4",
        short_description="Full btx baseline, tool use first, 0.05%",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-4x7B-math_rl-code_rl-tool_use_sft-0.05-1e-4",
        short_description="Full 4x7b, 5x7b minus safety, 0.05%",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="flexolmo-3x7B-math_rl-code_rl-0.05-1e-4",
        short_description="Full 4x7b, 5x7b minus safety and tool use, 0.05%",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="7b-merge-model-baseline-5-domains",
        short_description="merge baseline with mid-training",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="7b-merge-model-baseline-5-domains-test-manual-latest",
        short_description="merge baseline with mid-training",
        long_description="",
        groups=["default"],
    ),
    ### MODELS TO ADD:
    # 7b math and code without anneal (SFT and RL)
    # merged 7b (with RL, no mid-training)
    # upper bound model (with SFT and RL)
    # different amounts of router training
    # router training LR sweep?
    ModelConfig(
        model_name="BTX-5x7B-Test-5-Domains-tool-first-FIXED-0.05-1e-4-4-active",
        short_description="Fixed BTX with fewer active experts",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="BTX-5x7B-Test-5-Domains-tool-first-FIXED-0.05-1e-4-3-active",
        short_description="Fixed BTX with fewer active experts",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="BTX-5x7B-Test-5-Domains-tool-first-FIXED-0.05-1e-4-2-active",
        short_description="Fixed BTX with fewer active experts",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="BTX-5x7B-Test-5-Domains-tool-first-FIXED-0.05-1e-4-1-active",
        short_description="Fixed BTX with fewer active experts",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="olmo2-7b-CONTINUED-mixed_all_sft-fixed",
        short_description="Continued pretraining baseline - SFT only",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="FlexOlmo-5x7B-final-math_sft-FINAL-0.05-1e-4",
        short_description="Upgrade baseline - math sft",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="grpo_math_only_flex-base-mixed-anneal-7b-DPO-olmo2-6e-7_step_100",
        short_description="7B retrain from base baseline RLVR",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="grpo_math_only_flex-base-BASE-fixed-7b-DPO-olmo2-6e-7_step_100",
        short_description="7B retrain from base baseline RLVR (no mid-train)",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7_step_300",
        short_description="7B continued post-train (no mid-train)",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="FlexOlmo-5x7B-final-0.01-1e-4",
        short_description="Final model RT percent sweep",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="FlexOlmo-5x7B-final-0.1-1e-4",
        short_description="Final model RT percent sweep",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="FlexOlmo-5x7B-final-0.25-1e-4",
        short_description="Final model RT percent sweep",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="FlexOlmo-5x7B-final-0.5-1e-4",
        short_description="Final model RT percent sweep",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="FlexOlmo-5x7B-final-0.75-1e-4",
        short_description="Final model RT percent sweep",
        long_description="",
        groups=["default"],
    ),
    ModelConfig(
        model_name="FlexOlmo-5x7B-final-1.0-1e-4",
        short_description="Final model RT percent sweep",
        long_description="",
        groups=["default"],
    ),
]

OUTPUT_FILE = "results.csv"
MISSING_EVALS_SCRIPT = "run_missing_evals.sh"
# ---------------------------------

TASKS = [
    "mmlu:cot::hamish_zs_reasoning_deepseek",
    "popqa",
    "simpleqa",
    "bbh:cot::hamish_zs_reasoning",
    "gpqa",
    "zebralogic",
    "agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek",
    "minerva_math::hamish_zs_reasoning_deepseek",
    "gsm8k",
    "omega_500",
    "codex_humanevalplus",
    "mbppplus",
    "livecodebench_codegeneration",
    "alpaca_eval",
    "ifeval",
    "ifeval_ood",
    "bfcl_all::std",
    "do_anything_now:default",
    "harmbench:default",
    "trustllm_jailbreaktrigger:default",
    "wildguardtest:default",
    "wildjailbreak:benign",
]

# Category averages: each maps a display name to the task keys that compose it.
# If ANY task in a category is missing, the average is "n/a".
CATEGORY_AVERAGES = OrderedDict([
    ("Knowledge Average", [
        "mmlu:cot::hamish_zs_reasoning_deepseek",
        "popqa",
        "simpleqa",
    ]),
    ("Reasoning Average", [
        "bbh:cot::hamish_zs_reasoning",
        "gpqa",
        "zebralogic",
        "agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek",
    ]),
    ("Chat Average", [
        "alpaca_eval",
        "ifeval",
    ]),
    ("Math Average", [
        "gsm8k",
        "minerva_math::hamish_zs_reasoning_deepseek",
    ]),
    ("Code Average", [
        "codex_humanevalplus",
        "mbppplus",
    ]),
    ("Tool Use Average", [
        "bfcl_all::std",
    ]),
    ("Safety Average", [
        "do_anything_now:default",
        "harmbench:default",
        "trustllm_jailbreaktrigger:default",
        "wildguardtest:default",
        "wildjailbreak:benign",
    ]),
])

AVERAGE_FIELDS = ["Overall Average"] + list(CATEGORY_AVERAGES.keys())

META_FIELDS = ["model_name", "short_description", "long_description", "model_path", "group"]

# ---------------------------------------------------------------------------
# Eval submission config — maps datalake task names to eval submission strings
# ---------------------------------------------------------------------------
DATALAKE_TO_EVAL_TASK = {
    "mmlu:cot::hamish_zs_reasoning_deepseek": "mmlu:cot::hamish_zs_reasoning_deepseek",
    "popqa": "popqa::hamish_zs_reasoning_deepseek",
    "simpleqa": "simpleqa::tulu-thinker_deepseek",
    "bbh:cot::hamish_zs_reasoning": "bbh:cot::hamish_zs_reasoning",
    "gpqa": "gpqa:0shot_cot::qwen3-instruct",
    "zebralogic": "zebralogic::hamish_zs_reasoning_deepseek",
    "agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek": "agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek",
    "minerva_math::hamish_zs_reasoning_deepseek": "minerva_math::hamish_zs_reasoning_deepseek",
    "gsm8k": "gsm8k::zs_cot_latex_deepseek",
    "omega_500": "omega_500:0-shot-chat_deepseek",
    "codex_humanevalplus": "codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek",
    "mbppplus": "mbppplus:0-shot-chat::tulu-thinker_deepseek",
    "livecodebench_codegeneration": "livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags",
    "alpaca_eval": "alpaca_eval_v3::hamish_zs_reasoning_deepseek",
    "ifeval": "ifeval::hamish_zs_reasoning_deepseek",
    "ifeval_ood": "ifeval_ood::tulu-thinker",
    "bfcl_all::std": "bfcl_all::std",
    "do_anything_now:default": "do_anything_now::default",
    "harmbench:default": "harmbench::default",
    "trustllm_jailbreaktrigger:default": "trustllm_jailbreaktrigger::default",
    "wildguardtest:default": "wildguardtest::default",
    "wildjailbreak:benign": "wildjailbreak::benign",
}

SAFETY_TASKS = {
    "do_anything_now:default", "harmbench:default",
    "trustllm_jailbreaktrigger:default", "wildguardtest:default",
    "wildjailbreak:benign",
}
MATH_2GPU_TASKS = {"minerva_math::hamish_zs_reasoning_deepseek"}

COMMON_SUBMIT_ARGS = (
    "--cluster ai2/saturn ai2/ceres "
    "--is_tuned "
    "--workspace ai2/flex2 "
    "--priority urgent "
    "--preemptible "
    "--use_hf_tokenizer_template "
    "--run_oe_eval_experiments "
    "--evaluate_on_weka "
    "--run_id placeholder "
    "--oe_eval_max_length 4096 "
    "--process_output r1_style "
    "--skip_oi_evals"
)


def build_submit_command(
    model_name: str,
    model_path: str,
    eval_tasks: list[str],
    beaker_image: str,
    gpu_multiplier: int | None = None,
) -> str:
    """Build a shell command string for submitting eval jobs."""
    cmd = (
        f"uv run python scripts/submit_eval_jobs.py \\\n"
        f"    --model_name {model_name} \\\n"
        f"    --location {model_path} \\\n"
        f"    {COMMON_SUBMIT_ARGS} \\\n"
        f"    --oe_eval_tasks {','.join(eval_tasks)} \\\n"
        f"    --beaker_image {beaker_image}"
    )
    if gpu_multiplier:
        cmd += f" \\\n    --gpu_multiplier {gpu_multiplier}"
    return cmd


def generate_missing_evals_commands(
    model_name: str,
    short_description: str,
    model_path: str,
    missing_tasks: list[str],
) -> list[str]:
    """Given a list of missing datalake task names, return shell command strings."""
    if not model_path:
        print(f"  WARNING: no model_path for {model_name}, can't generate eval commands")
        return []

    regular, math, safety = [], [], []
    for datalake_task in missing_tasks:
        eval_task = DATALAKE_TO_EVAL_TASK.get(datalake_task)
        if eval_task is None:
            print(f"  WARNING: no eval mapping for '{datalake_task}', skipping")
            continue
        if datalake_task in SAFETY_TASKS:
            safety.append(eval_task)
        elif datalake_task in MATH_2GPU_TASKS:
            math.append(eval_task)
        else:
            regular.append(eval_task)

    commands = []
    if regular:
        commands.append(f"# {short_description} — regular tasks ({len(regular)} missing)")
        commands.append(build_submit_command(model_name, model_path, regular, "jacobm/oe-eval-flex-olmo-9-29-5"))
    if math:
        commands.append(f"# {short_description} — math tasks ({len(math)} missing)")
        commands.append(build_submit_command(model_name, model_path, math, "jacobm/oe-eval-flex-olmo-9-29-5", gpu_multiplier=2))
    if safety:
        commands.append(f"# {short_description} — safety tasks ({len(safety)} missing)")
        commands.append(build_submit_command(model_name, model_path, safety, "maliam/flexolmo-libraries-safety", gpu_multiplier=2))
    return commands


def find_experiments(model_name: str) -> list[dict]:
    response = requests.get(
        f"{BASE_URL}/bluelake/find-experiments/",
        params={
            "from_created_dt": FROM_DATE,
            "model_name": model_name,
            "limit": 10_000,
            "return_fields": "experiment_id,model_name,tags",
        },
        headers={"accept": "application/json"},
    )
    response.raise_for_status()
    return response.json()


def get_metrics(experiment_id: str) -> tuple[list[dict], str | None]:
    """Fetch metrics for an experiment.

    Returns (task_entries, model_path) where model_path is extracted from
    model_config if present in the response.
    """
    response = requests.get(
        f"{BASE_URL}/greenlake/metrics-all/{experiment_id}",
        headers={"accept": "application/json"},
    )
    response.raise_for_status()
    data = response.json()

    model_path = None

    # Response may be a dict with top-level model_config + tasks list,
    # or a flat list of task entries.
    if isinstance(data, dict):
        model_path = data.get("model_config", {}).get("model_path")
        task_entries = data.get("tasks", [data])  # fall back to the dict itself
    else:
        task_entries = data
        # Check if any entry has model_config
        for entry in data:
            if isinstance(entry, dict) and "model_config" in entry:
                model_path = entry["model_config"].get("model_path")
                if model_path:
                    break

    return task_entries, model_path


def compute_averages(task_scores: dict[str, float]) -> dict[str, str | float]:
    """Compute category averages and an overall average across categories.

    Returns a dict keyed by average name. A category is "n/a" if any of its
    constituent tasks is missing from task_scores. The overall average is the
    mean of the non-"n/a" category averages, or "n/a" if all categories are missing.
    """
    cat_values: dict[str, str | float] = {}
    for cat_name, task_keys in CATEGORY_AVERAGES.items():
        scores = [task_scores.get(t) for t in task_keys]
        if any(s is None or s == "" for s in scores):
            cat_values[cat_name] = "n/a"
        else:
            cat_values[cat_name] = sum(scores) / len(scores)

    valid = [v for v in cat_values.values() if v != "n/a"]
    overall = sum(valid) / len(valid) if len(valid) == len(cat_values) else "n/a"

    return {"Overall Average": overall, **cat_values}


def fetch_run_results(query_model_name: str, exact_model_name: str) -> tuple[dict[str, float], str | None]:
    """Fetch experiments and collect task scores for a single run variant.

    Args:
        query_model_name: The name to search the datalake with.
        exact_model_name: The exact model_name to filter results to.

    Returns:
        (task_scores dict, model_path or None)
    """
    experiments = find_experiments(query_model_name)
    experiments = [e for e in experiments if e["model_name"] == exact_model_name]

    task_scores: dict[str, float] = {}
    model_path: str | None = None

    for exp in experiments:
        experiment_id = exp["experiment_id"]
        try:
            metrics_list, exp_model_path = get_metrics(experiment_id)
        except requests.HTTPError as e:
            print(f"  [HTTP error] {experiment_id}: {e.response.status_code} {e.response.text[:200]}")
            continue
        except Exception as e:
            print(f"  [Error] {experiment_id}: {e}")
            continue

        if model_path is None and exp_model_path:
            model_path = exp_model_path

        for entry in metrics_list:
            task_name = entry.get("task_name", "") or entry.get("alias", "")
            score = entry.get("metrics", {}).get("primary_score")
            if task_name in task_scores and DEBUG:
                print(f"  [Warning] duplicate result for {task_name}, keeping first")
            else:
                task_scores[task_name] = score

    # Normalize alpaca_eval from 0-100 to 0-1 to match other tasks before scaling
    if "alpaca_eval" in task_scores and task_scores["alpaca_eval"] not in (None, ""):
        task_scores["alpaca_eval"] = task_scores["alpaca_eval"] / 100

    # Scale all scores from 0-1 to 0-100
    for key in task_scores:
        if task_scores[key] not in (None, ""):
            task_scores[key] = task_scores[key] * 100

    return task_scores, model_path


def average_task_scores(all_run_scores: list[dict[str, float]]) -> dict[str, float]:
    """Average task scores across runs. Only includes a task if ALL runs have it."""
    if not all_run_scores:
        return {}
    averaged: dict[str, float] = {}
    # Use the union of all task keys, but only keep tasks present in every run
    all_keys = set()
    for scores in all_run_scores:
        all_keys.update(scores.keys())

    for key in all_keys:
        values = []
        for scores in all_run_scores:
            v = scores.get(key)
            if v is None or v == "":
                break
            values.append(v)
        if len(values) == len(all_run_scores):
            averaged[key] = sum(values) / len(values)

    return averaged


def main():
    rows = []
    all_shell_commands: list[str] = []

    for config in MODEL_CONFIGS:
        base_name = config.model_name
        print(f"\n{'='*60}")
        print(f"Processing model: {base_name}")
        print(f"{'='*60}")

        # Fetch results for each run variant
        run_results: dict[str, dict[str, float]] = {}  # suffix -> task_scores
        model_path: str | None = None

        for suffix in RUN_SUFFIXES:
            run_name = f"{base_name}{suffix}"
            print(f"\n  Finding experiments for run: {run_name}")
            task_scores, run_model_path = fetch_run_results(run_name, run_name)

            if model_path is None and run_model_path:
                model_path = run_model_path

            found = [t for t in TASKS if t in task_scores]
            print(f"    Matched {len(found)}/{len(TASKS)} target tasks")

            if found:
                run_results[suffix] = task_scores
            else:
                print(f"    No results found for {run_name}, skipping")

        if model_path:
            config.model_path = model_path
            print(f"\n  Model path: {model_path}")

        # Emit rows for each run that has results
        for suffix, task_scores in run_results.items():
            run_name = f"{base_name}{suffix}"
            run_label = f" (run {suffix[-1]})" if suffix else " (run 1)"
            missing = [t for t in TASKS if t not in task_scores]

            averages = compute_averages(task_scores)

            # Generate shell commands for missing evals on this run
            if missing:
                cmds = generate_missing_evals_commands(
                    run_name, f"{config.short_description}{run_label}", config.model_path, missing,
                )
                all_shell_commands.extend(cmds)

            for group in config.groups:
                rows.append({
                    "model_name": run_name,
                    "short_description": f"{config.short_description}{run_label}",
                    "long_description": config.long_description,
                    "model_path": config.model_path,
                    "group": group,
                    **averages,
                    **{t: task_scores.get(t, "") for t in TASKS},
                })

        if COMPUTE_AVERAGE_ROWS:
            # Compute and emit the average row if all 3 runs exist
            if len(run_results) == len(RUN_SUFFIXES):
                print(f"\n  All {len(RUN_SUFFIXES)} runs found — computing average row")
                avg_scores = average_task_scores(list(run_results.values()))
                averages = compute_averages(avg_scores)

                for group in config.groups:
                    rows.append({
                        "model_name": f"{base_name}-average",
                        "short_description": f"{config.short_description} (avg of {len(RUN_SUFFIXES)} runs)",
                        "long_description": config.long_description,
                        "model_path": config.model_path,
                        "group": group,
                        **averages,
                        **{t: avg_scores.get(t, "") for t in TASKS},
                    })
            else:
                present = [s if s else "(base)" for s in run_results.keys()]
                print(f"\n  Only {len(run_results)}/{len(RUN_SUFFIXES)} runs found ({', '.join(present)}), skipping average row")

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS + AVERAGE_FIELDS + TASKS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_FILE}")

    # Write shell script for missing evals
    if all_shell_commands:
        script = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n\n".join(all_shell_commands) + "\n"
        with open(MISSING_EVALS_SCRIPT, "w") as f:
            f.write(script)
        os.chmod(MISSING_EVALS_SCRIPT, 0o755)
        n_cmds = len([c for c in all_shell_commands if not c.startswith("#")])
        print(f"Wrote {MISSING_EVALS_SCRIPT} with {n_cmds} submit commands")
    else:
        print("All evals complete — no missing evals script needed")


if __name__ == "__main__":
    main()