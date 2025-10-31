datasets=(
    "01K7VW6NXREPT1RY0QHM1QSWFG"
    "01K7WVCJTVG93M0ZPG1R7GG55D"
)

concurrency=128


for dataset in "${datasets[@]}"; do
    beaker dataset fetch $dataset --output /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/datasets/$dataset --concurrency $concurrency
done


datasets_paths="/weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/datasets/01K7VW6NXREPT1RY0QHM1QSWFG /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/datasets/01K7WVCJTVG93M0ZPG1R7GG55D"
labels="grpo_math_mix8k_p64_4_F_olmo3-r-nonreason-sft-lc-permissive-dpo__1__1760644496_step_700-on-bbh grpo_math_mix8k_p64_4_F_olmo3-r-nonreason-sft-lc-permissive-dpo__1__1760644496_step_850-on-bbh"

python download_evals_analyze_lengths/download_and_analyze.py --output-dir length_analyses/faeze5/aime --dataset-paths $datasets_paths --labels $labels