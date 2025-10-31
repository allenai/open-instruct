#!/bin/bash

search_strings=(
    "permissive-lr-8e-5-seed-dpo-dolci-keyfilt_lb10__42__1761774506"
    "permissive-lr-8e-5-seed-dpo-dolci-keyfilt_lb20__42__1761774601"
    "permissive-lr-8e-5-seed-dpo-dolci-keyfilt_lb30__42__1761791282"
    "permissive-seed-dpo-dolci-qwen4b-lr1e-6__42__1761815002"
    "permissive-seed-dpo-dolci-qwen4b-lr7e-7__42__1761817305"
    "permissive-seed-dpo-dolci-qwen8b-lr1e-6__42__1761816189"
    "permissive-seed-dpo-dolci-qwen8b-lr7e-7__42__1761817305"
    "permissive-seed-dpo-dolci-qwen14b-lr1e-6__42__1761813575"
    "permissive-seed-dpo-dolci-qwen14b-lr7e-7__42__1761814714"
    "permissive-lr-8e-5-seed11235-dpo-olmo2-1e-6__42__1760713743"
    "permissive-lr-8e-5-seed11235-dpo-only_delta__42__1760591141"
    "permissive-lr-8e-5-seed11235-dpo-only_gpt__42__1760591050"
    "permissive-lr-8e-5-seed11235-dpo-only_deltagpt__42__1760641981"
    # "permissive-seed-dpo-dolci-qwen1.7b_reject-lr1e-6__42__1761893834"
)

output_dir="/weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/data/scott"

# Refresh DuckDB cache once first to avoid lock conflicts
echo "Refreshing DuckDB cache once before parallel execution..."
python /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/download_and_analyze.py "${search_strings[0]}" --output-dir $output_dir/"${search_strings[0]}"

# Now run the rest in parallel (they'll use the fresh cache)
for search_string in "${search_strings[@]:1}"; do
   python /weka/oe-adapt-default/saurabhs/repos/open-instruct-evals/download_evals_analyze_lengths/download_and_analyze.py $search_string --output-dir $output_dir/$search_string --no-refresh &
done

wait