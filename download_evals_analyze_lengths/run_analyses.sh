search_strings=(
    "permissive-lr-8e-5-seed-dpo-dolci-keyfilt_lb10__42__1761774506"
    "permissive-lr-8e-5-seed-dpo-dolci-keyfilt_lb20__42__1761774601"
    "permissive-lr-8e-5-seed-dpo-dolci-keyfilt_lb30__42__1761791282"
)

for search_string in "${search_strings[@]}"; do
   python download_evals_analyze_lengths/download_and_analyze.py $search_string
done