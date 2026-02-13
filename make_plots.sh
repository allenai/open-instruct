#!/usr/bin/env bash
mkdir -p length-investigation/plots

for f in length-investigation/counts/*.json; do
    name=$(basename "$f" .json)
    echo "Plotting $name..."
    uv run python plot_lengths.py "$f" \
        --output "length-investigation/plots/${name}.png" \
        --max-tokens 60000 \
        --bins 1000
done

echo "Done! Plots saved to length-investigation/plots/"
