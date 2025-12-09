#!/bin/bash
# Monitor Beaker experiments

set -e

while true; do
    clear
    echo "=== Beaker Experiment Monitor ==="
    echo "Time: $(date)"
    echo ""

    # Get all running experiments for this user
    experiments=$(beaker experiment list --author finbarrt --format json 2>/dev/null | jq -r '.[0:10] | .[] | .id' 2>/dev/null || echo "")

    if [ -z "$experiments" ]; then
        echo "No recent experiments found"
    else
        echo "Recent experiments:"
        echo "-------------------"
        for exp_id in $experiments; do
            status=$(beaker experiment get "$exp_id" 2>&1 | tail -1 | awk '{print $NF}')
            name=$(beaker experiment get "$exp_id" --format json 2>/dev/null | jq -r '.name // "unnamed"' 2>/dev/null || echo "unknown")
            created=$(beaker experiment get "$exp_id" 2>&1 | tail -1 | awk '{print $4}')
            echo "  $exp_id: $status ($created)"
        done
    fi

    echo ""
    echo "Press Ctrl+C to exit. Refreshing in 30s..."
    sleep 30
done
