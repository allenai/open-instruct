python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=ai2-llm&wpn=open_instruct_public&xaxis=_step&ceik=chat_template_name&cen=chat_template_name&metrics=train_loss&metrics=learning_rate&metric_names=Training Loss&metric_names=Learning Rate' \
        'tulu?tag=no-tag-679-g2d47f44&tag=tulu3_8b_sft&cl=Tulu3 8B SFT' \
    --env-ids 'tulu' \
    --env-ids_str 'Tulu3 8B SFT' \
    --pc.time_unit h \
    --pc.ncols 2 \
    --pc.ncols-legend 1 \
    --pc.colors "#105257" "#F0529C" "#B11BE8" "#0FCB8C" \
    --output-filename open_instruct/tulu3_8b_sft \
    --scan-history \
    --no-check-empty-runs \
    --wandb-entity-name ai2-llm \
    --wandb-project-name open_instruct_public \
    --report


