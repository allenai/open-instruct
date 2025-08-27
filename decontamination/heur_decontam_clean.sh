python heur_decontam_clean.py \
  --input_dir olmo3_usable_decontam \
  --index_type text \
  --per_eval_counts_tsv out/per_eval_old_vs_new.tsv \
  --per_train_union_counts_tsv out/per_train_union_old_vs_new.tsv \
  --pivot_new_tsv out/pivot_new.tsv \
  --kept_removed_ids_dir out/id_lists \
  --decontaminate \
  --workers 16